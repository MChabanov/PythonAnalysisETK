#!/usr/bin/env python3
"""Raw access to the AMR components of 3D Carpet output.

The resampling backends interpolate the components onto a surface. The
field-line tracer (``../FieldLines``) and the volume integrals
(``../Integrals``) need the grid points themselves: the components of every
refinement level that meet a box, their geometry, and the values of any
variable on an index sub-box of them. :class:`Snapshot` provides that for one
iteration, on the same postcactus file layer the 3D backend walks
(``GridReader._get_files``, ``GridH5File.get_level_comps`` /
``_get_dataset``), and shares tables of contents between variables through
:class:`toc_share.TocShare`.

Conventions
-----------
* Geometry is in x, y, z order (``shape``, ``x0``, ``dx``, ``nghost``), as in
  postcactus. Data keep the HDF5 order ``[k, j, i]`` = ``[z, y, x]``: no
  transpose, so slabs stay C-contiguous.
* A component **owns** its interior points ``nghost .. shape - 1 - nghost`` -
  the ghost zones belong to a neighbour - i.e. the box
  ``x0 + (nghost - 0.5) dx .. x1 - (nghost - 0.5) dx``. This is the ownership
  rule of the resampling backends and of postcactus'
  ``RegData.sample_intersect``. Owned boxes of one level are disjoint; where
  levels overlap the finer one takes precedence.
* All grid functions of a Carpet run share one grid hierarchy, so the geometry
  is read once per iteration, from one variable, and reused for every other
  variable read at that iteration. Each dataset opened for a read is checked
  against it.
"""

import numpy as np

import toc_share
from toc_share import file_key


class GeometryMismatch(Exception):
    """A variable's component differs from the hierarchy read for the snapshot."""


class Component(object):
    """One AMR component: geometry only, no data."""

    __slots__ = ("level", "comp", "key", "shape", "x0", "dx", "nghost")

    def __init__(self, level, comp, key, shape, x0, dx, nghost):
        self.level = int(level)
        self.comp = int(comp)
        self.key = key                      # (file key, level, comp)
        self.shape = np.asarray(shape, dtype=int)
        self.x0 = np.asarray(x0, dtype=float)
        self.dx = np.asarray(dx, dtype=float)
        self.nghost = np.asarray(nghost, dtype=int)

    @property
    def x1(self):
        """The last grid point."""
        return self.x0 + (self.shape - 1) * self.dx

    def owned_box(self):
        """``(lo, hi)`` of the region this component owns (see the module doc)."""
        margin = self.nghost - 0.5
        return self.x0 + margin * self.dx, self.x1 - margin * self.dx

    def index_range(self, lo, hi, owned=True):
        """Inclusive index range ``(i0, i1)`` of the points within ``[lo, hi]``.

        With ``owned`` only owned points count. None if there are none.
        """
        lo = np.asarray(lo, dtype=float)
        hi = np.asarray(hi, dtype=float)
        i0 = np.ceil((lo - self.x0) / self.dx - 1e-9).astype(int)
        i1 = np.floor((hi - self.x0) / self.dx + 1e-9).astype(int)
        first, last = np.zeros(3, dtype=int), self.shape - 1
        if owned:
            first, last = self.nghost, self.shape - 1 - self.nghost
        i0 = np.maximum(i0, first)
        i1 = np.minimum(i1, last)
        if np.any(i1 < i0):
            return None
        return i0, i1

    def coords(self, i0, i1):
        """The 1D x, y, z coordinates of the points ``i0 .. i1`` (inclusive)."""
        return [self.x0[d] + self.dx[d] * np.arange(i0[d], i1[d] + 1)
                for d in range(3)]

    def __repr__(self):
        return ("Component(level %d, comp %d, shape %s, x0 %s, dx %s)"
                % (self.level, self.comp, self.shape.tolist(),
                   self.x0.tolist(), self.dx.tolist()))


def boxes_overlap(lo1, hi1, lo2, hi2):
    return bool(np.all(lo1 <= hi2) and np.all(lo2 <= hi1))


class Snapshot(object):
    """The grid hierarchy of one iteration, and raw reads of any variable at it.

    ``xyz`` is the 3D omni reader (``SimDir.grid.xyz``), ``ref_field`` the
    variable whose files the hierarchy is read from (any 3D one does), and
    ``tocs`` an optional :class:`toc_share.TocShare`. Call :meth:`close` when
    done: it offers the parsed tables of contents to later snapshots and closes
    the files.
    """

    def __init__(self, xyz, it, ref_field, tocs=None):
        self.xyz = xyz
        self.it = int(it)
        self.tocs = tocs
        self._files = {}          # {var: [GridH5File]}
        self._by_key = {}         # {var: {file key: GridH5File}}
        self.components = []
        self._walk(ref_field)
        self.levels = sorted(set(c.level for c in self.components))

    # -- the hierarchy -----------------------------------------------------

    def _files_of(self, var):
        if var not in self._files:
            src, cut = self.xyz._get_src(var)
            if any(c is not None for c in cut) or not hasattr(src, "_get_files"):
                raise ValueError("%r has no 3D HDF5 output (*.xyz.file_*.h5)"
                                 % var)
            files = src._get_files(var, self.it)
            if self.tocs is not None:
                self.tocs.prepare(var, files)
            self._files[var] = files
            self._by_key[var] = {file_key(f): f for f in files}
        return self._files[var]

    def _walk(self, field):
        files = self._files_of(field)
        src, _ = self.xyz._get_src(field)
        try:
            self._walk_files(src, files, field)
        except KeyError:
            # A shared table of contents named a dataset these files lack.
            if self.tocs is None or not self.tocs.reject(field, files):
                raise
            self.components = []
            self._walk_files(src, files, field)

    def _walk_files(self, src, files, field):
        for level in sorted(src._get_levels(files, self.it)):
            for h5file in files:
                fkey = file_key(h5file)
                for comp in h5file.get_level_comps(self.it, level):
                    dset = h5file._get_dataset(field, self.it, level, comp)
                    shape, x0, dx, ng = _geometry(dset)
                    self.components.append(
                        Component(level, comp, (fkey, level, comp), shape, x0,
                                  dx, ng))

    def select(self, lo, hi, levels=None, owned=True):
        """Components whose (owned) box meets ``[lo, hi]``, coarse to fine."""
        lo = np.asarray(lo, dtype=float)
        hi = np.asarray(hi, dtype=float)
        out = []
        for c in self.components:
            if levels is not None and c.level not in levels:
                continue
            clo, chi = c.owned_box() if owned else (c.x0, c.x1)
            if boxes_overlap(clo, chi, lo, hi):
                out.append(c)
        return out

    def level_spacing(self, level):
        for c in self.components:
            if c.level == level:
                return c.dx.copy()
        raise KeyError("no component on level %d" % level)

    # -- data --------------------------------------------------------------

    def read(self, var, comp, i0, i1):
        """``var`` on the points ``i0 .. i1`` (inclusive, x/y/z) of ``comp``.

        Returns a ``[k, j, i]`` array. Raises :class:`GeometryMismatch` if
        ``var``'s dataset does not have the component's geometry.
        """
        files = self._files_of(var)
        try:
            dset = self._dataset(var, comp)
        except KeyError:
            if self.tocs is None or not self.tocs.reject(var, files):
                raise
            dset = self._dataset(var, comp)
        shape, x0, dx, _ = _geometry(dset)
        if not (np.array_equal(shape, comp.shape) and np.array_equal(x0, comp.x0)
                and np.array_equal(dx, comp.dx)):
            raise GeometryMismatch("%s level %d component %d: shape %s x0 %s, "
                                   "hierarchy has %s %s"
                                   % (var, comp.level, comp.comp, shape, x0,
                                      comp.shape, comp.x0))
        sl = tuple(slice(int(a), int(b) + 1)
                   for a, b in zip(i0[::-1], i1[::-1]))
        return np.asarray(dset[sl])

    def _dataset(self, var, comp):
        fkey, level, c = comp.key
        h5file = self._by_key[var].get(fkey)
        if h5file is None:
            raise KeyError("%s has no file matching %s" % (var, fkey))
        return h5file._get_dataset(var, self.it, level, c)

    def close(self):
        """Offer parsed tables of contents to later snapshots; close the files."""
        for var, files in self._files.items():
            if self.tocs is not None:
                self.tocs.contribute(var, files)
            for h5file in files:
                h5file.close()
        self._files.clear()
        self._by_key.clear()


def _geometry(dset):
    """``(shape, x0, dx, nghost)`` of a component dataset, x/y/z order."""
    shape = np.array(dset.shape[::-1], dtype=int)
    nghost = dset.attrs.get("cctk_nghostzones", None)
    if nghost is None:
        nghost = np.zeros_like(shape)
    return (shape, np.asarray(dset.attrs["origin"], dtype=float),
            np.asarray(dset.attrs["delta"], dtype=float),
            np.asarray(nghost, dtype=int))


def new_toc_share(path=None, log=print):
    """A :class:`toc_share.TocShare`, loaded from ``path`` if given."""
    tocs = toc_share.TocShare()
    if path:
        try:
            n = tocs.load(path)
        except (OSError, ValueError, EOFError) as exc:
            log("  WARNING: ignoring TOC cache %s: %s" % (path, exc))
        else:
            log("TOC cache %s: %d table(s) of contents loaded" % (path, n))
    return tocs
