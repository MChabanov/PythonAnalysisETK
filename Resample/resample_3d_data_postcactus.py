#!/usr/bin/env python3
"""Resample an arbitrary 2D plane out of 3D Einstein Toolkit output into HDF5.

postcactus backend. This is the 3D counterpart of
``resample_2d_data_postcactus.py``: instead of reading data Carpet already
wrote as a 2D slice, it reads the full 3D AMR output and interpolates it onto a
2D grid lying in a user-specified plane - given by a **normal vector** and a
point on it, so the plane need not be axis-aligned.

Everything else is deliberately identical to the 2D pipeline: the same YAML
config, the same MPI structure (parallel scan, parallel iteration query, warm
SimDir pickle, chunked resampling and merge), and - most importantly - the same
on-disk schema. The output files are therefore read by ``read_data.py`` and
rendered by ``../Frames/make_frames.py`` **without any change**: the in-plane
coordinates ``(u, v)`` are stored as the ``x`` and ``y`` axes.

Usage
-----
    mpirun -n <N> python resample_3d.py config_3d.yaml
    mpirun -n <N> python resample_3d_data_postcactus.py config_3d.yaml

See ``config_example_3d.yaml`` for all options.

Why this is not just ``read(geom=...)`` on 3D data
--------------------------------------------------
postcactus can resample a grid hierarchy onto a *regular, axis-aligned* box
(``CompData.sample``), which is what the 2D backend uses. A tilted plane is
neither, so its points have to be interpolated individually. Doing that through
``CompData.interp_mlinear`` (one Python call per point) is hopeless for a
million points, and reading the hierarchy at all is worse: one iteration of one
variable of a production 3D run is several GB, essentially all of it nowhere
near the plane.

So this backend walks the AMR components itself, mirroring what
``GridReader._read_sampled`` does for boxes:

1. For each component, read only its **grid metadata** (origin/spacing/shape
   from the HDF5 attributes - no field data) and ask
   :meth:`~plane_geom.PlaneSpec.uv_window` which corner of the ``(u, v)`` grid
   could possibly land inside it. Components the plane misses are skipped
   without ever touching their data, which is nearly all of them.
2. For the ones it hits, read *only* the bounding box of the sample points that
   fall inside (``read_comp(bbox=...)`` turns that into an HDF5 hyperslab),
   padded by the interpolation stencil.
3. Interpolate that block to those points in one vectorised call
   (``RegData.sample_generic`` -> ``scipy.ndimage.map_coordinates``) and write
   them into the output.

Levels are processed coarse to fine so finer data overwrites coarser, and a
component only claims points inside its *interior* (ghost zones are read, for
the interpolation stencil, but belong to a neighbour) - the same ownership rule
as postcactus' ``RegData.sample_intersect``. Points no level covers keep
``outside_value``.

Spatial derivatives
-------------------
The config's optional ``derivatives`` block (see ``config_example_3d.yaml``)
adds variables such as ``dBx_dx`` or ``d2gxx_dxdy``: derivatives of a 3D field
with respect to the simulation coordinates x, y, z, cut onto the plane like any
other variable - one output file each, same schema. They are evaluated *before*
interpolation, on each component's native grid, by centred finite differences
(``derivatives_3d.py``): step 2 reads the hyperslab with the halo widened by the
stencil half-width, the derivative is taken on that block and trimmed to the
points where the whole stencil fits, and only then interpolated in step 3. So
the plane carries interpolated derivatives, never derivatives of interpolated
data, and a stencil never straddles two refinement levels.

The ownership rule leaves a sample point at least ``nghost - 0.5`` cells inside
the component's data, which is exactly enough for a stencil half-width of 2
(accuracy 4) with ``nghost = 3``. A wider stencil than the ghost zones allow
shrinks the region each component claims instead, so the next coarser level
fills the band; this is logged once per variable.

Component-geometry cache
------------------------
Step 1 opens every component of every level just to read four attributes - per
slice, thousands of small reads, most of them for components the plane misses.
But all grid functions of a Carpet run share one grid hierarchy: component c of
level l in ``Bx.xyz.file_12.h5`` has the same origin, spacing, shape and ghost
zones as in ``gxx.xyz.file_12.h5`` at the same iteration. So the geometry is
cached per iteration under a key that does not involve the variable (output
directory, file suffix, level, component), and a rank resampling several
variables at one iteration - the normal case with ``resample_chunks`` - pays for
the metadata pass once instead of once per variable.

The assumption is checked wherever it matters: every component the plane hits
is opened anyway to be read, and its real attributes are compared with the
cached ones. On a mismatch the slice is redone without the cache, the variable
is excluded from it for the rest of the run, and a warning is logged. Set
``geometry_cache: no`` to switch it off. It costs ~1-2 MB per cached iteration.

Measured on the 30M BBH run (xy plane, 800x800, one rank): bit-identical output,
~1 s saved per slice once the files are in the page cache (4.2 -> 3.1 s), and
no measurable gain on cold slices (35-120 s under load). Carpet stores every
component as ONE gzip chunk, so reading even a thin hyperslab reads and
inflates the whole component - the "MiB read" in the log counts only the slab
kept.

Where a cold slice's time goes
------------------------------
Profiled on the same run (xy plane, one rank, 40-70 s per cold slice): the
data reads are only 6-8 s. The rest is per-file metadata on Lustre, paid by
every (variable, iteration) slice because each variable has its own 64
per-process files per output directory:

* 14-34 s parsing the files' tables of contents. The SimDir pickle carries the
  parsed TOC of only the first file of each restart (all the iteration query
  needs), so ``get_level_comps`` makes postcactus list every dataset name of
  the other 63 - ~0.3 s per file, cold-Lustre latency, not an h5py
  inefficiency (a single-pass ``links.iterate`` is no faster);
* 8-15 s opening the 64 files (~0.15 s each);
* 9-14 s looking up datasets by name for the metadata pass.

Hence walking levels fine to coarse - skipping components whose points finer
levels already own, and every level coarser than the first that covers the
plane - does not pay: it was implemented and verified bit-identical (21
slices, including derivatives, accuracy 6 and the co-rotating plane), and cut
the datasets touched per xy slice by ~40%, but the ones it skips are the small
coarse ones, and cold slices on compute nodes came out ~10% *slower* in 12 of
15 pairs (2 of 2 faster on the login node) - presumably because the coarse-first
walk lets Lustre readahead serve the later fine-level reads. What does pay is
not re-parsing TOCs - next section.

Shared tables of contents
-------------------------
The files one process wrote into one output directory hold the same datasets
whatever the variable, so a TOC parsed for ``alp.xyz.file_12.h5`` serves
``gxx.xyz.file_12.h5`` too. ``toc_share.TocShare`` keeps the TOCs a rank has
parsed, keyed like the geometry cache by (directory, file suffix), and gives
them to the files of the next variable - after checking, per variable and
directory, that the file the iteration query parsed for it has the identical
TOC (``share_tocs``, default yes). A rank resampling several variables at one
iteration then parses one variable's TOCs instead of every variable's. With
``toc_cache: <path>`` they are also kept between runs: loaded at the start,
and the ones parsed are merged into the file at the end, so a later run
parses none for directories it has seen. Output is bit-identical either way;
should a shared TOC name a dataset a file lacks, the slice is redone with the
variable's own TOCs and sharing stops for it in that directory.

Measured (xy plane, one rank on a compute node, cold files, 6 + 5 pairs, all
bit-identical to the production output): a variable after the first at an
iteration took 22.2 s with shared TOCs against 33.6 s without (-34%; pairs
0.51-0.85), and the first variable at an iteration 25.4 s with the TOCs from
``toc_cache`` against 35.8 s (-29%; pairs 0.60-0.96).
"""

import fnmatch
import os
import pickle

import numpy as np
from postcactus import grid_data as gd

from resample_common import abort, comm, grid_bounds, log, rank, run
from resample_2d_data_postcactus import PostcactusBackend
from plane_geom import PlaneSpec
import derivatives_3d
import plane_motion
import toc_share
from toc_share import file_key as _file_key

# Interpolation order -> ghost cells needed around the sampled region. Same map
# postcactus uses in cactus_grid_h5.GridReader._read_sampled.
_STENCIL_HALO = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6}

# Names 3D output takes, used to catch a `simdir_exclude` that would hide the
# very files this backend reads (a 2D config usually excludes exactly these).
_SAMPLE_3D_NAMES = ("rho_b.xyz.h5", "rho_b.xyz.file_0.h5", "rho_b.file_0.h5")


class _GeometryMismatch(Exception):
    """A component's real geometry differs from the cached one."""


def _pack_geometry(geom):
    """``(shape, x0, dx, nghost)`` -> one compact float array, for the cache."""
    return np.concatenate([np.asarray(g, dtype=float) for g in geom])


def _unpack_geometry(row):
    return (row[0:3].astype(int), row[3:6].copy(), row[6:9].copy(),
            row[9:12].astype(int))


def _component_geometry(dset):
    """Return ``(shape, x0, dx, nghost)`` of one component from its attributes.

    postcactus' ``GridH5File._read_geom`` also reads the ``time`` attribute,
    which this backend does not need; skipping it matters because the metadata
    pass runs over *every* component of *every* level (thousands per iteration
    for per-process 3D output), whereas only the few the plane crosses are ever
    read. Axis order matches postcactus: the HDF5 shape reversed, i.e. x, y, z.
    """
    shape = np.array(dset.shape[::-1], dtype=int)
    nghost = dset.attrs.get("cctk_nghostzones", None)
    if nghost is None:
        nghost = np.zeros_like(shape)
    return (shape,
            np.asarray(dset.attrs["origin"], dtype=float),
            np.asarray(dset.attrs["delta"], dtype=float),
            np.asarray(nghost, dtype=int))


class Postcactus3DBackend(PostcactusBackend):
    """postcactus coupling for plane cuts through 3D data.

    Inherits the scan, the SimDir pickle, the iteration query and the warm
    state handling from :class:`~resample_2d_data_postcactus.PostcactusBackend`
    unchanged - they only care that the "plane index" is a postcactus omni
    reader, and here it is ``sd.grid.xyz`` instead of ``sd.grid.<plane>``. Only
    the read/resample step and the plane bookkeeping differ.
    """

    name = "postcactus_3d"

    def __init__(self):
        PostcactusBackend.__init__(self)
        self._axes_cache = None    # (u, v) in-plane axes; constant per run
        self._tables = None        # {var: FrameTable}, broadcast from rank 0
        self._points_cache = None  # (key, (spec, u, v, X, Y, Z)) - last frame
        self._reported = set()     # variables whose read stats were logged
        self._derivs = {}          # {output name: Derivative}, from the config
        self._geom_cache = {}      # {it: {(file key, level, comp): packed geometry}}
        self._uncacheable = set()  # fields whose geometry disagreed with the cache
        self._tocs = None          # toc_share.TocShare, if share_tocs
        self._toc_counts = [0, 0]  # TOCs [shared, parsed] by this rank
    # -- configuration -----------------------------------------------------

    def validate_config(self, cfg):
        """Parse the plane, add 3D-only defaults, and catch common mistakes.

        Only what can be checked before the simulation directory is scanned; a
        moving plane is fully resolved (and reported) in
        :meth:`build_time_tables`, which is also where ``auto`` values are read
        from the parfile.
        """
        kind = plane_motion.plane_kind(cfg["plane"])
        if kind not in ("static", "analytic", "trajectory"):
            abort("unknown plane kind %r (use static, analytic or trajectory)"
                  % kind)
        if kind == "static":
            try:
                log("Plane: %s" % self._static_spec(cfg).describe())
            except ValueError as exc:
                abort("bad `plane` for the 3D backend: %s" % exc)
        else:
            module = cfg["plane"].get("module")
            if kind == "analytic" and module is None:
                abort("an analytic plane needs `plane.module`: the path to a "
                      "Python file defining normal(t, it)")
            if module is not None and not os.path.isfile(
                    os.path.abspath(os.path.expanduser(str(module)))):
                abort("plane.module not found: %r (paths are relative to the "
                      "working directory)" % module)
            log("Plane: %s, resolved per iteration after the query" % kind)

        cfg.setdefault("outside_value", 0.0)
        cfg.setdefault("geometry_cache", True)
        cfg.setdefault("share_tocs", True)
        cfg.setdefault("toc_cache", None)
        if cfg["toc_cache"] and not cfg["share_tocs"]:
            abort("toc_cache needs share_tocs: yes (it stores the shared tables "
                  "of contents)")

        order = int(cfg["interp_order"])
        if order not in _STENCIL_HALO:
            abort("interp_order must be 0..5 for the 3D backend (got %r)"
                  % cfg["interp_order"])

        resolution, min_corner, max_corner = grid_bounds(cfg["grid"])
        if any(n < 1 for n in resolution):
            abort("grid.resolution must be >= 1 in each direction (got %r)"
                  % (resolution,))
        if any(a >= b for a, b in zip(min_corner, max_corner)):
            abort("the in-plane grid needs min < max in both directions "
                  "(got min %r, max %r)" % (min_corner, max_corner))
        log("In-plane grid: %d x %d points over u %s, v %s"
            % (resolution[0], resolution[1],
               [min_corner[0], max_corner[0]], [min_corner[1], max_corner[1]]))

        self._validate_derivatives(cfg)

        # A config copied from the 2D pipeline usually excludes 3D output.
        hidden = [p for p in cfg["simdir_exclude"]["files"]
                  if any(fnmatch.fnmatch(n, p) for n in _SAMPLE_3D_NAMES)]
        if hidden:
            abort("simdir_exclude.files would hide the 3D data this backend "
                  "reads (offending pattern(s): %s). Remove them - and make "
                  "sure simdir_exclude.dirs does not prune the 3D output "
                  "directory either." % ", ".join(repr(p) for p in hidden))

    @staticmethod
    def _validate_derivatives(cfg):
        """Expand the ``derivatives`` block into extra output variables.

        Each requested derivative is appended to ``cfg["variables"]`` under its
        output name (``dBx_dx`` ...) and recorded in ``cfg["derivative_vars"]``,
        so from here on the orchestrator treats it as an ordinary variable; only
        this backend knows to read the base field and differentiate it. The
        expanded config is what gets broadcast, so every rank sees the same.
        """
        try:
            accuracy, derivs = derivatives_3d.parse_config(cfg.get("derivatives"))
        except ValueError as exc:
            abort("bad `derivatives`: %s" % exc)
        cfg["variables"] = list(cfg["variables"] or [])
        clash = [d.name for d in derivs if d.name in cfg["variables"]]
        if clash:
            abort("derivative output name(s) %s collide with `variables`"
                  % ", ".join(map(repr, clash)))
        cfg["derivative_accuracy"] = accuracy
        cfg["derivative_vars"] = {d.name: d.as_dict() for d in derivs}
        cfg["variables"].extend(d.name for d in derivs)
        if derivs:
            log("Derivatives (centred, accuracy %d, w.r.t. simulation x/y/z): %s"
                % (accuracy, ", ".join(d.name for d in derivs)))
        if not cfg["variables"]:
            abort("nothing to do: `variables` is empty and no `derivatives` "
                  "were requested")

    def _derivative(self, cfg, var):
        """The :class:`~derivatives_3d.Derivative` behind ``var``, or None."""
        if cfg is not None:
            self._configure(cfg)
        return self._derivs.get(var)

    def _configure(self, cfg):
        """Load the derivative map from the (broadcast) config, once.

        Needed because :meth:`validate_config` runs on rank 0 only; the other
        ranks see the expanded config but not that call. Triggered from
        :meth:`plane_index`, which every rank calls before querying.
        """
        if not self._derivs and cfg.get("derivative_vars"):
            self._derivs = {
                name: derivatives_3d.Derivative(d["field"], d["axes"])
                for name, d in cfg["derivative_vars"].items()}

    def _field(self, var):
        """The 3D field actually read for output variable ``var``."""
        deriv = self._derivs.get(var)
        return deriv.field if deriv is not None else var

    def extra_output_attrs(self, cfg, var=None):
        """Record the plane, so a file is self-describing without the config.

        A static plane's geometry fits in attributes. A moving one does not:
        it is written per iteration instead (see
        :meth:`~plane_motion.FrameTable.per_iteration_data`), and a single
        normal here would be actively misleading.
        """
        kind = plane_motion.plane_kind(cfg["plane"])
        attrs = {
            "source_dims": "xyz",
            "plane_kind": kind,
            "outside_value": float(cfg["outside_value"]),
        }
        if kind == "static":
            spec = self._static_spec(cfg)
            attrs.update({
                "plane": spec.label,
                "plane_normal": spec.n_hat,
                "plane_origin": spec.origin,
                "plane_u_axis": spec.u_hat,
                "plane_v_axis": spec.v_hat,
            })
        else:
            attrs["plane"] = kind
        deriv = self._derivative(cfg, var)
        if deriv is not None:
            attrs.update({
                "derivative_of": deriv.field,
                "derivative_axes": deriv.letters,
                "derivative_accuracy": int(cfg["derivative_accuracy"]),
                "derivative_method": "centred finite differences on the native "
                                     "AMR grid, then interpolated",
            })
        return attrs

    # -- geometry ----------------------------------------------------------

    @staticmethod
    def _static_spec(cfg):
        """The single :class:`PlaneSpec` of a static plane."""
        plane = cfg["plane"]
        if isinstance(plane, dict):
            plane = {k: val for k, val in plane.items() if k != "kind"}
        return PlaneSpec.from_config(plane)

    def _axes(self, cfg):
        """The in-plane ``(u, v)`` axes - the output's ``x``/``y``."""
        if self._axes_cache is None:
            resolution, min_corner, max_corner = grid_bounds(cfg["grid"])
            self._axes_cache = (
                np.linspace(min_corner[0], max_corner[0], resolution[0]),
                np.linspace(min_corner[1], max_corner[1], resolution[1]))
        return self._axes_cache

    # -- the precomputed frame table ---------------------------------------

    def build_time_tables(self, cfg, var_iters, sim):
        """Rank 0: the plane's frame at every output iteration of every variable.

        Done here rather than per slice because ranks resample iterations out of
        order, so a frame can never be derived from its predecessor - and
        because having every frame at once is what lets continuity be imposed
        and verified before any field data is read.
        """
        trajectory = None
        if plane_motion.plane_kind(cfg["plane"]) == "trajectory":
            try:
                trajectory = plane_motion.Trajectory(
                    plane_motion.resolve_trajectory_path(
                        cfg["plane"].get("file", "auto"), sim, log))
            except ValueError as exc:
                abort("plane: %s" % exc)
            log("  trajectory: %s (%s)"
                % (trajectory.path, ", ".join(trajectory.available())))

        tables = {}
        for var, (iters, times) in var_iters.items():
            log("Plane frames for %s (%d iteration(s), t = %g .. %g):"
                % (var, len(iters), times[0], times[-1]))
            try:
                tables[var] = plane_motion.build_frame_table(
                    cfg["plane"], iters, times, sim=sim, log=log,
                    trajectory=trajectory)
            except ValueError as exc:
                abort("plane: %s" % exc)
        return tables

    def set_time_tables(self, tables):
        self._tables = tables

    def per_iteration_data(self, cfg, var, iters):
        return self._table_for(cfg, var).per_iteration_data(iters)

    def _table_for(self, cfg, var):
        """This variable's frame table, or a constant one for a static plane.

        The fallback lets ``read_slice`` be used directly (from a notebook or
        ``inspect_plane.py``) without going through ``run()``, which is where
        the tables normally come from.
        """
        if self._tables is not None and var in self._tables:
            return self._tables[var]
        if plane_motion.plane_kind(cfg["plane"]) != "static":
            raise RuntimeError(
                "no precomputed plane frames for %r; a %s plane needs the "
                "frame table built by run() (Backend.build_time_tables)"
                % (var, plane_motion.plane_kind(cfg["plane"])))
        spec = self._static_spec(cfg)
        return plane_motion.FrameTable([0], [0.0], [spec.origin], [spec.u_hat],
                                       [spec.v_hat], [spec.n_hat],
                                       label=spec.label, moving=False)

    def _frame(self, cfg, var, it):
        """``(spec, u, v, X, Y, Z)`` for one iteration, cached where constant."""
        table = self._table_for(cfg, var)
        key = (var, int(it)) if table.moving else "static"
        if self._points_cache is not None and self._points_cache[0] == key:
            return self._points_cache[1]

        u, v = self._axes(cfg)
        spec = table.spec_at(it) if table.moving else table.spec_at(
            table.iterations[0])
        value = (spec, u, v) + spec.points(u, v)
        self._points_cache = (key, value)
        return value

    def plane_index(self, sim, cfg):
        # The 3D omni reader. Same object type as sd.grid.<plane> in the 2D
        # backend, so the inherited query/warm-state code applies unchanged.
        self._configure(cfg)
        self._setup_tocs(cfg)
        return sim.grid.xyz

    def _setup_tocs(self, cfg):
        """Create the TOC share (and load its cache file), once per rank."""
        if self._tocs is not None or not cfg.get("share_tocs", True):
            return
        self._tocs = toc_share.TocShare()
        path = cfg.get("toc_cache")
        if path:
            try:
                n = self._tocs.load(path)
            except (OSError, ValueError, EOFError, pickle.UnpicklingError) as exc:
                log("  WARNING: ignoring TOC cache %s: %s" % (path, exc))
            else:
                log("TOC cache %s: %d table(s) of contents loaded"
                    % (path, n))

    def finish(self, cfg):
        """All ranks: report TOC sharing; merge the parsed TOCs into the cache."""
        if self._tocs is None:
            return
        counts = comm.gather(self._toc_counts, root=0)
        path = cfg.get("toc_cache")
        parts = comm.gather(self._tocs.parsed(), root=0) if path else None
        if rank != 0:
            return
        shared = sum(c[0] for c in counts)
        parsed = sum(c[1] for c in counts)
        log("Tables of contents: %d parsed, %d shared (%.0f%%)"
            % (parsed, shared, 100.0 * shared / max(shared + parsed, 1)))
        if path:
            try:
                dirs, files = toc_share.save(path, parts)
            except (OSError, ValueError, EOFError, pickle.UnpicklingError) as exc:
                log("  WARNING: could not update TOC cache %s: %s" % (path, exc))
            else:
                log("TOC cache %s: %d table(s) of contents in %d director(ies) "
                    "added or refreshed" % (path, files, dirs))

    # A derivative variable has the iterations and the cached file state of
    # the field it is taken of; the reader only knows the field's name.

    def query(self, plane_index, var):
        return PostcactusBackend.query(self, plane_index, self._field(var))

    def extract_warm_state(self, plane_index, variables):
        fields = list(dict.fromkeys(self._field(v) for v in variables))
        return PostcactusBackend.extract_warm_state(self, plane_index, fields)

    # -- read + resample ---------------------------------------------------

    def read_slice(self, plane_index, var, it, cfg):
        """Interpolate one iteration of ``var`` onto the plane.

        For a derivative variable, the base field is read and differentiated
        per component before interpolation (see the module docstring).
        """
        spec, u, v, xs, ys, zs = self._frame(cfg, var, it)
        order = int(cfg["interp_order"])
        deriv = self._derivative(cfg, var)
        field = self._field(var)
        if deriv is None:
            fd_trim = np.zeros(3, dtype=int)
            accuracy = None
        else:
            accuracy = int(cfg["derivative_accuracy"])
            fd_trim = derivatives_3d.trim(deriv.axes, accuracy)
        # The interpolation stencil sits on top of the finite-difference one.
        halo = _STENCIL_HALO[order] + fd_trim

        out = np.full((u.size, v.size), float(cfg["outside_value"]))

        src, cut = plane_index._get_src(field)
        if any(c is not None for c in cut):
            raise RuntimeError(
                "%r resolves to a lower-dimensional source for this plane; "
                "the 3D backend needs genuine 3D output (*.xyz.h5 / "
                "*.file_*.h5)" % var)
        if not hasattr(src, "_get_files"):
            raise RuntimeError(
                "%r is served by %s, which has no per-component reader; the "
                "3D backend needs HDF5 output"
                % (var, type(src).__name__))

        # Report once per variable how much of the 3D data the plane needed -
        # the quickest way to spot a mis-specified plane. `covered` tracks which
        # output points some level reached; it is only allocated for that one
        # slice, so the steady state carries no extra cost.
        report = var not in self._reported
        files = src._get_files(field, it)
        args = (src, files, field, it, spec, u, v, xs, ys, zs, order, halo,
                deriv, accuracy, fd_trim, report)

        tocs = self._tocs if cfg.get("share_tocs", True) else None
        n_shared = tocs.prepare(field, files) if tocs is not None else 0
        n_parsed = sum(f._toc is None for f in files)
        try:
            stats = self._composite_checked(out, cfg, var, field, it, args)
        except KeyError as exc:
            # h5py: a dataset some TOC lists is not in the file. Only ours to
            # handle if the TOC was a shared one.
            if tocs is None or not tocs.reject(field, files):
                raise
            print("[rank %d] %s: WARNING a shared table of contents does not "
                  "match %s's files at it=%d (%s); %s parses its own in that "
                  "directory from now on" % (rank, var, field, it, exc, field),
                  flush=True)
            out.fill(float(cfg["outside_value"]))
            n_parsed = sum(f._toc is None for f in files)
            n_shared = 0
            stats = self._composite_checked(out, cfg, var, field, it, args)
        if tocs is not None:
            tocs.contribute(field, files)
        self._toc_counts[0] += n_shared
        self._toc_counts[1] += n_parsed

        if report:
            self._reported.add(var)
            print("[rank %d] %s: plane touches %d of %d component(s) at it=%d, "
                  "%.1f MiB read, %.1f%% of the grid covered, geometry of %d "
                  "from the cache, tables of contents: %d shared, %d parsed"
                  % (rank, var, stats["read"], stats["scanned"], it,
                     stats["cells"] * 8 / 2 ** 20,
                     100.0 * stats["covered"].mean(), stats["cached"],
                     n_shared, n_parsed),
                  flush=True)
            if stats["shrunk"]:
                print("[rank %d] %s: WARNING the %d-cell stencil is wider than "
                      "the ghost zones of %d component(s); each claims less "
                      "of the plane, so the next coarser level fills the band "
                      "(or outside_value where there is none, e.g. between "
                      "components of the coarsest level)"
                      % (rank, var, fd_trim.max(), stats["shrunk"]),
                      flush=True)
        return out

    def _composite_checked(self, out, cfg, var, field, it, args):
        """:meth:`_composite`, with the geometry cache where it holds."""
        cache = None
        if cfg.get("geometry_cache", True) and field not in self._uncacheable:
            cache = self._geom_cache.setdefault(int(it), {})
        try:
            return self._composite(out, cache, *args)
        except _GeometryMismatch as exc:
            self._uncacheable.add(field)
            print("[rank %d] %s: WARNING component geometry differs from the "
                  "cache (%s); %s is resampled without the geometry cache from "
                  "now on" % (rank, var, exc, field), flush=True)
            out.fill(float(cfg["outside_value"]))
            return self._composite(out, None, *args)

    def _composite(self, out, cache, src, files, field, it, spec, u, v,
                   xs, ys, zs, order, halo, deriv, accuracy, fd_trim, report):
        """Composite every component of one iteration onto ``out``; return stats.

        ``cache`` is this iteration's geometry cache, or None to read every
        component's attributes from the file. Raises :class:`_GeometryMismatch`
        if a cached geometry turns out to be wrong for ``field``.
        """
        stats = {"scanned": 0, "read": 0, "cells": 0, "shrunk": 0, "cached": 0,
                 "covered": np.zeros_like(out, dtype=bool) if report else None}
        # Coarse to fine, so finer refinement levels overwrite coarser ones.
        for level in sorted(src._get_levels(files, it)):
            for h5file in files:
                fkey = _file_key(h5file) if cache is not None else None
                for comp in h5file.get_level_comps(it, level):
                    self._add_component(out, h5file, field, it, level, comp,
                                        spec, u, v, xs, ys, zs, order, halo,
                                        stats, deriv, accuracy, fd_trim,
                                        cache, (fkey, level, comp))
        return stats

    @staticmethod
    def _add_component(out, h5file, var, it, level, comp,
                       spec, u, v, xs, ys, zs, order, halo, stats,
                       deriv=None, accuracy=None, fd_trim=None,
                       cache=None, key=None):
        """Composite one AMR component onto the plane, reading as little as possible.

        With ``deriv`` the plane gets that derivative of ``var`` instead,
        evaluated on the component's own grid before interpolation; ``halo``
        then already includes the stencil half-width ``fd_trim``. With
        ``cache`` (a dict) the component's geometry is taken from / stored
        under ``key`` instead of being read from the file every time.
        """
        # Metadata only - this touches the HDF5 attributes, not the field.
        stats["scanned"] += 1
        row = cache.get(key) if cache is not None else None
        if row is not None:
            shape, x0, dx, nghost = _unpack_geometry(row)
            stats["cached"] += 1
        else:
            dset = h5file._get_dataset(var, it, level, comp)
            geom = _component_geometry(dset)
            if cache is not None:
                cache[key] = _pack_geometry(geom)
            shape, x0, dx, nghost = geom
        x1 = x0 + (shape - 1) * dx

        # The region this component owns: its interior (ghost zones belong to a
        # neighbour) grown by the half cell each grid point stands for. Same
        # rule as postcactus' RegData.sample_intersect. A derivative is only
        # defined where its stencil fits inside the data, i.e. at least
        # `fd_trim` cells in; with ghost zones at least that wide (the normal
        # case) this changes nothing.
        margin = nghost - 0.5
        if fd_trim is not None and np.any(fd_trim > margin):
            margin = np.maximum(margin, fd_trim)
            stats["shrunk"] += 1
        own0 = x0 + margin * dx
        own1 = x1 - margin * dx
        if np.any(own1 <= own0):
            return
        if not spec.intersects_box(own0, own1):
            return

        window = spec.uv_window(own0, own1, u, v)
        if window is None:
            return
        i0, i1, j0, j1 = window

        # Exact test on the (small) candidate window.
        bx, by, bz = xs[i0:i1, j0:j1], ys[i0:i1, j0:j1], zs[i0:i1, j0:j1]
        inside = ((bx >= own0[0]) & (bx <= own1[0]) &
                  (by >= own0[1]) & (by <= own1[1]) &
                  (bz >= own0[2]) & (bz <= own1[2]))
        if not inside.any():
            return
        px, py, pz = bx[inside], by[inside], bz[inside]

        # A cached geometry decided that this component matters; before using
        # it to read, check it against the real one (the dataset is opened for
        # the read anyway, so this costs only the attribute reads).
        if row is not None:
            real = _pack_geometry(_component_geometry(
                h5file._get_dataset(var, it, level, comp)))
            if not np.array_equal(real, row):
                raise _GeometryMismatch("level %d component %d" % (level, comp))

        # Read just the box holding those points (plus the stencil halo, which
        # read_comp adds itself); an HDF5 hyperslab, not the whole component.
        lo = np.array([px.min(), py.min(), pz.min()])
        hi = np.array([px.max(), py.max(), pz.max()])
        block = h5file.read_comp(var, it, level, comp,
                                bbox=[lo, hi, list(np.broadcast_to(halo, 3))])
        if block is None:
            return
        stats["read"] += 1
        stats["cells"] += block.data.size

        if deriv is not None:
            # Differentiate on the native grid; the result covers the block
            # minus the stencil half-width at each end of the derivative axes.
            bdx = block.dx()
            data, cut = derivatives_3d.differentiate(block.data, bdx,
                                                     deriv.axes, accuracy)
            block = gd.RegData(block.x0() + cut * bdx, bdx, data)

        # One vectorised interpolation for all points of this component.
        values = block.sample_generic([px, py, pz], order=order, mode="nearest")

        target = out[i0:i1, j0:j1]
        target[inside] = values
        out[i0:i1, j0:j1] = target
        if stats["covered"] is not None:
            stats["covered"][i0:i1, j0:j1] |= inside


def main():
    run(Postcactus3DBackend())


if __name__ == "__main__":
    main()
