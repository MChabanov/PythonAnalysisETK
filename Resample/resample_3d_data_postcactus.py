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
"""

import fnmatch
import os

import numpy as np

from resample_common import abort, grid_bounds, log, rank, run
from resample_2d_data_postcactus import PostcactusBackend
from plane_geom import PlaneSpec
import plane_motion

# Interpolation order -> ghost cells needed around the sampled region. Same map
# postcactus uses in cactus_grid_h5.GridReader._read_sampled.
_STENCIL_HALO = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6}

# Names 3D output takes, used to catch a `simdir_exclude` that would hide the
# very files this backend reads (a 2D config usually excludes exactly these).
_SAMPLE_3D_NAMES = ("rho_b.xyz.h5", "rho_b.xyz.file_0.h5", "rho_b.file_0.h5")


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

        # A config copied from the 2D pipeline usually excludes 3D output.
        hidden = [p for p in cfg["simdir_exclude"]["files"]
                  if any(fnmatch.fnmatch(n, p) for n in _SAMPLE_3D_NAMES)]
        if hidden:
            abort("simdir_exclude.files would hide the 3D data this backend "
                  "reads (offending pattern(s): %s). Remove them - and make "
                  "sure simdir_exclude.dirs does not prune the 3D output "
                  "directory either." % ", ".join(repr(p) for p in hidden))

    def extra_output_attrs(self, cfg):
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
        return sim.grid.xyz

    # -- read + resample ---------------------------------------------------

    def read_slice(self, plane_index, var, it, cfg):
        """Interpolate one iteration of ``var`` onto the plane."""
        spec, u, v, xs, ys, zs = self._frame(cfg, var, it)
        order = int(cfg["interp_order"])
        halo = _STENCIL_HALO[order]

        out = np.full((u.size, v.size), float(cfg["outside_value"]))

        src, cut = plane_index._get_src(var)
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
        stats = {"scanned": 0, "read": 0, "cells": 0,
                 "covered": np.zeros_like(out, dtype=bool) if report else None}

        files = src._get_files(var, it)
        # Coarse to fine, so finer refinement levels overwrite coarser ones.
        for level in sorted(src._get_levels(files, it)):
            for h5file in files:
                for comp in h5file.get_level_comps(it, level):
                    self._add_component(out, h5file, var, it, level, comp,
                                        spec, u, v, xs, ys, zs, order, halo,
                                        stats)

        if report:
            self._reported.add(var)
            print("[rank %d] %s: plane touches %d of %d component(s) at it=%d, "
                  "%.1f MiB read, %.1f%% of the grid covered"
                  % (rank, var, stats["read"], stats["scanned"], it,
                     stats["cells"] * 8 / 2 ** 20,
                     100.0 * stats["covered"].mean()),
                  flush=True)
        return out

    @staticmethod
    def _add_component(out, h5file, var, it, level, comp,
                       spec, u, v, xs, ys, zs, order, halo, stats):
        """Composite one AMR component onto the plane, reading as little as possible."""
        # Metadata only - this touches the HDF5 attributes, not the field.
        stats["scanned"] += 1
        dset = h5file._get_dataset(var, it, level, comp)
        shape, x0, dx, nghost = _component_geometry(dset)
        x1 = x0 + (shape - 1) * dx

        # The region this component owns: its interior (ghost zones belong to a
        # neighbour) grown by the half cell each grid point stands for. Same
        # rule as postcactus' RegData.sample_intersect.
        own0 = x0 + (nghost - 0.5) * dx
        own1 = x1 - (nghost - 0.5) * dx
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

        # Read just the box holding those points (plus the stencil halo, which
        # read_comp adds itself); an HDF5 hyperslab, not the whole component.
        lo = np.array([px.min(), py.min(), pz.min()])
        hi = np.array([px.max(), py.max(), pz.max()])
        block = h5file.read_comp(var, it, level, comp,
                                bbox=[lo, hi, [halo] * 3])
        if block is None:
            return
        stats["read"] += 1
        stats["cells"] += block.data.size

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
