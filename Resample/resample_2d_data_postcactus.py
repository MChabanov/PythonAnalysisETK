#!/usr/bin/env python3
"""Resample 2D slices from an Einstein Toolkit simulation into HDF5 files.

postcactus backend. It reads grid data through postcactus, resamples each
requested variable onto a fixed regular grid, and writes one HDF5 file per
variable. See ``resample_2d_data_kuibit.py`` for the equivalent kuibit backend.

All of the orchestration (MPI setup, config, pickled-SimDir cache, the serial
*and* parallel iteration query, warm-state gather, round-robin scheduling and
the HDF5 streaming output) lives in ``resample_common.py`` and is shared with
the kuibit backend. This module contains *only* the coupling to postcactus,
collected in :class:`PostcactusBackend`.

Design notes
------------
* **MPI-parallel, any rank count.** Variables are distributed round-robin
  across ranks; each rank writes its own output files (no gather-to-rank-0,
  no single-rank memory blow-up).
* **Bounded memory.** Each variable is streamed iteration-by-iteration into
  its HDF5 dataset, so a rank never holds more than one 2D slice at a time.
* **Correct per-variable iterations.** Iteration/time lists are queried per
  variable (different variables may be output at different cadences).
* **Two query modes** (config ``parallel_query``): the query can run entirely
  on rank 0 (serial) or be split across ranks (parallel), with the warmed
  SimDir reassembled on rank 0 so the saved pickle is identical either way.
* **Self-describing output.** Each file stores the iterations, physical times,
  and coordinate axes alongside the data, plus metadata as attributes.

Usage
-----
    mpirun -n <N> python resample_2d_data_postcactus.py config.yaml

(or via the launcher: ``mpirun -n <N> python resample_2d.py config.yaml`` with
``backend: postcactus`` in the config.)

See ``config_example.yaml`` for all options. Read the output with
``read_data.py`` (or any HDF5 reader).
"""

import fnmatch
import re

import numpy as np

from postcactus.simdir import SimDir
from postcactus import grid_data as gd

from resample_common import Backend, grid_bounds, run

# Directory names postcactus always excludes from its scan (see
# SimDir._scan_folders); the parallel walk must use the same set.
_STANDARD_EXCLUDES = {"SIMFACTORY", "report", "movies", "tmp", "temp"}


def _extract_array(slice_obj):
    """Return the underlying numpy array from a postcactus resampled slice."""
    data = getattr(slice_obj, "data", None)
    if data is None:
        data = np.asarray(slice_obj)
    return np.asarray(data)


class PostcactusBackend(Backend):
    """All postcactus-specific coupling for the resampling pipeline.

    The "plane index" is the omni reader ``sd.grid.<plane>``. Per-variable warm
    state lives in the underlying HDF5 ``GridReader`` reachable as
    ``omni._src[var]``: the cached ``_iters``/``_times``/``_restarts`` entries
    plus the parsed table of contents on each ``GridH5File`` in ``_vars[var]``
    (the file objects pickle safely - their ``__getstate__`` drops the open
    handle but keeps the parsed TOC).
    """

    name = "postcactus"

    def __init__(self):
        self._geom = None  # built lazily; constant for a run

    def scan(self, cfg):
        excl = cfg["simdir_exclude"]
        max_depth = int(cfg["scan_max_depth"])
        # Only pass exclusion kwargs when actually requested: older postcactus
        # builds expose SimDir(path, max_depth) without exclude_dirs/
        # exclude_files, and the common (no-exclusion) case should work there too.
        if excl["dirs"] or excl["files"]:
            return SimDir(
                cfg["simdir"], max_depth=max_depth,
                exclude_dirs=excl["dirs"] or None,
                exclude_files=excl["files"] or None,
            )
        return SimDir(cfg["simdir"], max_depth=max_depth)

    def scan_shallow(self, cfg):
        # max_depth=0 walks nothing but still captures path + SIMFACTORY parfile.
        return SimDir(cfg["simdir"], max_depth=0)

    def walk_spec(self, cfg):
        excl = set(_STANDARD_EXCLUDES) | set(cfg["simdir_exclude"]["dirs"])
        skip_file = None
        patterns = cfg["simdir_exclude"]["files"]
        if patterns:
            # One combined regex matched against basenames, as in postcactus.
            rx = re.compile("|".join(fnmatch.translate(str(p)) for p in patterns))
            skip_file = rx.match
        return {"excluded_dirs": excl, "skip_file": skip_file,
                "max_depth": int(cfg["scan_max_depth"])}

    def plane_index(self, sim, cfg):
        # sd.grid is a cached omni reader (HDF5 + ASCII) per plane; the same
        # reader is used for the query and the resampling read, so data files
        # are scanned once and the iterations match the plane resampled.
        return getattr(sim.grid, cfg["plane"])

    def query(self, plane_index, var):
        return plane_index.get_iters(var), plane_index.get_times(var)

    def extract_warm_state(self, plane_index, variables):
        warm = {}
        for var in variables:
            gr = plane_index._src[var]
            warm[var] = (
                gr._iters.get(var),
                gr._times.get(var),
                gr._restarts.get(var),
                gr._vars.get(var),  # {restart: [GridH5File]} with parsed TOCs
            )
        return warm

    def inject_warm_state(self, plane_index, warm):
        for var, (iters, times, restarts, files) in warm.items():
            gr = plane_index._src[var]
            if iters is not None:
                gr._iters[var] = iters
            if times is not None:
                gr._times[var] = times
            if restarts is not None:
                gr._restarts[var] = restarts
            if files is not None:
                gr._vars[var] = files

    def _geometry(self, cfg):
        if self._geom is None:
            resolution, min_corner, max_corner = grid_bounds(cfg["grid"])
            self._geom = gd.RegGeom(resolution, min_corner, x1=max_corner)
        return self._geom

    def read_slice(self, plane_index, var, it, cfg):
        slice_obj = plane_index.read(
            var, int(it), geom=self._geometry(cfg),
            adjust_spacing=0, order=cfg["interp_order"],
        )
        return _extract_array(slice_obj)


def main():
    run(PostcactusBackend())


if __name__ == "__main__":
    main()
