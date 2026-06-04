#!/usr/bin/env python3
"""Resample 2D slices from an Einstein Toolkit simulation into HDF5 files.

postcactus backend. It reads grid data through postcactus, resamples each
requested variable onto a fixed regular grid, and writes one HDF5 file per
variable. See ``resample_2d_data_kuibit.py`` for the equivalent kuibit backend.
Backend-agnostic helpers (MPI setup, config, pickled-SimDir cache, iteration
bookkeeping) are shared between the backends via ``resample_common.py``.

Design notes
------------
* **MPI-parallel, any rank count.** Variables are distributed round-robin
  across ranks (``variables[rank::size]``). Each rank writes its own output
  files, so there is *no* gather-to-rank-0 step and therefore no single-rank
  memory blow-up. Run with any ``N`` from 1 up to (and beyond) the number of
  variables; extra ranks simply have nothing to do.
* **Bounded memory.** Each variable is streamed iteration-by-iteration
  straight into its HDF5 dataset, so a rank never holds more than one 2D slice
  in memory at a time.
* **Correct per-variable iterations.** Iteration/time lists are queried *per
  variable* (different variables may be output at different cadences), computed
  once on rank 0 and broadcast.
* **Scan once, broadcast.** The SimDir is opened and queried on rank 0 only,
  then broadcast (with its parsed HDF5 tables of contents) to all ranks, so
  the simulation directory is scanned a single time per job. With
  ``simdir_pickle`` in the config, the scan can also be cached on disk and
  reused across jobs.
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

import argparse
import os
import time

import numpy as np

from postcactus.simdir import SimDir
from postcactus import grid_data as gd

import h5py

from resample_common import (
    comm, rank, size, log, abort,
    load_config, grid_bounds, safe_filename,
    open_simdir, save_simdir_pickle, collect_iterations,
)


# ---------------------------------------------------------------------------
# Per-variable processing
# ---------------------------------------------------------------------------

def extract_array(slice_obj):
    """Return the underlying numpy array from a postcactus resampled slice."""
    data = getattr(slice_obj, "data", None)
    if data is None:
        data = np.asarray(slice_obj)
    return np.asarray(data)


def process_variable(sd, var, iters, times, geom, coords, cfg):
    """Resample all iterations of one variable and stream them into HDF5."""
    resolution, _, _ = grid_bounds(cfg["grid"])
    nx, ny = resolution
    out_dtype = np.dtype(cfg["dtype"])
    plane_reader = getattr(sd.grid, cfg["plane"])

    out_path = os.path.join(
        cfg["output_dir"], "%s__%s.h5" % (safe_filename(var), cfg["label"])
    )

    compression = None
    comp_opts = None
    if cfg["compression_level"]:
        compression = "gzip"
        comp_opts = int(cfg["compression_level"])

    n_iter = len(iters)
    with h5py.File(out_path, "w") as h5:
        dset = h5.create_dataset(
            "data",
            shape=(n_iter, nx, ny),
            dtype=out_dtype,
            compression=compression,
            compression_opts=comp_opts,
            # Chunk per-slice so reads of a single time are cheap.
            chunks=(1, nx, ny),
        )

        timing = {"read": 0.0, "write": 0.0}
        t_var = time.perf_counter()
        for i, it in enumerate(iters):
            t0 = time.perf_counter()
            slice_obj = plane_reader.read(          # read + resample
                var, int(it), geom=geom,
                adjust_spacing=0, order=cfg["interp_order"],
            )
            t1 = time.perf_counter()
            dset[i] = extract_array(slice_obj).astype(out_dtype, copy=False)
            t2 = time.perf_counter()
            timing["read"] += t1 - t0
            timing["write"] += t2 - t1

            if (i + 1) % 50 == 0:
                print("[rank %d] %s: %d/%d (%.2f s/it)"
                      % (rank, var, i + 1, n_iter,
                         (time.perf_counter() - t_var) / (i + 1)), flush=True)

        # Coordinates and time axis travel with the data.
        h5.create_dataset("iterations", data=np.asarray(iters, dtype=np.int64))
        h5.create_dataset("times", data=np.asarray(times, dtype=np.float64))
        h5.create_dataset("x", data=coords[0])
        h5.create_dataset("y", data=coords[1])

        h5.attrs["variable"] = var
        h5.attrs["label"] = cfg["label"]
        h5.attrs["plane"] = cfg["plane"]
        h5.attrs["simdir"] = cfg["simdir"]
        h5.attrs["interp_order"] = cfg["interp_order"]
        h5.attrs["resolution"] = resolution

    return out_path, n_iter, timing


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # --- Parse args & load config (rank 0), then broadcast ---
    cfg = None
    if rank == 0:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("config", help="Path to the YAML configuration file.")
        args = parser.parse_args()
        cfg = load_config(args.config)
        os.makedirs(cfg["output_dir"], exist_ok=True)
    cfg = comm.bcast(cfg, root=0)

    log("Checkpoint Start: simulation %r" % cfg["label"])
    t_start = time.perf_counter()

    # --- Resampling geometry (identical on every rank) ---
    resolution, min_corner, max_corner = grid_bounds(cfg["grid"])
    geom = gd.RegGeom(resolution, min_corner, x1=max_corner)
    coords = (
        np.linspace(min_corner[0], max_corner[0], resolution[0]),
        np.linspace(min_corner[1], max_corner[1], resolution[1]),
    )

    # --- Open the simulation and query iterations on rank 0 only, then
    # broadcast. The SimDir is broadcast *after* the iteration query so the
    # parsed HDF5 tables of contents travel with it and no other rank has to
    # re-scan directories or re-parse file metadata. ---
    sd = None
    var_iters = None
    if rank == 0:
        t0 = time.perf_counter()
        # simdir_exclude prunes the scan; irrelevant (already baked in)
        # when the SimDir is loaded from a pickle.
        excl = cfg["simdir_exclude"]
        sd = open_simdir(cfg, lambda path: SimDir(
            path,
            exclude_dirs=excl["dirs"] or None,
            exclude_files=excl["files"] or None,
        ))
        log("SimDir ready: %.2f s" % (time.perf_counter() - t0))

        log("Querying iterations per variable ...")
        t0 = time.perf_counter()
        # sd.grid is a cached omni reader (HDF5 + ASCII) per plane, the same
        # one process_variable reads from - so the data files are scanned
        # only once, and the iterations match the plane actually resampled.
        plane_reader = getattr(sd.grid, cfg["plane"])
        var_iters = collect_iterations(
            lambda var: (plane_reader.get_iters(var), plane_reader.get_times(var)),
            cfg["variables"], cfg["iteration_stride"],
            cfg["iteration_min"], cfg["iteration_max"],
        )
        for var, (iters, _) in var_iters.items():
            log("  %-26s %d iterations" % (var, iters.size))
        log("Iteration query: %.2f s" % (time.perf_counter() - t0))

        # Save after the query so the pickle includes the parsed TOCs.
        if cfg["simdir_pickle"]["path"] and not cfg["simdir_pickle"]["pickled"]:
            save_simdir_pickle(sd, cfg)

    t0 = time.perf_counter()
    sd = comm.bcast(sd, root=0)
    var_iters = comm.bcast(var_iters, root=0)
    log("Broadcast SimDir + iterations: %.2f s" % (time.perf_counter() - t0))

    if not var_iters:
        abort("no variables with usable 2D data were found.")

    # --- Round-robin assignment of variables to ranks ---
    all_vars = list(var_iters.keys())
    my_vars = all_vars[rank::size]

    log("Checkpoint Load: %d variables across %d rank(s)" % (len(all_vars), size))

    for var in my_vars:
        iters, times = var_iters[var]
        out_path, n_iter, tm = process_variable(sd, var, iters, times, geom,
                                                coords, cfg)
        busy = tm["read"] + tm["write"]
        print(
            "[rank %d] wrote %s (%d iterations) "
            "[read+resample %.1f s, write %.1f s; %.2f s/it]"
            % (rank, out_path, n_iter, tm["read"], tm["write"],
               busy / max(n_iter, 1)),
            flush=True,
        )

    comm.Barrier()
    log("Checkpoint End: all variables written to %s (total wall %.1f s)"
        % (cfg["output_dir"], time.perf_counter() - t_start))


if __name__ == "__main__":
    main()
