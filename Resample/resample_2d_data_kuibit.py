#!/usr/bin/env python3
"""Resample 2D slices from an Einstein Toolkit simulation into HDF5 files.

kuibit backend. This is the kuibit (https://github.com/Sbozzolo/kuibit) port of
``resample_2d_data_postcactus.py`` (which uses the older postcactus). It is
intentionally a near-copy: the MPI distribution, YAML config, round-robin
scheduling, HDF5 streaming output, and the on-disk schema are identical, so the
two backends are interchangeable and produce files readable by the same
``read_data.py``. Backend-agnostic helpers (MPI setup, config, pickled-SimDir
cache, iteration bookkeeping) are shared via ``resample_common.py``.

Only the read/resample layer differs. In kuibit the whole postcactus chain
(GridH5Dir + GridASCIIDir + GridOmniReader + RegGeom + grid.xy.read) collapses
into:

    sim.gridfunctions[plane][variable][iteration].to_UniformGridData(...)

Design notes (unchanged from the postcactus version)
-----------------------------------------------------
* MPI-parallel, any rank count: variables are distributed round-robin
  (``variables[rank::size]``); each rank writes its own files (no gather, no
  single-rank memory blow-up).
* Bounded memory: each variable is streamed iteration-by-iteration into HDF5.
* Correct per-variable iterations: queried per variable (different variables may
  be output at different cadences), computed once on rank 0 and broadcast.
* Scan once, broadcast: the SimDir is opened and queried on rank 0 only, then
  broadcast to all ranks. With ``simdir_pickle`` in the config, the scan can
  also be cached on disk and reused across jobs.
* Self-describing output: iterations, physical times, and coordinate axes are
  stored alongside the data, plus metadata as attributes.

Usage
-----
    mpirun -n <N> python resample_2d_data_kuibit.py config.yaml

(or via the launcher: ``mpirun -n <N> python resample_2d.py config.yaml`` with
``backend: kuibit`` in the config.)

See ``config_example.yaml`` for all options. ``interp_order >= 1`` selects
multilinear resampling (kuibit ``resample=True``); ``interp_order == 0`` selects
nearest-neighbour (``resample=False``).
"""

import argparse
import os
import time

import numpy as np

from kuibit.simdir import SimDir

import h5py

from resample_common import (
    comm, rank, size, log, abort,
    load_config, grid_bounds, safe_filename,
    open_simdir, save_simdir_pickle, collect_iterations,
)


# ---------------------------------------------------------------------------
# Per-variable processing
# ---------------------------------------------------------------------------

def process_variable(sim, var, iters, times, coords, cfg):
    """Resample all iterations of one variable and stream them into HDF5."""
    resolution, min_corner, max_corner = grid_bounds(cfg["grid"])
    nx, ny = resolution
    out_dtype = np.dtype(cfg["dtype"])
    reader = sim.gridfunctions[cfg["plane"]][var]
    # kuibit: resample=True is multilinear interpolation, False is nearest.
    resample = int(cfg["interp_order"]) >= 1

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

        timing = {"load": 0.0, "resample": 0.0, "write": 0.0}
        t_var = time.perf_counter()
        for i, it in enumerate(iters):
            t0 = time.perf_counter()
            hgd = reader[int(it)]                       # read from disk
            t1 = time.perf_counter()
            ugd = hgd.to_UniformGridData(               # merge + interpolate
                resolution, min_corner, max_corner,
                resample=resample, iteration=int(it),
            )
            t2 = time.perf_counter()
            dset[i] = np.asarray(ugd.data).astype(out_dtype, copy=False)
            t3 = time.perf_counter()
            timing["load"] += t1 - t0
            timing["resample"] += t2 - t1
            timing["write"] += t3 - t2

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
        h5.attrs["backend"] = "kuibit"

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

    log("Checkpoint Start: simulation %r (kuibit backend)" % cfg["label"])
    t_start = time.perf_counter()

    # --- Coordinate axes for the resampling grid (identical on every rank) ---
    resolution, min_corner, max_corner = grid_bounds(cfg["grid"])
    coords = (
        np.linspace(min_corner[0], max_corner[0], resolution[0]),
        np.linspace(min_corner[1], max_corner[1], resolution[1]),
    )

    # --- Open the simulation and query iterations on rank 0 only, then
    # broadcast. The SimDir is broadcast *after* the iteration query so
    # kuibit's cached metadata travels with it and no other rank has to
    # re-scan directories. ---
    sim = None
    var_iters = None
    if rank == 0:
        t0 = time.perf_counter()
        if cfg["simdir_exclude"]["dirs"] or cfg["simdir_exclude"]["files"]:
            log("  WARNING: simdir_exclude is not supported by the kuibit "
                "backend and will be ignored")
        sim = open_simdir(cfg, SimDir)
        log("SimDir ready: %.2f s" % (time.perf_counter() - t0))

        log("Querying iterations per variable ...")
        t0 = time.perf_counter()
        gf = sim.gridfunctions[cfg["plane"]]

        def query(var):
            reader = gf[var]
            return reader.available_iterations, reader.available_times

        var_iters = collect_iterations(
            query, cfg["variables"], cfg["iteration_stride"],
            cfg["iteration_min"], cfg["iteration_max"],
        )
        for var, (iters, _) in var_iters.items():
            log("  %-26s %d iterations" % (var, iters.size))
        log("Iteration query: %.2f s" % (time.perf_counter() - t0))

        # Save after the query so the pickle includes the cached metadata.
        if cfg["simdir_pickle"]["path"] and not cfg["simdir_pickle"]["pickled"]:
            save_simdir_pickle(sim, cfg)

    t0 = time.perf_counter()
    sim = comm.bcast(sim, root=0)
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
        out_path, n_iter, tm = process_variable(sim, var, iters, times,
                                                coords, cfg)
        busy = tm["load"] + tm["resample"] + tm["write"]
        print(
            "[rank %d] wrote %s (%d iterations) "
            "[load %.1f s, resample %.1f s, write %.1f s; %.2f s/it]"
            % (rank, out_path, n_iter, tm["load"], tm["resample"],
               tm["write"], busy / max(n_iter, 1)),
            flush=True,
        )

    comm.Barrier()
    log("Checkpoint End: all variables written to %s (total wall %.1f s)"
        % (cfg["output_dir"], time.perf_counter() - t_start))


if __name__ == "__main__":
    main()
