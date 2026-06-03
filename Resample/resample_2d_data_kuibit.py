#!/usr/bin/env python3
"""Resample 2D slices from an Einstein Toolkit simulation into HDF5 files.

kuibit backend. This is the kuibit (https://github.com/Sbozzolo/kuibit) port of
``resample_2d_data_postcactus.py`` (which uses the older postcactus). It is
intentionally a near-copy: the MPI distribution, YAML config, round-robin
scheduling, HDF5 streaming output, and the on-disk schema are identical, so the
two backends are interchangeable and produce files readable by the same
``read_data.py``.

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
import re
import sys
import time

import numpy as np
import yaml

from kuibit.simdir import SimDir

from mpi4py import MPI

import h5py


# ---------------------------------------------------------------------------
# MPI setup
# ---------------------------------------------------------------------------

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def log(message):
    """Print a progress message from rank 0 only, flushed immediately."""
    if rank == 0:
        print(message, flush=True)


def abort(message):
    """Print an error on rank 0 and tear down all ranks cleanly."""
    if rank == 0:
        print("ERROR: " + message, file=sys.stderr, flush=True)
    comm.Abort(1)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_config(path):
    """Load and validate the YAML configuration on rank 0."""
    with open(path) as f:
        cfg = yaml.safe_load(f)

    required = ["simdir", "label", "variables"]
    missing = [k for k in required if k not in cfg or cfg[k] is None]
    if missing:
        abort("config is missing required key(s): " + ", ".join(missing))

    # Fill in defaults.
    cfg.setdefault("output_dir", "./resampled")
    cfg.setdefault("plane", "xy")
    cfg.setdefault("interp_order", 1)
    cfg.setdefault("dtype", "float64")
    cfg.setdefault("iteration_stride", 1)
    cfg.setdefault("iteration_min", None)
    cfg.setdefault("iteration_max", None)
    cfg.setdefault("compression_level", 4)

    grid = cfg.setdefault("grid", {})
    grid.setdefault("resolution", [1000, 1000])
    grid.setdefault("box_bound", 50.0)

    if cfg["plane"] not in ("xy", "xz", "yz"):
        abort("plane must be one of xy, xz, yz (got %r)" % cfg["plane"])

    return cfg


def grid_bounds(grid):
    """Return (resolution, min_corner, max_corner) as lists from the config."""
    resolution = list(grid["resolution"])
    if grid.get("min") is not None and grid.get("max") is not None:
        return resolution, list(grid["min"]), list(grid["max"])
    b = float(grid["box_bound"])
    return resolution, [-b, -b], [b, b]


# ---------------------------------------------------------------------------
# Simulation metadata (computed on rank 0, broadcast to all)
# ---------------------------------------------------------------------------

def collect_iterations(sim, variables, plane, stride, it_min, it_max):
    """Return {variable: (iterations, times)} for every variable that exists.

    Iterations are queried per-variable because different variables can be
    written at different cadences. Variables with no 2D data are dropped with
    a warning.
    """
    gf = sim.gridfunctions[plane]

    result = {}
    for var in variables:
        try:
            reader = gf[var]
            iters = np.asarray(reader.available_iterations, dtype=np.int64)
            times = np.asarray(reader.available_times, dtype=np.float64)
        except Exception as exc:  # noqa: BLE001 - report and skip
            log("  WARNING: skipping %r (could not read iterations: %s)" % (var, exc))
            continue

        if iters.size == 0:
            log("  WARNING: skipping %r (no 2D data found)" % var)
            continue

        # Restrict range, then subsample.
        mask = np.ones(iters.shape, dtype=bool)
        if it_min is not None:
            mask &= iters >= it_min
        if it_max is not None:
            mask &= iters <= it_max
        iters, times = iters[mask], times[mask]
        iters, times = iters[::stride], times[::stride]

        if iters.size == 0:
            log("  WARNING: skipping %r (no iterations left after range/stride)" % var)
            continue

        result[var] = (iters, times)
    return result


# ---------------------------------------------------------------------------
# Per-variable processing
# ---------------------------------------------------------------------------

def safe_filename(name):
    """Turn a variable name like 'vel[0]' into a filesystem-safe token."""
    return re.sub(r"[^0-9A-Za-z._-]+", "_", name).strip("_")


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

    # --- Open the simulation on every rank ---
    t0 = time.perf_counter()
    sim = SimDir(cfg["simdir"])
    log("SimDir scan: %.2f s" % (time.perf_counter() - t0))

    # --- Coordinate axes for the resampling grid (identical on every rank) ---
    resolution, min_corner, max_corner = grid_bounds(cfg["grid"])
    coords = (
        np.linspace(min_corner[0], max_corner[0], resolution[0]),
        np.linspace(min_corner[1], max_corner[1], resolution[1]),
    )

    # --- Per-variable iterations: compute once on rank 0, broadcast ---
    var_iters = None
    if rank == 0:
        log("Querying iterations per variable ...")
        t0 = time.perf_counter()
        var_iters = collect_iterations(
            sim, cfg["variables"], cfg["plane"], cfg["iteration_stride"],
            cfg["iteration_min"], cfg["iteration_max"],
        )
        for var, (iters, _) in var_iters.items():
            log("  %-26s %d iterations" % (var, iters.size))
        log("Iteration query: %.2f s" % (time.perf_counter() - t0))
    var_iters = comm.bcast(var_iters, root=0)

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
