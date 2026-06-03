#!/usr/bin/env python3
"""Shared helpers for the 2D resampling backends.

Everything here is backend-agnostic: MPI setup and logging, YAML config
loading/validation, the pickled-SimDir cache, the per-variable iteration
bookkeeping, and small utilities. The backend scripts
(``resample_2d_data_postcactus.py`` / ``resample_2d_data_kuibit.py``) keep
only the read/resample layer that actually differs between the libraries.

Importing this module initialises MPI (through mpi4py), exactly as the
backend scripts did before the helpers were extracted.
"""

import os
import pickle
import re
import sys

import numpy as np
import yaml

from mpi4py import MPI


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

    # Pickled SimDir support (see open_simdir / save_simdir_pickle).
    sp = cfg.get("simdir_pickle") or {}
    sp.setdefault("pickled", False)
    sp.setdefault("path", None)
    cfg["simdir_pickle"] = sp
    if sp["pickled"] and not sp["path"]:
        abort("simdir_pickle.pickled is yes but simdir_pickle.path is not set")
    if sp["pickled"] and not os.path.isfile(sp["path"]):
        abort("simdir_pickle.path does not exist: %r" % sp["path"])

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
# Pickled SimDir cache
# ---------------------------------------------------------------------------

def open_simdir(cfg, simdir_factory):
    """Return the SimDir for the configured simulation (called on rank 0).

    ``simdir_factory`` is the backend's SimDir class (postcactus or kuibit).
    With ``simdir_pickle.pickled: yes`` the SimDir is unpickled from
    ``simdir_pickle.path`` instead of scanning the simulation directory.
    Otherwise it is constructed the usual way (recursive directory scan).
    """
    sp = cfg["simdir_pickle"]
    if sp["pickled"]:
        log("Loading pickled SimDir from %s" % sp["path"])
        try:
            with open(sp["path"], "rb") as f:
                sd = pickle.load(f)
        except Exception as exc:  # noqa: BLE001 - any load failure is fatal
            abort("could not load pickled SimDir from %r: %s" % (sp["path"], exc))
        if os.path.abspath(sd.path) != os.path.abspath(cfg["simdir"]):
            log("  WARNING: pickle was created for %r but config simdir is %r"
                % (sd.path, cfg["simdir"]))
        return sd
    return simdir_factory(cfg["simdir"])


def save_simdir_pickle(sd, cfg):
    """Save the SimDir to simdir_pickle.path for reuse by later runs.

    Called after the iteration query, so the pickle also carries the
    reader's cached file metadata (e.g. parsed HDF5 tables of contents):
    a later run with ``pickled: yes`` skips both the directory scan and
    that metadata work. A failed save only warns - the resampling run
    itself is unaffected.
    """
    path = cfg["simdir_pickle"]["path"]
    try:
        with open(path, "wb") as f:
            pickle.dump(sd, f)
        log("Saved SimDir pickle to %s (set simdir_pickle.pickled: yes to reuse)"
            % path)
    except Exception as exc:  # noqa: BLE001 - cache write is best-effort
        log("  WARNING: could not save SimDir pickle to %r: %s" % (path, exc))


# ---------------------------------------------------------------------------
# Simulation metadata (computed on rank 0, broadcast to all)
# ---------------------------------------------------------------------------

def collect_iterations(query, variables, stride, it_min, it_max):
    """Return {variable: (iterations, times)} for every variable that exists.

    ``query(var)`` is a backend-supplied callable returning the available
    ``(iterations, times)`` for one variable (raising if it has no data).
    Iterations are queried per-variable because different variables can be
    written at different cadences. Variables with no 2D data are dropped
    with a warning.
    """
    result = {}
    for var in variables:
        try:
            iters, times = query(var)
            iters = np.asarray(iters, dtype=np.int64)
            times = np.asarray(times, dtype=np.float64)
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
# Small utilities
# ---------------------------------------------------------------------------

def safe_filename(name):
    """Turn a variable name like 'vel[0]' into a filesystem-safe token."""
    return re.sub(r"[^0-9A-Za-z._-]+", "_", name).strip("_")
