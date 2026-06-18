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

import argparse
import os
import pickle
import re
import sys
import time
from abc import ABC, abstractmethod

import numpy as np
import yaml
import h5py

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
    # Query mode: distribute the per-variable iteration query across ranks
    # (parallel) or run it entirely on rank 0 (serial). See run().
    cfg.setdefault("parallel_query", True)

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

    # Scan exclusions, honoured by the postcactus backend only.
    se = cfg.get("simdir_exclude") or {}
    for key in ("dirs", "files"):
        val = se.get(key) or []
        if isinstance(val, str):
            val = [val]
        se[key] = [str(v) for v in val]
    cfg["simdir_exclude"] = se

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

def open_simdir(cfg, backend):
    """Return the SimDir for the configured simulation (called on rank 0).

    With ``simdir_pickle.pickled: yes`` the SimDir is unpickled from
    ``simdir_pickle.path`` instead of scanning the simulation directory.
    Otherwise it is built by ``backend.scan(cfg)`` (a recursive directory
    scan in the underlying library).
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
    return backend.scan(cfg)


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


# ---------------------------------------------------------------------------
# Backend abstraction
# ---------------------------------------------------------------------------

class Backend(ABC):
    """The single point of coupling to a reader library (postcactus / kuibit).

    Everything the orchestrator (``run``) needs from the underlying library is
    expressed through this interface; the two backend modules implement it and
    contain *no* other library-specific code. The orchestrator never imports
    postcactus or kuibit, so only the selected backend's heavy dependency is
    pulled in.

    The "plane index" returned by :meth:`plane_index` is the per-plane reader
    object (postcactus ``sd.grid.<plane>``; kuibit ``sim.gridfunctions[plane]``).
    It is built once per rank and threaded through query / warm-state / read.
    Querying a variable (``query``) populates the library's internal caches for
    that variable; :meth:`extract_warm_state` / :meth:`inject_warm_state` move
    exactly those caches between ranks so a parallel query can reassemble, on
    rank 0, the same fully-warmed SimDir a serial query would have produced.
    """

    #: Short name, stored in the output files as the ``backend`` attribute.
    name = "abstract"

    @abstractmethod
    def scan(self, cfg):
        """Build and return the SimDir by scanning ``cfg['simdir']`` on disk."""

    @abstractmethod
    def plane_index(self, sim, cfg):
        """Materialise and return the per-plane reader for ``cfg['plane']``.

        Called on rank 0 before the SimDir is broadcast (so the plane index
        travels with it) and on every rank before resampling.
        """

    @abstractmethod
    def query(self, plane_index, var):
        """Return ``(iterations, times)`` for ``var``, warming its caches.

        May raise if the variable has no data for this plane;
        :func:`collect_iterations` catches and skips such variables.
        """

    @abstractmethod
    def extract_warm_state(self, plane_index, variables):
        """Return a picklable blob with the warmed cache state for ``variables``."""

    @abstractmethod
    def inject_warm_state(self, plane_index, warm):
        """Merge a blob from :meth:`extract_warm_state` into ``plane_index``."""

    @abstractmethod
    def read_slice(self, plane_index, var, it, cfg):
        """Read+resample one iteration of ``var`` and return a 2D ndarray."""


# ---------------------------------------------------------------------------
# Per-variable processing (backend-agnostic HDF5 streaming)
# ---------------------------------------------------------------------------

def process_variable(backend, plane_index, var, iters, times, coords, cfg):
    """Resample all iterations of one variable and stream them into HDF5."""
    resolution, _, _ = grid_bounds(cfg["grid"])
    nx, ny = resolution
    out_dtype = np.dtype(cfg["dtype"])

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
            arr = backend.read_slice(plane_index, var, int(it), cfg)  # read + resample
            t1 = time.perf_counter()
            dset[i] = np.asarray(arr).astype(out_dtype, copy=False)
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
        h5.attrs["backend"] = backend.name

    return out_path, n_iter, timing


# ---------------------------------------------------------------------------
# Orchestrator (shared by both backends)
# ---------------------------------------------------------------------------

def _query_subset(backend, plane_index, variables, cfg):
    """Run the iteration query for ``variables`` and return {var: (iters, times)}."""
    return collect_iterations(
        lambda var: backend.query(plane_index, var),
        variables, cfg["iteration_stride"],
        cfg["iteration_min"], cfg["iteration_max"],
    )


def run(backend):
    """Drive the full resampling pipeline for the given backend.

    Two query modes, selected by ``parallel_query`` in the config:

    * **serial** - rank 0 opens the SimDir, queries *every* variable (warming
      all caches), optionally saves the pickle, then broadcasts the SimDir and
      the iteration tables to all ranks. Variables are resampled round-robin.

    * **parallel** - rank 0 opens the SimDir and broadcasts it *cold*; every
      rank queries its own ``variables[rank::size]`` (warming exactly the
      variables it will resample, so nothing has to be broadcast back). The
      iteration tables *and* the warmed cache state are gathered to rank 0,
      which injects the latter and saves the pickle through the *same* code
      path as the serial mode - so the on-disk pickle is identical either way.
    """
    # --- Parse args & load config (rank 0), then broadcast ---
    cfg = None
    if rank == 0:
        parser = argparse.ArgumentParser()
        parser.add_argument("config", help="Path to the YAML configuration file.")
        args = parser.parse_args()
        cfg = load_config(args.config)
        os.makedirs(cfg["output_dir"], exist_ok=True)
    cfg = comm.bcast(cfg, root=0)

    parallel = bool(cfg["parallel_query"])
    log("Checkpoint Start: simulation %r (%s backend, %s query)"
        % (cfg["label"], backend.name, "parallel" if parallel else "serial"))
    t_start = time.perf_counter()

    # --- Coordinate axes for the resampling grid (identical on every rank) ---
    resolution, min_corner, max_corner = grid_bounds(cfg["grid"])
    coords = (
        np.linspace(min_corner[0], max_corner[0], resolution[0]),
        np.linspace(min_corner[1], max_corner[1], resolution[1]),
    )

    if parallel:
        my_var_iters = _run_parallel_query(backend, cfg)
    else:
        my_var_iters = _run_serial_query(backend, cfg)

    # --- Resample this rank's assigned variables ---
    # Each rank rebuilds the plane index from its (now warm) SimDir copy.
    sim = _SIM_CACHE["sim"]
    plane_index = backend.plane_index(sim, cfg)

    log("Checkpoint Load: resampling %d variable(s) on rank 0 "
        "(round-robin across %d rank(s))" % (len(my_var_iters), size))

    for var, (iters, times) in my_var_iters.items():
        out_path, n_iter, tm = process_variable(
            backend, plane_index, var, iters, times, coords, cfg)
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


# rank-local handle on the SimDir, set by the query helpers and reused for
# resampling (avoids threading the big object through every return value).
_SIM_CACHE = {"sim": None}


def _run_serial_query(backend, cfg):
    """Serial query: rank 0 queries everything, then broadcast. Returns this
    rank's {var: (iters, times)} for resampling."""
    sim = None
    var_iters = None
    if rank == 0:
        t0 = time.perf_counter()
        sim = open_simdir(cfg, backend)
        plane_index = backend.plane_index(sim, cfg)
        log("SimDir ready: %.2f s" % (time.perf_counter() - t0))

        log("Querying iterations per variable ...")
        t0 = time.perf_counter()
        var_iters = _query_subset(backend, plane_index, cfg["variables"], cfg)
        for var, (iters, _) in var_iters.items():
            log("  %-26s %d iterations" % (var, iters.size))
        log("Iteration query: %.2f s" % (time.perf_counter() - t0))

        if cfg["simdir_pickle"]["path"] and not cfg["simdir_pickle"]["pickled"]:
            save_simdir_pickle(sim, cfg)

    t0 = time.perf_counter()
    sim = comm.bcast(sim, root=0)
    var_iters = comm.bcast(var_iters, root=0)
    log("Broadcast SimDir + iterations: %.2f s" % (time.perf_counter() - t0))

    if not var_iters:
        abort("no variables with usable 2D data were found.")

    _SIM_CACHE["sim"] = sim
    all_vars = list(var_iters.keys())
    return {v: var_iters[v] for v in all_vars[rank::size]}


def _run_parallel_query(backend, cfg):
    """Parallel query: rank 0 scans and broadcasts a cold SimDir; every rank
    queries its own slice of the variables. The iteration tables and warmed
    cache state are gathered to rank 0 for an identical pickle. Returns this
    rank's {var: (iters, times)} for resampling."""
    # --- rank 0 scans (or loads pickle) and materialises the plane index so
    # it travels with the broadcast; then broadcast (cold if freshly scanned,
    # already warm if loaded from a pickle). ---
    sim = None
    if rank == 0:
        t0 = time.perf_counter()
        sim = open_simdir(cfg, backend)
        backend.plane_index(sim, cfg)  # force-build so it travels with the bcast
        log("SimDir ready: %.2f s" % (time.perf_counter() - t0))

    t0 = time.perf_counter()
    sim = comm.bcast(sim, root=0)
    log("Broadcast SimDir: %.2f s" % (time.perf_counter() - t0))
    _SIM_CACHE["sim"] = sim

    plane_index = backend.plane_index(sim, cfg)

    # --- Each rank queries its own slice of the variables. ---
    my_query_vars = cfg["variables"][rank::size]
    t0 = time.perf_counter()
    my_var_iters = _query_subset(backend, plane_index, my_query_vars, cfg)
    log("Iteration query (rank 0 subset of %d/%d vars): %.2f s"
        % (len(my_query_vars), len(cfg["variables"]), time.perf_counter() - t0))

    # --- Gather iteration tables and warm cache state to rank 0. ---
    my_warm = backend.extract_warm_state(plane_index, list(my_var_iters.keys()))
    t0 = time.perf_counter()
    all_var_iters = comm.gather(my_var_iters, root=0)
    all_warm = comm.gather(my_warm, root=0)

    if rank == 0:
        per_rank = {}
        for vi in all_var_iters:
            per_rank.update(vi)
        # Preserve the configured variable order in the summary.
        merged = {var: per_rank[var]
                  for var in cfg["variables"] if var in per_rank}
        for var, (iters, _) in merged.items():
            log("  %-26s %d iterations" % (var, iters.size))
        log("Gather iterations + warm state: %.2f s" % (time.perf_counter() - t0))

        # Reassemble the fully-warmed SimDir on rank 0 and pickle it exactly
        # as the serial path does.
        for blob in all_warm:
            backend.inject_warm_state(plane_index, blob)
        if cfg["simdir_pickle"]["path"] and not cfg["simdir_pickle"]["pickled"]:
            save_simdir_pickle(sim, cfg)

        total = len(merged)
    else:
        total = None

    total = comm.bcast(total, root=0)
    if not total:
        abort("no variables with usable 2D data were found.")

    # Each rank resamples exactly the variables it queried (already warm here).
    return my_var_iters
