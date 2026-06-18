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
    # Scan mode: distribute the directory walk over ranks by handing each rank
    # a share of the top-level subdirectories (parallel) or walk on rank 0 only
    # (serial). scan_max_depth bounds the recursion depth in both cases.
    cfg.setdefault("parallel_scan", True)
    cfg.setdefault("scan_max_depth", 8)
    # Resampling granularity: split each variable's iterations into this many
    # contiguous chunks and distribute (variable, chunk) tasks across ranks, so
    # more ranks than variables can be kept busy; the per-variable chunks are
    # merged into one file afterwards. 1 = one file per variable, no merge.
    cfg.setdefault("resample_chunks", 1)
    if int(cfg["resample_chunks"]) < 1:
        abort("resample_chunks must be >= 1 (got %r)" % cfg["resample_chunks"])

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
        """Build and return the SimDir by scanning ``cfg['simdir']`` on disk
        (recursion bounded by ``cfg['scan_max_depth']``)."""

    @abstractmethod
    def scan_shallow(self, cfg):
        """Build a SimDir without walking subdirectories (``max_depth=0``).

        Used by the parallel scan: the cheap top-level build captures path and
        SIMFACTORY parfile metadata, then the rank-gathered file list is
        injected via :func:`inject_file_index`.
        """

    @abstractmethod
    def walk_spec(self, cfg):
        """Return ``{excluded_dirs, skip_file, max_depth}`` for the parallel walk.

        ``excluded_dirs`` is a set of directory *names* not descended into,
        ``skip_file`` is ``None`` or a callable taking a basename and returning
        truthy to drop the file, and ``max_depth`` bounds recursion. These must
        match the library's own scan so the parallel walk yields the same files.
        """

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

def variable_path(cfg, var):
    """Final output path for a variable's resampled file."""
    return os.path.join(
        cfg["output_dir"], "%s__%s.h5" % (safe_filename(var), cfg["label"])
    )


def process_variable(backend, plane_index, var, iters, times, coords, cfg,
                     out_path=None, write_meta=True):
    """Resample the given iterations of one variable and stream them into HDF5.

    With ``write_meta`` the file is a complete, self-describing output (coords +
    attributes). Chunked resampling writes partial files with ``write_meta`` off
    (only ``data``/``iterations``/``times``); the merge step adds coords + attrs.
    """
    resolution, _, _ = grid_bounds(cfg["grid"])
    nx, ny = resolution
    out_dtype = np.dtype(cfg["dtype"])

    if out_path is None:
        out_path = variable_path(cfg, var)

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

        # The iteration/time axis always travels with the data (the merge step
        # relies on it); coordinates and metadata only for complete files.
        h5.create_dataset("iterations", data=np.asarray(iters, dtype=np.int64))
        h5.create_dataset("times", data=np.asarray(times, dtype=np.float64))
        if write_meta:
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
# Parallel directory walk (shared; driven by backend.walk_spec)
# ---------------------------------------------------------------------------

def inject_file_index(sim, allfiles, dirs):
    """Replace a SimDir's scanned file index with a precomputed one.

    Both backends expose the same ``allfiles``/``dirs``/``logfiles``/
    ``errfiles``/``parfiles`` attributes and derive their grid readers lazily
    from ``allfiles``, so overriding the index before the readers are built is
    enough. SIMFACTORY parfile metadata captured by the shallow build is kept.
    """
    sim.allfiles = list(allfiles)
    sim.dirs = list(dirs)

    def by_ext(ext):
        return [f for f in allfiles if os.path.splitext(f)[1] == ext]

    sim.logfiles = by_ext(".out")
    sim.errfiles = by_ext(".err")
    # The shallow build already found the SIMFACTORY parfile (excluded from the
    # walk); keep it first, then the data-directory parfiles.
    sim.parfiles = list(getattr(sim, "parfiles", []) or []) + by_ext(".par")
    sim.has_parfile = bool(sim.parfiles)


def _walk_subtree(top, start_level, excluded, skip_file, max_depth):
    """Recursively walk one subtree, mirroring the libraries' own walk.

    Files are collected for directories at levels below ``max_depth``; symlinks
    are skipped (``follow_symlinks=False``); directories named in ``excluded``
    are not descended into; ``skip_file`` (or None) drops files by basename.
    """
    files, dirs = [], []
    stack = [(top, start_level)]
    while stack:
        path, level = stack.pop()
        dirs.append(path)
        if level >= max_depth:
            continue
        try:
            scan = os.scandir(path)
        except OSError:
            continue
        with scan:
            for e in scan:
                try:
                    is_file = e.is_file(follow_symlinks=False)
                    is_dir = (not is_file) and e.is_dir(follow_symlinks=False)
                except OSError:
                    continue
                if is_file:
                    if skip_file is None or not skip_file(e.name):
                        files.append(e.path)
                elif is_dir and e.name not in excluded:
                    stack.append((e.path, level + 1))
    return files, dirs


def _parallel_scan(backend, cfg):
    """Collective: distribute the top-level subdirectories across ranks, walk
    each subtree, and assemble the file index on rank 0 (returns the SimDir on
    rank 0, None elsewhere)."""
    spec = backend.walk_spec(cfg)
    excluded, skip_file = spec["excluded_dirs"], spec["skip_file"]
    max_depth = int(spec["max_depth"])
    simdir = cfg["simdir"]

    # rank 0 lists the immediate entries: top-level files + subdirs to farm out.
    payload = None
    if rank == 0:
        top_files, top_subdirs = [], []
        if max_depth >= 1:
            with os.scandir(simdir) as scan:
                for e in scan:
                    try:
                        is_file = e.is_file(follow_symlinks=False)
                        is_dir = (not is_file) and e.is_dir(follow_symlinks=False)
                    except OSError:
                        continue
                    if is_file:
                        if skip_file is None or not skip_file(e.name):
                            top_files.append(e.path)
                    elif is_dir and e.name not in excluded:
                        top_subdirs.append(e.path)
        payload = (top_files, top_subdirs)
    top_files, top_subdirs = comm.bcast(payload, root=0)

    # Each rank fully walks its share of the top-level subdirectories.
    my_files, my_dirs = [], []
    for d in top_subdirs[rank::size]:
        f, dd = _walk_subtree(d, 1, excluded, skip_file, max_depth)
        my_files.extend(f)
        my_dirs.extend(dd)

    all_files = comm.gather(my_files, root=0)
    all_dirs = comm.gather(my_dirs, root=0)

    sim = None
    if rank == 0:
        allfiles = list(top_files)
        dirs = [simdir]
        for fl in all_files:
            allfiles.extend(fl)
        for dl in all_dirs:
            dirs.extend(dl)
        sim = backend.scan_shallow(cfg)
        inject_file_index(sim, allfiles, dirs)
        log("Parallel scan: %d top-level subdir(s) across %d rank(s), %d files"
            % (len(top_subdirs), size, len(allfiles)))
    return sim


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

    The pipeline has three phases, each with a config-selected serial/parallel
    mode so any combination can be compared against the others:

    * **scan** (``parallel_scan``) - build the SimDir. Parallel mode hands each
      rank a share of the top-level subdirectories to walk, then assembles the
      file index on rank 0; serial mode walks entirely on rank 0.
    * **query** (``parallel_query``) - obtain the per-variable iteration tables.
      Parallel mode splits the variables across ranks and gathers the warmed
      cache state to rank 0 so the pickle is identical to the serial path.
    * **resample** (``resample_chunks``) - 1 means one file per variable written
      by one rank; >1 splits each variable into contiguous iteration chunks,
      distributes ``(variable, chunk)`` tasks across ranks, and merges the
      chunks per variable afterwards (so more ranks than variables stay busy).
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

    parallel_q = bool(cfg["parallel_query"])
    chunks = int(cfg["resample_chunks"])
    log("Checkpoint Start: simulation %r (%s backend; %s scan, %s query, "
        "%d resample chunk(s))"
        % (cfg["label"], backend.name,
           "parallel" if cfg["parallel_scan"] else "serial",
           "parallel" if parallel_q else "serial", chunks))
    t_start = time.perf_counter()

    # --- Coordinate axes for the resampling grid (identical on every rank) ---
    resolution, min_corner, max_corner = grid_bounds(cfg["grid"])
    coords = (
        np.linspace(min_corner[0], max_corner[0], resolution[0]),
        np.linspace(min_corner[1], max_corner[1], resolution[1]),
    )

    # --- Phase 1: acquire the (cold) SimDir on rank 0 ---
    sim_cold = _acquire_cold_simdir(backend, cfg)

    # --- Phase 2: query iterations; returns this rank's SimDir, its own
    # {var: (iters, times)} for resampling, and (where available) the full map.
    if parallel_q:
        sim, my_var_iters, full_var_iters = _query_parallel(backend, cfg, sim_cold)
    else:
        sim, my_var_iters, full_var_iters = _query_serial(backend, cfg, sim_cold)

    plane_index = backend.plane_index(sim, cfg)

    # --- Phase 3: resample ---
    if chunks <= 1:
        _resample_direct(backend, plane_index, my_var_iters, coords, cfg)
    else:
        # Chunked resampling reassigns work globally, so every rank needs the
        # full iteration map and a fully-warmed SimDir. In serial-query mode
        # both are already broadcast; in parallel-query mode, broadcast rank 0's
        # reassembled warm SimDir and the merged map now.
        if parallel_q:
            t0 = time.perf_counter()
            sim = comm.bcast(sim, root=0)
            full_var_iters = comm.bcast(full_var_iters, root=0)
            plane_index = backend.plane_index(sim, cfg)
            log("Broadcast warm SimDir for chunked resampling: %.2f s"
                % (time.perf_counter() - t0))
        _resample_chunked(backend, plane_index, full_var_iters, coords, cfg, chunks)

    comm.Barrier()
    log("Checkpoint End: all variables written to %s (total wall %.1f s)"
        % (cfg["output_dir"], time.perf_counter() - t_start))


def _acquire_cold_simdir(backend, cfg):
    """Build the SimDir (not yet queried). Returns it on rank 0 (None elsewhere).

    Uses the parallel directory walk when ``parallel_scan`` is on, there is more
    than one rank, and we are not loading a pickle; otherwise rank 0 scans (or
    loads the pickle) on its own. The plane index is materialised on rank 0 so
    it travels with the later broadcast.
    """
    sp = cfg["simdir_pickle"]
    if bool(cfg["parallel_scan"]) and size > 1 and not sp["pickled"]:
        t0 = time.perf_counter()
        sim = _parallel_scan(backend, cfg)
        if rank == 0:
            backend.plane_index(sim, cfg)
            log("SimDir ready (parallel scan): %.2f s" % (time.perf_counter() - t0))
        return sim

    sim = None
    if rank == 0:
        t0 = time.perf_counter()
        sim = open_simdir(cfg, backend)
        backend.plane_index(sim, cfg)
        log("SimDir ready: %.2f s" % (time.perf_counter() - t0))
    return sim


def _query_serial(backend, cfg, sim_cold):
    """Serial query: rank 0 queries everything, then broadcasts the warm SimDir
    and the full iteration map. Returns (sim, my {var:(iters,times)}, full map)."""
    sim = sim_cold
    var_iters = None
    if rank == 0:
        plane_index = backend.plane_index(sim, cfg)
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

    all_vars = list(var_iters.keys())
    my_var_iters = {v: var_iters[v] for v in all_vars[rank::size]}
    return sim, my_var_iters, var_iters


def _query_parallel(backend, cfg, sim_cold):
    """Parallel query: broadcast the cold SimDir; every rank queries its own
    slice of the variables. Iteration tables and warmed cache state are gathered
    to rank 0 for an identical pickle. Returns (sim, my {var:(iters,times)},
    full map on rank 0 / None elsewhere)."""
    t0 = time.perf_counter()
    sim = comm.bcast(sim_cold, root=0)
    log("Broadcast SimDir: %.2f s" % (time.perf_counter() - t0))

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

    full_var_iters = None
    total = None
    if rank == 0:
        per_rank = {}
        for vi in all_var_iters:
            per_rank.update(vi)
        # Preserve the configured variable order in the summary.
        full_var_iters = {var: per_rank[var]
                          for var in cfg["variables"] if var in per_rank}
        for var, (iters, _) in full_var_iters.items():
            log("  %-26s %d iterations" % (var, iters.size))
        log("Gather iterations + warm state: %.2f s" % (time.perf_counter() - t0))

        # Reassemble the fully-warmed SimDir on rank 0 and pickle it exactly
        # as the serial path does.
        for blob in all_warm:
            backend.inject_warm_state(plane_index, blob)
        if cfg["simdir_pickle"]["path"] and not cfg["simdir_pickle"]["pickled"]:
            save_simdir_pickle(sim, cfg)

        total = len(full_var_iters)

    total = comm.bcast(total, root=0)
    if not total:
        abort("no variables with usable 2D data were found.")

    # Each rank resamples exactly the variables it queried (already warm here).
    return sim, my_var_iters, full_var_iters


# ---------------------------------------------------------------------------
# Resampling phase (direct one-file-per-variable, or chunked + merge)
# ---------------------------------------------------------------------------

def _resample_direct(backend, plane_index, my_var_iters, coords, cfg):
    """One file per variable: each rank resamples the variables assigned to it."""
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


def _part_path(cfg, var, chunk_idx):
    """Path of one (variable, chunk) partial file."""
    return os.path.join(
        cfg["output_dir"],
        "%s__%s.part%04d.h5" % (safe_filename(var), cfg["label"], chunk_idx),
    )


def _resample_chunked(backend, plane_index, full_var_iters, coords, cfg, chunks):
    """Split each variable into contiguous iteration chunks, distribute the
    (variable, chunk) tasks across ranks, then merge the chunks per variable."""
    all_vars = list(full_var_iters.keys())

    # Build the global task list in a deterministic order.
    tasks = []
    for var in all_vars:
        iters, times = full_var_iters[var]
        for chunk_idx, idx in enumerate(np.array_split(np.arange(len(iters)), chunks)):
            if idx.size == 0:
                continue
            tasks.append((var, chunk_idx, iters[idx], times[idx]))

    log("Checkpoint Load: %d (variable, chunk) task(s) across %d rank(s) "
        "[%d variable(s) x up to %d chunk(s)]"
        % (len(tasks), size, len(all_vars), chunks))

    for var, chunk_idx, it_chunk, t_chunk in tasks[rank::size]:
        part_path = _part_path(cfg, var, chunk_idx)
        _, n_iter, tm = process_variable(
            backend, plane_index, var, it_chunk, t_chunk, coords, cfg,
            out_path=part_path, write_meta=False)
        print(
            "[rank %d] wrote %s (chunk %d, %d iterations) "
            "[read+resample %.1f s, write %.1f s]"
            % (rank, part_path, chunk_idx, n_iter, tm["read"], tm["write"]),
            flush=True,
        )

    comm.Barrier()

    # Merge: one rank per variable concatenates its chunks into the final file.
    for var in all_vars[rank::size]:
        _merge_variable(backend, var, coords, cfg, chunks)


def _merge_variable(backend, var, coords, cfg, chunks):
    """Concatenate a variable's chunk partials into its final file, streaming
    slice-by-slice (bounded memory), then delete the partials."""
    parts = [_part_path(cfg, var, ci) for ci in range(chunks)]
    parts = [p for p in parts if os.path.exists(p)]
    if not parts:
        log("  WARNING: no chunks found to merge for %r" % var)
        return

    resolution, _, _ = grid_bounds(cfg["grid"])
    nx, ny = resolution
    out_dtype = np.dtype(cfg["dtype"])
    compression = "gzip" if cfg["compression_level"] else None
    comp_opts = int(cfg["compression_level"]) if cfg["compression_level"] else None

    # First pass over the small index datasets to size the output.
    iters_parts, times_parts = [], []
    for p in parts:
        with h5py.File(p, "r") as h5:
            iters_parts.append(h5["iterations"][:])
            times_parts.append(h5["times"][:])
    iters_cat = np.concatenate(iters_parts)
    times_cat = np.concatenate(times_parts)
    total = iters_cat.shape[0]

    final_path = variable_path(cfg, var)
    with h5py.File(final_path, "w") as h5:
        dset = h5.create_dataset(
            "data", shape=(total, nx, ny), dtype=out_dtype,
            compression=compression, compression_opts=comp_opts,
            chunks=(1, nx, ny),
        )
        offset = 0
        for p in parts:
            with h5py.File(p, "r") as part:
                src = part["data"]
                n = src.shape[0]
                for i in range(n):  # per-slice copy keeps memory bounded
                    dset[offset + i] = src[i]
                offset += n

        h5.create_dataset("iterations", data=iters_cat.astype(np.int64))
        h5.create_dataset("times", data=times_cat.astype(np.float64))
        h5.create_dataset("x", data=coords[0])
        h5.create_dataset("y", data=coords[1])
        h5.attrs["variable"] = var
        h5.attrs["label"] = cfg["label"]
        h5.attrs["plane"] = cfg["plane"]
        h5.attrs["simdir"] = cfg["simdir"]
        h5.attrs["interp_order"] = cfg["interp_order"]
        h5.attrs["resolution"] = resolution
        h5.attrs["backend"] = backend.name

    for p in parts:
        os.remove(p)

    print("[rank %d] merged %s (%d chunks, %d iterations)"
          % (rank, final_path, len(parts), total), flush=True)
