# 2D resampling pipeline

Resample 2D slices from Einstein Toolkit simulations onto a fixed regular grid
and store them as self-describing HDF5 files — one per variable — for fast
downstream analysis.

Two interchangeable backends are provided: **postcactus** (the original library)
and **kuibit** (its actively-maintained successor). They share the same config,
MPI structure, and on-disk schema, so output files are read by the same
`read_data.py` regardless of which produced them.

## Files

| File                            | Purpose                                                |
| ------------------------------- | ------------------------------------------------------ |
| `resample_2d.py`                | Launcher — dispatches to the backend named in config.  |
| `resample_2d_data_postcactus.py`| `PostcactusBackend` — all postcactus coupling, nothing else. |
| `resample_2d_data_kuibit.py`    | `KuibitBackend` — all kuibit coupling, nothing else.   |
| `resample_common.py`            | The whole backend-agnostic pipeline: MPI/config, pickle cache, serial + parallel query, warm-state gather, HDF5 streaming (`run()`), and the `Backend` interface. |
| `config_example.yaml`           | Documented template config; copy and edit per sim.     |
| `read_data.py`                  | Helpers to read the HDF5 output back into numpy.        |

## Usage

```bash
cp config_example.yaml my_sim.yaml      # edit backend, simdir, label, grid, vars

# Launcher picks the backend from the `backend:` key in the config:
mpirun -n 8 python resample_2d.py my_sim.yaml

# Or call a backend directly (the `backend:` key is then ignored):
mpirun -n 8 python resample_2d_data_postcactus.py my_sim.yaml   # postcactus
mpirun -n 8 python resample_2d_data_kuibit.py my_sim.yaml       # kuibit
```

`-n` (MPI ranks) can be any value ≥ 1. Variables are spread round-robin across
ranks and each rank writes its own files, so there is no required process count
and no gather step. Iterations are streamed slice-by-slice into HDF5, so a rank
never holds more than one 2D slice in memory. To use **more ranks than
variables**, set `resample_chunks` (see below) to split each variable across
ranks.

Output goes to `output_dir` as `<variable>__<label>.h5`, e.g.
`rho_b__dU_10_15_linear_HR.h5`.

## Startup cost: the iteration query and the SimDir pickle cache

Startup has two costs: the **directory scan** (one recursive walk) and the
**per-variable iteration query** (parse each file's table of contents, then read
the time attribute of *every* iteration). On network filesystems the query
often dominates, because it is many small metadata reads × number of variables.

### Parallel vs. serial query (`parallel_query`)

```yaml
parallel_query: yes   # default
```

- **`yes` (default):** rank 0 does the directory scan once and broadcasts the
  SimDir; then **every rank queries its own slice of `variables`**
  (`variables[rank::size]`) and resamples exactly those. The query cost is
  divided across ranks (up to the number of variables). The per-rank warmed
  caches are gathered to rank 0, which reassembles the fully-warmed SimDir and
  writes the pickle — so the on-disk pickle is **identical** to the serial path.
- **`no`:** rank 0 queries *every* variable itself, then broadcasts the SimDir
  and the iteration tables. This is the original behaviour, kept for A/B
  comparison; both modes produce identical output files.

### Parallel vs. serial scan (`parallel_scan`)

```yaml
parallel_scan: yes    # default
scan_max_depth: 8     # bound recursion depth (both modes)
```

- **`yes` (default):** rank 0 lists the top-level subdirectories below `simdir`
  and hands each rank a share to walk recursively; the file lists are gathered
  and the index assembled on rank 0. This parallelises the metadata-bound walk
  across ranks and helps trees with **many subdirectories** (e.g. many
  `output-NNNN/` restarts). It does little for a single huge *flat* directory
  (one server still serializes that listing), and is skipped when loading a
  pickle.
- **`no`:** rank 0 walks the whole tree itself (original behaviour).

Both modes find the same files. The cheapest scan win is still pruning: set
`scan_max_depth` low, and use `simdir_exclude.dirs` (below) to skip
checkpoint/3D subtrees entirely. On top of that, the scan + query result can be
cached on disk between jobs:

```yaml
simdir_pickle:
  pickled: no                 # this run scans, then saves the pickle
  path: ./simdir_cache.pkl
```

The first run (`pickled: no`) scans normally and saves the SimDir — including
the parsed HDF5 metadata — to `path`. Subsequent runs with `pickled: yes` load
it from there and skip the directory walk and metadata parsing entirely. This
matters most on network filesystems (Lustre/GPFS), where metadata operations
dominate startup time.

**Caveat:** the pickle is a snapshot. If the simulation produces new output,
rerun once with `pickled: no` to refresh it; a stale pickle silently misses
the new iterations. (For the postcactus backend this feature needs the `dev`
branch of [MChabanov/PyCactus](https://github.com/MChabanov/PyCactus)
≥ `b71cf5d`, which made its SimDir picklable; kuibit supports pickling
natively.)

### Excluding checkpoints / 3D data from the scan

Simulation directories are often dominated by files irrelevant to 2D
resampling: checkpoint sets (`checkpoint.chkpt.it_*.file_*.h5`, one file per
MPI process per checkpoint) and per-process 3D output. The postcactus backend
can skip them (needs PyCactus `dev` ≥ the `exclude_dirs/exclude_files` commit):

```yaml
simdir_exclude:
  dirs: [checkpoints, 3D]                  # folder names pruned from the walk
  files: ["checkpoint.chkpt.*", "*.xyz.h5"]  # basename globs dropped
```

`dirs` prunes whole subtrees and is what actually cuts scan time — organise
bulky output into such folders (or move it out of the simulation directory
entirely; symlinked folders are also skipped). `files` only filters the
results (the walk still lists every entry), useful when checkpoints sit in
the same folders as the 2D data. **Make sure the globs never match the files
you resample** (`*.xy.h5` etc.) or the parfiles. The kuibit backend ignores
this option (kuibit's SimDir has no equivalent) and warns if it is set.

## Scaling resampling past the variable count (`resample_chunks`)

By default the unit of work is one variable → one file → one rank, so resampling
parallelism is capped at the number of variables (and load is uneven when some
variables have far more iterations than others). To keep more ranks busy:

```yaml
resample_chunks: 4    # default 1
```

With `resample_chunks: N`, each variable's iterations are split into `N`
contiguous chunks and the `(variable, chunk)` tasks are distributed round-robin
across ranks — so up to `variables × N` ranks do useful work. Each chunk is
written to a temporary `…__<label>.partNNNN.h5` file, then one rank per variable
**merges** the chunks (streaming slice-by-slice, bounded memory) into the single
final `…__<label>.h5` and deletes the partials. The merged output is identical
to `resample_chunks: 1`, so this is purely a scaling knob.

Useful when `-n` exceeds the number of variables, especially for long time
series. Note the extra merge pass reads and rewrites the data once; and with
many ranks all reading the AMR files at once you may become read-bandwidth
bound rather than CPU bound.

## Reading the output

```python
from read_data import load_variable, slice_at_iteration, iter_slices

rho = load_variable("resampled/rho_b__dU_10_15_linear_HR.h5")
field0, t0 = slice_at_iteration("resampled/rho_b__dU_10_15_linear_HR.h5", 1024)
for it, t, field in iter_slices("resampled/rho_b__dU_10_15_linear_HR.h5"):
    ...   # memory-friendly streaming
```

Each file stores the field stack `(n_iter, nx, ny)` plus `iterations`, `times`,
and the `x`/`y` coordinate axes, with the variable name, plane, simdir and
resolution as HDF5 attributes.

## Key differences vs. the old pickle scripts

- **No OOM:** no gather-to-rank-0; bounded per-rank memory via streaming writes.
- **Correct iterations:** queried per variable (not assumed equal to `rho_b`).
- **Times saved:** the iteration→time map is stored in every file.
- **Config-driven:** sim path, grid, variables, stride etc. live in YAML — no
  source edits to retarget a new simulation.
- **Any rank count:** works with 1..N ranks instead of requiring exactly 16.

## postcactus vs. kuibit backends

Both backends sample the same grid (`interp_order >= 1` ⇒ multilinear; `0` ⇒
nearest neighbour) and produce byte-identical schema. A cross-check on `rho_b`
showed their resampled values agree to **machine precision on most iterations**;
the only differences are a handful of points (≈0.2%) sitting exactly on
refinement-level boundaries, where the two libraries make different choices about
which overlapping AMR patch to sample. This is an inherent library difference,
not a bug. kuibit additionally reads openPMD output and is the maintained
option going forward.

**Performance** (193 iterations of `rho_b`, 400×400, single process; both
scripts print these timers):

| stage          | postcactus | kuibit            |
| -------------- | ---------- | ----------------- |
| read + resample| 29.9 s     | 170.3 s + 161.2 s |
| HDF5 write     | 6.6 s      | 6.3 s             |
| per iteration  | **0.19 s** | **1.75 s**        |

postcactus is ~9× faster here because its `read(geom=...)` only loads and
interpolates the AMR components that intersect the target grid, whereas kuibit
always reconstructs the full component hierarchy per iteration (its
`read_on_grid` is just a wrapper around the full read; checked kuibit 1.6.1).
**Prefer the postcactus backend for bulk resampling when it is available**;
use kuibit where postcactus isn't installed or for openPMD data.

### Environments used here

- postcactus: available in the `base` conda env (Python 3.8).
- kuibit: needs Python ≥ 3.9; installed in a dedicated `kuibit` conda env
  (`conda create -n kuibit python=3.11 && pip install kuibit h5py mpi4py pyyaml`).
