# Resampling pipeline

Resample slices from Einstein Toolkit simulations onto a fixed regular grid and
store them as self-describing HDF5 files — one per variable — for fast
downstream analysis.

There are **two pipelines**, sharing one config schema, one MPI structure and
one on-disk schema:

| Pipeline | Launcher | Reads | Use when |
| -------- | -------- | ----- | -------- |
| **2D** | `resample_2d.py` | the 2D slices Carpet already wrote (`*.xy.h5` …) | you want an xy/xz/yz cut — much cheaper |
| **3D → plane** | `resample_3d.py` | the full 3D output (`*.xyz.h5`, `*.file_*.h5`) | you want **any other plane**, given by a normal vector |

Because they write the same schema, `read_data.py` and
`../Frames/make_frames.py` read either without knowing which produced the file.
The 2D pipeline has two interchangeable backends — **postcactus** (the original
library) and **kuibit** (its actively-maintained successor); the 3D pipeline
currently has postcactus only.

## Files

| File                            | Purpose                                                |
| ------------------------------- | ------------------------------------------------------ |
| `resample_2d.py`                | 2D launcher — dispatches to the backend named in config. |
| `resample_2d_data_postcactus.py`| `PostcactusBackend` — all postcactus coupling, nothing else. |
| `resample_2d_data_kuibit.py`    | `KuibitBackend` — all kuibit coupling, nothing else.   |
| `resample_3d.py`                | 3D launcher — same dispatch, 3D backends.             |
| `resample_3d_data_postcactus.py`| `Postcactus3DBackend` — subclasses the 2D postcactus backend; only the read/resample step differs. |
| `plane_geom.py`                 | `PlaneSpec`: the plane's in-plane frame, its sample points, and the culling tests that keep 3D reads small. numpy only. |
| `plane_motion.py`               | Time-dependent planes: builds the per-iteration frame table (static / analytic / trajectory-driven), reads trajectory files, enforces continuity. |
| `plane_module_example.py`       | Template for an `analytic` plane — copy and edit.       |
| `inspect_plane.py`              | Print the frame table a config would produce, without resampling. Use it to check a rotating plane (above all its time offset) before launching. |
| `resample_common.py`            | The whole backend-agnostic pipeline: MPI/config, pickle cache, serial + parallel query, warm-state gather, HDF5 streaming (`run()`), and the `Backend` interface. |
| `config_example.yaml`           | Documented template config (2D); copy and edit per sim. |
| `config_example_3d.yaml`        | Documented template config (3D → plane).              |
| `read_data.py`                  | Helpers to read the HDF5 output back into numpy.        |
| `compare_output.py`             | Diff two output files — used to cross-check the 3D path against the 2D one. |

## Usage

```bash
cp config_example.yaml my_sim.yaml      # edit backend, simdir, label, grid, vars

# Launcher picks the backend from the `backend:` key in the config:
mpirun -n 8 python resample_2d.py my_sim.yaml

# Or call a backend directly (the `backend:` key is then ignored):
mpirun -n 8 python resample_2d_data_postcactus.py my_sim.yaml   # postcactus
mpirun -n 8 python resample_2d_data_kuibit.py my_sim.yaml       # kuibit

# Cutting an arbitrary plane out of the 3D output instead:
cp config_example_3d.yaml my_sim_3d.yaml       # edit `plane`, then
mpirun -n 16 python resample_3d.py my_sim_3d.yaml
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
series. The merge is cheap: chunks are copied verbatim (compressed bytes
relocated with `read_direct_chunk`/`write_direct_chunk`, no decompress/
recompress), so it is disk-bandwidth bound, not CPU bound. With many ranks all
reading the AMR files at once you may instead hit read-bandwidth limits during
the resample phase.

## Cutting an arbitrary plane out of the 3D output

`resample_3d.py` reads the full 3D AMR output and interpolates it onto a 2D
grid lying in a plane you specify. Everything outside the `plane` key behaves
exactly as above — same scan/query/pickle/chunk options, same output schema.

```yaml
plane: xz                      # shorthand: the plane through the origin

plane:                         # or a mapping, for any other plane
  normal: [0.0, 1.0, 1.0]      # three numbers, or an axis name (x/y/z);
                               # need not be a unit vector, sign is irrelevant
  origin: [0.0, 0.0, 0.0]      # optional: a point the plane passes through
  u_axis: x                    # optional: pins the +u direction in the plane
```

The `grid:` block is then read in the plane's own coordinates `(u, v)`, and
those are what the output stores as its `x` and `y` axes — which is why
`../Frames` and the notebooks need no change.

### In-plane axes

A normal fixes the plane but not the rotation of coordinates inside it. The
default convention: **drop the coordinate axis most aligned with the normal and
keep the other two, in x < y < z order** (projected into the plane and
orthonormalised). So an axis-aligned normal reproduces the 2D pipeline exactly —
normal z gives `(u, v) = (x, y)`, normal y gives `(x, z)`, normal x gives
`(y, z)` — and flipping the normal's sign never flips the picture. Override it
with `u_axis`. Every file records `plane_normal`, `plane_origin`,
`plane_u_axis`, `plane_v_axis` as attributes, so a plot is never ambiguous
about what it shows.

Note that for an oblique plane the frame axes carry no physical name: label
your plots `u`/`v` (in M or km) rather than `x`/`y`.

### Planes that rotate with time

`plane.kind` selects between a fixed plane and two moving ones. The output
schema is unchanged — `x`/`y` are still the fixed in-plane axes — but the
geometry becomes per-iteration datasets rather than attributes.

| `kind` | The plane is | Output geometry |
| ------ | ------------ | --------------- |
| `static` (default) | fixed; everything above applies | `plane_*` attributes |
| `analytic` | whatever a Python module of yours returns per time | `plane_*` datasets |
| `trajectory` | tied to a binary read from a trajectory file | `plane_*` + `bh*` datasets |

#### Trajectory-driven

The usual form names a **second in-plane vector**; the plane then spans it
together with the separation vector, so *both bodies lie in the plane at every
time* and it tilts whenever they leave it.

```yaml
plane:
  kind: trajectory
  file: auto                     # or a path; auto = AnalyticalSpacetime::traj_table_name
  time_offset: auto              # or a number; auto = AnalyticalSpacetime::AST_t0
  origin: com                    # com | midpoint | bh1 | bh2 | [x,y,z] | module
  second_vector: orbital
  u_axis: separation             # co-rotating (default), or x/y/z for inertial
```

Prefer the presets defined by the **orbit** over those defined by the
coordinate axes:

| `second_vector` | The plane holds the separation and… | Gives |
| --------------- | ----------------------------------- | ----- |
| `orbital` | the relative velocity `v₂ − v₁` | the **instantaneous orbital plane** — exact for an inclined, precessing or eccentric orbit |
| `orbital_meridional` | the orbital angular momentum axis `d × v_rel` | a **meridional cut through both bodies**, vertical w.r.t. the *orbit* and perpendicular to the orbital motion |
| `spin1` / `spin2` | that body's spin axis | that body's **meridional plane**, facing the companion |
| `equatorial` | `ẑ × d` | the coordinate-frame version of `orbital` |
| `meridional` | `ẑ` | the coordinate-frame version of `orbital_meridional` |

`equatorial`/`meridional` quietly assume the orbit lies in `z = 0`; the
velocity-based pair is the same thing done properly, and reduces to them exactly
when that assumption holds. A **constant** vector is also accepted, but a binary
that turns through it makes the plane degenerate — the pipeline measures the
angle and refuses below 0.06°, warning below 2.9°.

For a plane that need *not* contain the separation, give `normal` instead
(`orbital` | `spin1` | `spin2` | an axis | a vector | `module`). `normal: spin1`
with `origin: bh1` is one body's **equatorial plane** — the one its accretion
flow wants to align with.

`u_axis` fixes which in-plane direction is `+u`, i.e. the output's `x`:

- `separation` (default) is **co-rotating**: the bodies sit still at
  `u = ±(distance to the origin)`, `v = 0`, and only the gas moves.
- an axis name is **inertial**: the frame stays as near that fixed direction as
  the plane allows, so the binary visibly orbits inside it.

#### Analytic

Copy `plane_module_example.py`, edit it, and point at it. Only `normal(t, it)`
is required; `origin(t, it)` and `u_axis(t, it)` are optional. Only rank 0
imports it, so it may do anything.

```yaml
plane:
  kind: analytic
  module: ./my_plane.py
```

#### Why the frames are precomputed

Ranks resample iterations **out of order** (round-robin, and split into chunks),
so a frame can never be derived from the previous one. But the static in-plane
convention — drop the axis most aligned with the normal — is *discontinuous*: as
the normal rotates past a boundary, `u` and `v` swap and the movie flips. So the
whole table is built on rank 0 right after the iteration query, where continuity
can be imposed *and verified* before any field data is read. That also reads the
trajectory file once instead of once per rank.

The startup log reports what it found:

```
  time offset (from parfile AST_t0): 0.203891322579
  second in-plane vector: orbital angular momentum axis d x v_rel
  separation and second in-plane vector stay at least 90.00 degrees apart
  frame turns 1694.3 degrees in total over 397 output frame(s); largest step 4.33 degrees
  bh1_uv: |u| up to 15.0000, |v| up to 0.000e+00, off-plane residual 1.776e-15
```

That `|v| up to 0` is the self-check: both bodies are exactly in the plane.

#### Two things to watch

**The time offset.** `t_trajectory = t_simulation + time_offset`. Getting it
wrong rotates every frame by a constant angle *with no other symptom* — the
movie still looks plausible. Hence `auto`, which reads
`AnalyticalSpacetime::AST_t0` from the parfile (searching all of them, and
warning if they disagree). To check it by hand:

```bash
python inspect_plane.py my_sim_3d.yaml --times-from ../2d/rho_b__run_xz.h5
```

The body positions it prints at `t = 0` must match `$x_punc1`/`$y_punc1` in the
parfile.

**Output cadence.** With `u_axis: separation` the frame turns with the binary, so
the output has to resolve the **orbit**, not just the flow. On the 30M run the
orbital period is ~1077 M: the 2D cadence (every 1024 iterations, 397 frames)
gives 4.3°/frame, but the 3D cadence (every 4096, ~100 frames) gives 17°/frame
and the movie visibly steps. The log reports the largest step and warns above
30°.

### Why this reads so little

One iteration of one variable of a production 3D run is several GB spread over
thousands of AMR components — reading it to fill one 2D slice would be absurd.
So the backend walks the components itself and, for each, uses only the HDF5
*attributes* (origin, spacing, shape) to ask whether the plane's sampled patch
can reach it at all; that test is exact (the feasible `(u, v)` region of a box
is a convex polygon, clipped analytically in `plane_geom.uv_window`). Only the
components it hits are read, and only the bounding box of the points that land
inside them, as an HDF5 hyperslab. Interpolation is then one vectorised
`map_coordinates` call per component. Levels are processed coarse to fine so
finer data wins, and a component only claims points in its interior — ghost
zones are read for the stencil but belong to a neighbour, exactly as in
postcactus' own `sample_intersect`.

Measured on the 30M BBH production run (9 levels, `dx` 8 → 0.031, 64 files ×
4608 components per variable per iteration, 800×800 output):

| plane | components read | data read | per iteration |
| ----- | --------------- | --------- | ------------- |
| `xz` (axis-aligned) | 360 of 4608 | 81 MiB | ~5.5 s |
| tilted 45° | 546 of 4608 | 458 MiB | ~8 s |
| *a full 3D read would be* | 4608 | ~6.4 GiB | — |

The oblique plane reads more because a diagonal cut through a component needs
most of that component's bounding box. That cost is set by the plane's
orientation, **not** by the output resolution — halving `grid.resolution` will
barely speed it up. Each variable logs its own figures on its first slice:

```
[rank 3] rho_b: plane touches 360 of 4608 component(s) at it=0, 80.9 MiB read (100.0% covered)
```

`% covered` is the fraction of the output grid some refinement level reached;
well under 100% means the plane sticks out of the domain (those points get
`outside_value`).

Because a 3D slice costs seconds rather than milliseconds, `resample_chunks` is
worth much more here than for 2D — set it so that `variables × chunks` ≥ ranks.

### Cross-checking it

`plane: xy|xz|yz` samples exactly the grid the 2D pipeline does, so the two
must agree to round-off. That makes a very strong end-to-end test — run both
with the same `grid:` and `interp_order`, then:

```bash
python compare_output.py 3d_out/rho_b__run_3d_xz.h5 2d_out/rho_b__run_xz.h5
```

On the production run above this gives a worst relative difference of
**5×10⁻¹³** over 640 000 points (median ~10⁻¹⁵) — i.e. the two paths are the
same calculation. Run it after touching the pipeline.

### Gotchas

- **Use a separate SimDir pickle.** A pickle written by a 2D run whose
  `simdir_exclude` pruned the 3D output directory is missing exactly the files
  this pipeline needs. Conversely a 3D config should prune the *2D* directories.
  The backend refuses to start if `simdir_exclude.files` would hide 3D data,
  but it cannot know which of your directory *names* holds it.
- **3D output is usually written far less often** than 2D (every 4096 vs 1024
  iterations on the run above), so the movie has fewer frames — that is the
  data, not the pipeline. `iteration_stride` is rarely needed.
- `interp_order` accepts 0 (nearest) and 1 (linear) as in 2D; 2–5 select spline
  interpolation of that order.

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
resolution as HDF5 attributes. Files from the 3D pipeline are the same in every
respect — `x`/`y` hold the in-plane `(u, v)` axes — and carry the plane's
geometry in extra attributes (`plane_normal`, `plane_origin`, `plane_u_axis`,
`plane_v_axis`, `source_dims`, `plane_kind`, `outside_value`).

For a **rotating** plane the geometry cannot be an attribute, so those four
`plane_*` become `(n_iter, 3)` datasets instead, and a trajectory-driven plane
adds the bodies:

| Dataset | Shape | |
| ------- | ----- | - |
| `plane_normal`, `plane_origin`, `plane_u_axis`, `plane_v_axis` | `(n_iter, 3)` | the frame at each stored iteration |
| `bh1_position`, `bh2_position` | `(n_iter, 3)` | body positions in simulation coordinates |
| `bh1_uv`, `bh2_uv` | `(n_iter, 2)` | the same, in the plane's own `(u, v)` — ready to overplot as markers |
| `separation` | `(n_iter,)` | `|x₂ − x₁|` |

`make_frames.py` ignores all of it and renders as before; the datasets are there
so a panel can annotate the bodies, and so a finished figure is never ambiguous
about the cut it shows.

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
