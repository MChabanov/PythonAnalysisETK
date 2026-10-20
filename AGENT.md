# Operating notes

Practical knowledge for *running* this pipeline on a cluster, as opposed to
what the code does — that is in `Resample/README.md` and `Frames/README.md`,
which stay the reference for options, schema and physics. This file holds the
things that were learned the hard way: batch-environment setup, how to tell a
job succeeded where the usual tools are missing, how to verify a moving-plane
run after the fact, and the movie-encoding recipes.

Written after the co-rotating meridional run described in §3, which is the
worked example §1-6 refer to; §7-8 record the field-line congruence work and
the single-snapshot cuts built on it, §9 what is still open, §10 the planes
about each hole of the precessing 3C 186 run, §11 the sphere resampler.

---

## 1. Submitting jobs (green-prairies / OpenHPC + SLURM)

### Batch shells do not inherit the interactive environment

A batch shell lands in conda `base`, where `import postcactus` raises
`ModuleNotFoundError`. Any `job.batch` that resamples or renders must set up its
own environment. Prepend:

```bash
if ! command -v mpirun >/dev/null 2>&1; then
    source /etc/profile.d/lmod.sh 2>/dev/null || true
    module load gnu12 openmpi4 2>/dev/null || true
fi
source ~/anaconda3/etc/profile.d/conda.sh
conda activate etk-analysis
```

**Order matters.** `mpi4py` in `etk-analysis` is built against OpenHPC's
**OpenMPI 4.1.6**, so the module must be loaded *and* the conda env must not
shadow it with an MPI of its own. `ffmpeg`/`ffprobe` also live in
`etk-analysis`, not in `base`.

**numba** (2026-10-19, user): `pip install numba==0.61.2` in `etk-analysis`
added only numba 0.61.2 and llvmlite 0.44.0 (pip freeze before/after);
numpy stays the pip-installed 1.26.4. Do **not** `conda install numba`: it
pulls numba 0.68 with a conda-forge numpy 2.4 next to the pip numpy 1.26 -
two numpys and a major-version jump under postcactus. Verified after the
install on a compute node: postcactus/mpi4py/h5py import, `test_sphere_geom.py`
ALL OK, a sphere resample bit-identical to before; a parallel `@njit`
trilinear interpolation of 2e6 points takes 0.011 s on 6 threads (scipy
`map_coordinates` 0.28 s). For the field-line tracer. Undo: `pip uninstall
numba llvmlite`.

The launch line used on this cluster:

```bash
mpirun -mca pml cm -mca btl self,vader -mca mtl psm2 -x PSM2_MULTI_EP=0 \
    python ~/PythonAnalysisETK/Resample/resample_3d.py config.yaml
```

### Choosing a partition: `debug` is the shared one

The `short` partition named throughout this file **no longer exists**. As of
2026-09-03 the partitions are `gp-default`, `gp-long`, `wl-default`, `wl-long`,
`all-default`, `lm-largemem` and `debug`; a stale `--partition=short` is
rejected at submit time with *"invalid partition specified"*.

Which to pick is not just about the time limit:

| | | |
| - | - | - |
| `gp-default` / `gp-long` / `wl-default` / `wl-long` | `OverSubscribe=EXCLUSIVE` | a job takes a **whole node**. `squeue` showing every node allocated therefore does *not* mean every core is busy - it means whole nodes are claimed, and nothing of yours starts until one frees. With the queue full of 20-day-limit jobs that is an unbounded wait. |
| `debug` | `ExclusiveUser=NO`, `OverSubscribe=NO` | **multi-user**: jobs from different people share `lm-n0001` (48 cores, 12 h max), each getting its own CPUs. A short job schedules immediately alongside whatever else is running. |

So a full resample - 100 iterations, hours on 16 ranks - belongs on
`gp-default`, and it is worth the wait. But the **frame/movie steps are
minutes**, and queuing those behind a whole-node allocation is the wrong trade:
send them to `debug`. Measured: the 100-frame `corotating_meridional_Bfield`
render started within 20 s and finished in about 30 s on 16 ranks, sharing the
node with another user's `lm-largemem` job. The same goes for a **one-iteration
resample** (§8): 21 variables on 21 ranks took 75 s cold, 11 s with a warm
`toc_cache`.

Two courtesies, since the node is shared: do **not** pass `--exclusive`, and
keep `--ntasks-per-node` modest. `debug` sets `DefMemPerCPU=2048`, so 16 ranks
get 32 GB by default - ample for rendering a box, where a rank holds only a
couple of 800x800 float64 slices at a time, but not for the congruence scalars
on a **whole** 800x800 plane (~1.3 GB per rank before the figure is drawn): the
`*_box40` jobs ask for `--mem-per-cpu=4G`.

### Traps in batch scripts

- **`$0` is not the script.** SLURM runs a spool copy, so `$0` is
  `/var/spool/slurmd/jobNNNNN/slurm_script` and `dirname "$0"` is meaningless.
  Use `$SLURM_SUBMIT_DIR` — SLURM starts the script in the submission
  directory, so bare `PWD` is right too.
- **`/tmp` is node-local.** `#SBATCH --output=/tmp/...` writes on the compute
  node and is invisible from the login node. Log to a shared filesystem.
- **A queued job holds a snapshot of the script.** SLURM copies it at submit
  time, so editing the file does not change an already-queued job — cancel and
  resubmit.
- **`srun python` breaks anything that imports mpi4py** (every
  `make_frames.py`): OpenMPI here has no SLURM PMI support, so `MPI_Init`
  aborts under a direct `srun` launch. Plain `python` inside an sbatch script
  works; for a quick one-off, `sbatch --wait -p debug -n 1 --wrap "..."`.
  Scripts that do not import mpi4py (`plot_snapshot.py`) are fine under srun.
- **Plain `python` + mpi4py = pinned to ONE core** (2026-10-20). mpi4py makes
  the process an MPI singleton and OpenMPI binds it to core 0, so numba (or
  any) threads all share that core: `taskset -cp <pid>` showed affinity `0`,
  32 threads at 100% total. `OMPI_MCA_hwloc_base_binding_policy=none` did
  **not** help. Launch one threaded rank as `mpirun -n 1 --bind-to none ...`,
  several as `mpirun ... --map-by ppr:4:node:PE=8` (each rank bound to its 8
  cores); `Integrals/` scripts print `numba: N thread(s), M core(s)` and warn
  when M < N.
- **Logs through a pipe are block-buffered**: `python ... | grep -v x > log`
  shows nothing until the end; use `grep --line-buffered`. `/usr/bin/time` does
  not exist on the compute nodes (the job died at once).

### Judging success without `sacct`

`sacct` returns *"Slurm accounting storage is disabled"* on this cluster, so
exit states are **not** queryable after the fact. Judge from the logs and the
output instead:

```bash
squeue -u $USER                        # empty => finished (or failed)
tail -5  Resample/job.out              # want: "Checkpoint End: all variables written to ..."
tail -5  Frames/job.out                # want: "Movie written to ..." / "Done: N frame(s) rendered."
ls Resample/<out_dir>/*.h5 | wc -l     # one per variable
ls Frames/<frames_dir>/*.png | wc -l   # one per frame
```

### Chaining the stages

`--dependency=afterok:<jobid>` chains resample → frames → encoding, and the
whole chain then completes with **no login session or agent present**.

### Interrupted resampling is benign

With `resample_chunks: 1` each rank streams its own variable's file
independently, so a killed job leaves finished variables as complete, valid
files and only the in-progress ones truncated. Rerun with
`simdir_pickle.pickled: yes` and trim `variables:` to whatever is missing.

---

## 2. postcactus quirks seen in production

- `"Unused (redundant) folder .../output-000N-Vista/HDF5_3D"` on stderr is
  normal restart de-duplication, **not** an error.
- A trajectory `.h5` sitting in the simdir root is mis-identified as a 3D grid
  variable named after the file (e.g. `CircularOrbit_a09_m05_sep30_rot`).
  Harmless — just never request it as a variable.
- `Miscellaneous/` holds ~44 000 files per output directory and dominates the
  directory scan. Keep it in `simdir_exclude.dirs`.
- A 3D config must **invert** the 2D `simdir_exclude`: prune `HDF5_2D`, keep
  `HDF5_3D` — and use its own pickle. See `Resample/README.md` § Gotchas.

---

## 3. Worked example: co-rotating meridional movie

The 3D counterpart of a 2D `xz_plane` density movie — a meridional cut through
both black holes in a frame that turns with the binary, cut out of the full 3D
output using the trajectory's **velocity** data.

```
<sim>/3d_analysis/corotating_meridional/
├── Resample/config_..._3d_merid.yaml   the plane + grid spec (heavily commented)
├── Resample/job.batch                  16 ranks, `gp-default`, 5:45:00
└── Frames/                             make_frames.py + movie_config.yaml + job.batch
```

The plane:

```yaml
plane:
  kind: trajectory
  file: auto                        # -> the parfile's traj_table_name
  time_offset: auto                 # -> AnalyticalSpacetime::AST_t0
  origin: com
  second_vector: orbital_meridional # d x v_rel — the velocity-based definition
  u_axis: separation                # co-rotating: holes pinned at u = ±sep/2
```

`orbital_meridional` and `orbital` are preferred over the coordinate-frame
`meridional`/`equatorial` even for an exactly planar orbit where they coincide
numerically: they cost nothing and stay correct for an inclined or precessing
binary.

### Mimicking an existing 2D movie

To reproduce a 2D movie's look, copy the **edited** `make_frames.py` from that
analysis directory, not the repo one — production copies usually differ from the
repo defaults (colormap, `vmin`/`vmax`, title units).

Two things must change, and one usually does:

| | why |
| - | --- |
| `XLABEL`/`YLABEL` → `$u~[M]$` / `$v~[M]$` | the axes are the *plane's* frame, not the grid's. Keeping `x`/`z` would be wrong. |
| derived diagnostics dropped | quantities the run wrote only for its 2D slices (e.g. `sigma`, `inv_beta`) have no 3D data to cut. Recover them in a panel lambda: `sigma ~ smallb2/rho_b`, `inv_beta ~ smallb2/(2P)` — check your own definitions. |
| `resample_chunks` → 1 | when *variables = ranks* with equal iteration counts the split is already balanced, and each rank then reads its own variable's files instead of competing for the same ones. No merge step. |

### Measured cost

16 variables × 100 iterations, 800×800 over ±40 M, `interp_order: 1`, 16 ranks
on one node of the 30M BBH production run:

| | |
| - | - |
| per slice, rotating plane | ~535 of 4608 components, ~514 MiB read |
| per iteration per rank | **34.7 s** |
| total wall (16 vars in parallel) | **3506 s ≈ 58 min** |
| frame rendering + mp4, 100 frames | a few minutes on 16 ranks |

Budget generously: a cold single-rank estimate of ~50 s/slice suggested ~1.5 h
per rank, and the real figure was lower. This ran under the old `short`
partition's 6 h cap; `gp-default` now allows 2 days, so the time limit is no
longer the binding constraint — the wait for a free whole node is.

---

## 4. Verifying a moving-plane run

### Before launching — `inspect_plane.py`

Reads no field data, takes seconds:

```bash
python ~/PythonAnalysisETK/Resample/inspect_plane.py config.yaml \
    --times-from ../../2d_analysis/xz_plane/Resample/<out>/rho_b__<label>.h5
```

**The check that matters is the body positions at `t = 0`** — they must equal
`$x_punc1`/`$y_punc1` from the parfile. A wrong `time_offset` rotates every
frame by a constant angle *with no other symptom*; the movie still looks
plausible. Also confirm both bodies report `|v| = 0` (they are exactly in the
plane) and that the frame's largest step per frame is tolerable.

### After the run — is the frame actually co-rotating?

The output files carry `plane_*`, `bh1_uv`/`bh2_uv`, `bh*_position` and
`separation` per iteration precisely so this can be checked. The two failure
modes worth ruling out are a **label swap** (the holes trading places mid-movie)
and a **frame flip** (the picture mirroring):

```python
import h5py, numpy as np
f = h5py.File("<out>/rho_b__<label>.h5")
u1, u2 = f["bh1_uv"][:], f["bh2_uv"][:]
p1, p2 = f["bh1_position"][:], f["bh2_position"][:]

# 1. holes stay on their own side, and stay put
assert len(set(np.sign(u1[:, 0]))) == 1 and len(set(np.sign(u2[:, 0]))) == 1
print("|v| max", np.abs(u1[:, 1]).max(), np.abs(u2[:, 1]).max())   # want 0

# 2. no label swap: each hole is always nearer its own previous position
d_same  = np.linalg.norm(p1[1:] - p1[:-1], axis=1)
d_cross = np.linalg.norm(p1[1:] - p2[:-1], axis=1)
assert (d_cross > d_same).all()

# 3. no mirror flip: frame vectors never reverse between frames
for k in ("plane_u_axis", "plane_v_axis", "plane_normal"):
    V = f[k][:]
    assert np.einsum("ij,ij->i", V[1:], V[:-1]).min() > 0

# 4. u_axis really points bh1 -> bh2
d = p2 - p1; dh = d / np.linalg.norm(d, axis=1)[:, None]
print("dot(sep_hat, u_axis)", np.einsum("ij,ij->i", dh, f["plane_u_axis"][:]).min())  # want 1
```

Confirmed on the §3 run: 0 swaps, 0 sign flips, `|v| = 0.0e+00` exactly,
`dot = 1.000000000000`, monotonic rotation (1694.3° total, steps 16.9–17.3°, 0
reversals), and all 16 variable files carrying byte-identical plane/BH tables.

An independent check on the *field* data, which does not trust the metadata: the
density peak in each half-plane must sit next to the stated body position in
every frame (it landed within 0.43 M ≈ 4 cells, with 0 frames on the wrong side).

### Pipeline-level cross-check

`plane: xy|xz|yz` samples exactly the grid the 2D pipeline does, so
`compare_output.py` against a 2D run is a strong end-to-end test — 5×10⁻¹³
worst relative difference on the production run. Run it after touching the
pipeline. See `Resample/README.md` § Cross-checking it.

---

## 5. Movie encoding

`assemble_movie()` in `Frames/make_frames.py` already emits H.264 High /
`yuv420p`, which is what makes a movie play in QuickTime and Safari. To confirm
a new movie matches a reference one:

```bash
conda activate etk-analysis        # ffmpeg/ffprobe live here, not in base
ffprobe -v error -select_streams v:0 \
  -show_entries stream=codec_name,profile,level,width,height,pix_fmt,r_frame_rate,nb_frames \
  -show_entries format=duration -of default=noprint_wrappers=1 movie.mp4
```

**`+faststart`** (streaming on Mac/Safari — puts `moov` ahead of `mdat`) is a
remux, no re-encode, no quality loss:

```bash
ffmpeg -v warning -i M.mp4 -c copy -movflags +faststart M.tmp.mp4 && mv M.tmp.mp4 M.mp4
```

**A different duration** needs no re-render, only a re-encode from the PNGs:

```bash
ffmpeg -y -framerate 6 -i frames_%04d.png -c:v libx264 -pix_fmt yuv420p \
  -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -r 6 -movflags +faststart out_6fps.mp4
```

### The output-cadence caveat

3D output is typically written far less often than 2D — every 4096 vs 1024
iterations on the 30M run, i.e. **100 frames against 397**. At the inherited
`FRAMERATE: 25` the movie is 4.0 s rather than 15.9 s, and because a co-rotating
frame then turns **~17° per frame** (orbital period ~1077 M) the motion visibly
strobes. Re-encoding at 6 fps fixes the *duration* (16.7 s); **nothing fixes the
stepping** — it is the simulation's output cadence, not the pipeline. Only
denser 3D dumps, or an inertial view (`u_axis: x`, where the grid stops
rotating), would.

Keeping the 2D movie's `FRAMERATE` and shipping a second slower variant
alongside is a reasonable compromise.

---

## 6. Known gaps

- **BH markers are not plotted.** `bh1_uv`/`bh2_uv` are stored already in plot
  coordinates, but `Frames/make_frames.py` ignores them. Wiring them into a
  panel was deliberately deferred — the decision was "store positions only, no
  `Frames/` changes".
- **The 2D analysis dirs' `job.batch` files predate §1** and assume the
  submitting shell has postcactus. They need the environment block before they
  can be rerun.

---

## 7. Field-line congruence diagnostics — where the work stands (2026-10-06)

Goal: maps of the expansion Θ_B, shear Σ_B, twist Ω_B and curvature κ_B of the
magnetic field-line congruence, `~/magnetic_field_line_congruence.tex`,
"Practical evaluation" section. **Equation numbers:** the compiled manuscript's
E74/E78/E79/E80 are eqs. 73/77/78/79 of the tex compiled standalone (one
extra numbered equation precedes the section in the manuscript).

### Directory structure

```
~/PythonAnalysisETK/Resample/              (repo, branch main)
├── derivatives_3d.py                      NEW: centred FD (1st/2nd, mixed;
│                                          accuracy 2/4/6), stencil trimming,
│                                          `derivatives:` config parsing. numpy only
├── resample_3d_data_postcactus.py         `derivatives:` option + `geometry_cache`
│                                          + `share_tocs`/`toc_cache`; docstring
│                                          "Where a cold slice's time goes"
├── toc_share.py                           NEW: TOCs shared across variables, and the
│                                          `toc_cache` file (load / merge-save)
├── resample_common.py                     extra_output_attrs(cfg, var); 2D backends
│                                          refuse `derivatives`
├── config_example_3d.yaml, README.md      documented ("Spatial derivatives")

A=/lagoon/michailchabanov/Analysis/BBH_production/inspiral_IGM_sameBHint_HHR_newcool_drift_mag_LorGrid_flux/3d_analysis
$A/toc_cache_HHR_newcool_30M_flux_3d.pkl   shared TOC cache of every 3D config here
$A/derivatives_xy/
├── Resample/                              3D -> static xy plane, 800x800 over +-40 M
│   ├── config_30M_HR_q1_flux_3d_xy.yaml   22 variables + 27 derivatives, accuracy 4,
│   │                                      resample_chunks: 100
│   ├── job.batch                          gp-default, 4 nodes x 25 ranks (job 7433)
│   ├── config_30M_HR_q1_flux_3d_xy_dalp.yaml   + d_k alpha only (same label/dir)
│   ├── job_dalp.batch                     gp-default, 1 node x 25 ranks, 4.6 min (job 7522)
│   ├── simdir_cache_HHR_newcool_30M_flux_3d.pkl   copy of corotating_meridional's
│   └── HR_newcool_30M_flux_resampled_3d_xy/       52 files, 100 its each, ~22 GB:
│         rho_b P eps lcool smallb2 vel_0..2 betax..z alp w_lorentz
│         Bx By Bz gxx..gzz   dB{x,y,z}_d{x,y,z}   dg{xx..zz}_d{x,y,z}
│         dalp_d{x,y,z}
├── theta_sigma_omega_kappa/Frames/        BH1 movie (6 panels, 2x3)
│   ├── make_frames.py                     congruence_scalars() = the tex recipe
│   ├── movie_config.yaml                  40 input paths, scales, THETA_FORM
│   ├── job.batch                          debug, 40 ranks, ~5 min
│   ├── test_congruence_scalars.py         validation; must print ALL OK
│   ├── README.md                          definitions, checks, colour choices
│   ├── frames__HR_newcool_30M_flux_3d_xy/ PNGs + current movie (LINEAR scale)
│   ├── ..._BH1_..._3d_xy__log.mp4         log scale, floor 1e-2
│   └── ..._BH1_..._3d_xy__Theta_projector.mp4   first version (old Θ, log)
├── theta_sigma_omega_kappa_BH2/Frames/    BH2 movie: same code, config differs
│                                          (BH: 2, log scale, floor 1e-1)
├── ratios_current/Frames/                 BH1 movie (2x3): (S^2-O^2)/(S^2+O^2),
│   ├── make_frames.py                     S_B/k_B, |J|/|B| / rho_0, sigma + lines,
│   │                                      B^z/|B| (no lines). congruence_scalars
│   │                                      copied unchanged + current_density()
│   ├── movie_config.yaml                  44 input paths, measured ranges
│   ├── job.batch                          debug, 40 ranks, ~2 min (job 7523)
│   ├── test_congruence_scalars.py, test_current_density.py   both print ALL OK
│   ├── README.md                          definitions, units, identities, ranges
│   └── frames__HR_newcool_30M_flux_3d_xy/ 100 PNGs + ratios_current_BH1_...mp4
├── ratios_current_smoothed/Frames/        the same movie, every derivative field
│                                          Gaussian-smoothed (SMOOTH_SIGMA 0.2 M, full
│                                          plane) before computing; test_smoothing.py
│                                          (sigma 0 == ratios_current bit-for-bit); job 7528
├── theta_sigma_omega_kappa_box40/Frames/  the linear-scale congruence movie and the
└── ratios_current_smoothed_box40/Frames/  smoothed ratios movie on the WHOLE plane,
                                           [-40,40]^2 M fixed (BH: null, BOX_CENTER,
                                           HALF_WIDTH 40); colour ranges re-measured
                                           there; test_box.py (cut to the BH1 box ==
                                           the BH1 movie bit-for-bit); 4G/rank on
                                           debug; jobs 7533/7534
```

Panels of the congruence movie: Θ_B, Σ_B, Ω_B / κ_B, σ = b²/(ρ₀h) + lines,
B^z/|B| + lines. The last two copy `../2d_analysis/xy_sigma_BH1` and
`xy_Bz_overnormB_BH1` (lines 1.15x thicker, lw 0.3795). Box: 20 M following the
BH from the trajectory table (the `*_box40` movies: the whole plane, fixed). 6
fps, because 3D output gives 100 frames against 397 for the 2D movies.

### State

- The derivative code is committed on `corot` as `adb9b7a`, the TOC sharing as
  `b5e8960`; `corot` was merged into `main` on 2026-10-16 (PR #3, `9558ba2`).
  Since then only this file has changed in the repo - everything in §7-8 after
  `b5e8960` lives in the analysis directories, not in git.
- `derivatives_3d.py` + the backend path: polynomials exact, convergence rates
  2/4/6, synthetic multi-level AMR on a tilted plane exact to 1e-13, and on
  real data the resampled ∂f equals a finite difference of the resampled f
  bit-for-bit wherever one level owns the stencil (xz/yz test planes).
- `geometry_cache` (default on): bit-identical output, but saves only ~1 s per
  slice — see "Costs" below. Harmless; kept.
- Θ_B is computed in the **divergence-free form** Θ_B = −ℓ^k ∂_k|B|/|B| (tex
  eq. 33, `THETA_FORM: divfree`). The projector form P^ij𝓑_ij also carries the
  data's centred-difference ∇·B (~2% of |∂B|), which does **not** shrink with
  stencil order (2.6% → 2.1% → 2.0% at accuracy 2/4/6), so it is a fixed
  ~25% pointwise bias in Θ. The divergence-free form is also ~30% less
  stencil-sensitive. Θ_B stays the least robust scalar pointwise (90th-percentile
  change ~25% between accuracy 4 and 6; Σ, Ω, κ change by 1–2%).

### Rerunning

```bash
cd $A/derivatives_xy/Resample && sbatch job.batch          # ~50 min, 4 gp nodes
cd $A/derivatives_xy/Resample && sbatch job_dalp.batch     # ~5 min, 1 gp node
cd $A/derivatives_xy/theta_sigma_omega_kappa/Frames && sbatch job.batch   # ~5 min
cd $A/derivatives_xy/theta_sigma_omega_kappa_BH2/Frames && sbatch job.batch
cd $A/derivatives_xy/ratios_current/Frames && sbatch job.batch            # ~2 min
cd $A/derivatives_xy/ratios_current_smoothed/Frames && sbatch job.batch   # ~2 min
cd $A/derivatives_xy/theta_sigma_omega_kappa_box40/Frames && sbatch job.batch
cd $A/derivatives_xy/ratios_current_smoothed_box40/Frames && sbatch job.batch
# each Frames/ directory: python test_*.py -> ALL OK (inside an sbatch job, §1)
```

Movie knobs live in `movie_config.yaml`: `CONG_SCALE` (log/linear),
`CONG_VMIN/VMAX`, `THETA_LINTHRESH/THETA_VMAX` (log), `CONG_LIN_VMAX`,
`THETA_LIN_VMAX` (linear), `THETA_FORM` (divfree/projector), `BH`
(1/2, or null with `BOX_CENTER`), `HALF_WIDTH`, `SMOOTH_SIGMA`, `STREAM_*`.
Typical values in the BH1 box: medians Σ 1.1, Ω 0.9, κ 0.5, |Θ| 0.25 /M; 95th
percentiles 8.6 / 8.4 / 5.3 / 3.6; 99.5th 20–32.

### Findings worth not rediscovering

- **Costs.** Carpet stores each 3D component as **one gzip chunk**, so any
  hyperslab read inflates the whole component. The log's "MiB read" counts only
  the slab kept. At 100 ranks Lustre saturates: plain slices took ~100 s, only
  ~2.2x the throughput of 16 ranks. Derivative slices took **8–10 s**, because
  the same rank had just read the base field at the same iteration (page
  cache). So list plain variables before their derivatives (the config does),
  and set `resample_chunks` so that each rank handles one or a few iterations
  of *every* variable (`resample_chunks: 100` for 100 iterations: rank r gets
  iteration r of all 49). **Where a cold slice's time goes** (profiled, one
  rank, 40–70 s): data reads only 6–8 s; parsing the 64 per-process files'
  tables of contents 14–34 s (the pickle holds the parsed TOC of only one file
  per restart, so every slice re-lists ~200 dataset names per file at ~1.4 ms
  each - cold-Lustre latency; a single-pass `links.iterate` is no faster);
  opening them 8–15 s; dataset lookups 9–14 s.
- **Fine-to-coarse skipping was tried and reverted (2026-10-05).** Walking
  levels finest first, skipping components whose points finer levels own and
  every level coarser than the first that covers the plane: bit-identical on 21
  slices (plain, derivatives, accuracy 6, co-rotating plane), ~40% fewer
  datasets touched per xy slice (levels 0–3 = 2048 of 4608 never scanned) -
  but those are the small coarse ones, and cold slices on `debug` came out
  ~10% *slower* in 12 of 15 pairs (faster in 2 of 2 on the login node);
  reversing the in-level access order made it worse still. Presumably the
  coarse-first walk lets Lustre readahead serve the later fine-level reads. The
  version is not in git.
- **Shared tables of contents (2026-10-05, `toc_share.py`)** - what did pay.
  The files one process wrote into one output directory hold the same
  datasets whatever the variable, so `share_tocs` (default on) hands a rank's
  parsed TOCs to the next variable, keyed by (directory, file suffix), after
  checking per variable and directory that the query-parsed file's TOC is
  identical. `toc_cache: <path>` persists them (~50 kB per output directory;
  merged at the end of each run). Measured, one rank, cold, all bit-identical:
  a variable after the first at an iteration 22.2 vs 33.6 s (−34%), the first
  one with the cache 25.4 vs 35.8 s (−29%); end-to-end 4 ranks x 4 variables:
  75% of TOCs shared in the first run, 100% (0 parsed) in a rerun with the
  cache. Safety net, tested: a corrupted shared TOC is refused by the check,
  and one forced past it makes the slice be redone with the variable's own
  TOCs (KeyError fallback, warning) - output still identical. Every 3D config
  here points at `$A/toc_cache_HHR_newcool_30M_flux_3d.pkl`; the first run to
  use it was the single-snapshot xz cut of 2026-10-06 (§8), after which the
  next cut of that snapshot parsed 0 TOCs and took 11 s instead of 75 s.
- **2D cross-check of the xy run:** 99/100 iterations match `../2d_analysis/
  xy_plane` to ≤5e-13 for all 22 plain variables. **it = 81920** (first output
  after the Vista restart, 3240 components instead of 4608) differs for
  evolved fields by up to 4e-5 of the max at 617 points on the level-6 edge;
  metric agrees to 1e-13; raw Carpet data agree bit-for-bit. The 2D path gives
  level-6 interpolation there, the 3D cut level 5. **Resolved (2026-10-05):**
  traced to that 2D output step handling ghost zones differently, not to the
  pipeline. (`cctk_nghostzones` is 3 there in both 2D and 3D files, as
  elsewhere, so the difference does not show in the metadata. 81920 is the only
  frame taken from `Vista-Output/output-0007-Vista`: 45 files, 360 components
  per level instead of 64 x 512.)
- **B^i is Gaussian** (b = B/√4π): with the full metric, `smallb2/(B²/W² +
  (v·B)²)` = 1/4π to 1e-4 (frames 10/50/90). So Ampère's law is
  J = ε D(αB)/(4πα) exactly as in Gaussian units.
- **Reflection symmetry about z = 0:** ∂_z γ_ij and ∂_z α are ≤ 4e-15 on the xy
  plane (z-aligned spins, planar orbit).
- **|J|/|B| is not independent of the congruence:** 4πJ/|B| = (ℓ·curl ℓ)ℓ +
  ℓ × (κ − D_⊥ln(α|B|)), so J_∥/|B| = ±Ω_B/2π and |J|/|B| =
  [4Ω_B² + |κ_B − D_⊥ln(α|B|)|²]^½/4π (verified to 1e-13 in
  `test_current_density.py`). In the BH1 box the field-aligned part carries a
  median 61% of J². The ∂α term matters only next to the hole (median 0.05%,
  up to 2x).
- Σ_B/κ_B and (Σ²−Ω²)/(Σ²+Ω²) are pixel-noisy (comparable grid-scale
  quantities); `ratios_current` smooths nothing. In the BH1 box shear beats
  twist on 60% of pixels and curvature on 78%.
- **Smoothing the derivatives** (`ratios_current_smoothed`): a Gaussian of
  σ = 0.2 M cuts neighbouring-pixel sign flips of (Σ²−Ω²)/(Σ²+Ω²) from 32% to
  17%; wider barely helps (14% at 0.5 M) but keeps eroding the current
  (median |J|/|B| 0.28 → 0.18 → 0.11). Pixel values change by ~50%: the
  unsmoothed ones are grid-scale-dominated. Shear still beats twist on 58%,
  curvature on 73%. The ~1% residual dotted lines come partly from field
  reversals (low |B|) and the unsmoothed B in ℓ = B/|B|.
- **Whole plane vs the BH1 box** (`*_box40`): Θ_B, Σ_B, Ω_B, κ_B and |J|/|B|
  are ~3x smaller over [-40,40]^2 (95th percentiles |Θ| 1.06, Σ 2.7, Ω 2.6,
  κ 1.65; |J|/|B| median 0.056 vs 0.16), so the BH1 colour limits leave half
  the plane in the bottom 3% of the scale - the box40 movies use 2.5 / ±1 and
  [1e-3, 1]. The ratios are distributed as around BH1 (shear beats twist on
  60%, curvature on 74%); smoothing changes them by a median ~20% there (~50%
  in the BH1 box). A full-plane frame takes ~6 s and ~1.3 GB to compute,
  30-50 s to render on one rank.
- `compare_output.py` reports an **elementwise** relative difference, which
  blows up at zero crossings (gxy, B). Normalise by the field's global maximum
  instead before calling something a mismatch.
- `lcool` is NaN everywhere in the xy output, 2D and 3D alike.
- **YAML:** `1.0e2` parses as a string (YAML 1.1 wants `1.0e+2`). The movie
  scripts now cast limits with float(); older BH1 configs got away with it.
- Scratch test jobs must run from `/lagoon` — the session scratchpad is under
  node-local `/tmp` (§1). sympy is not in `etk-analysis` (base Anaconda has it);
  the congruence test is numerical and needs none.

### The xz counterpart: `derivatives_xz` (2026-10-07)

The linear congruence movie and the ratios/current movies (unsmoothed and
smoothed) on the plane **through BH1, normal −y, following it**:
y = y_BH1(t), axes (x − x_BH1, z), ±10 M, all 100 3D outputs.

```
$A/derivatives_xz/
├── Resample/
│   ├── config_30M_HR_q1_flux_3d_xz_BH1.yaml   trajectory plane, origin bh1, normal
│   │                                      [0,-1,0], u_axis x; +-20 M, 401x401
│   │                                      (0.1 M); 21 vars + d_k of alp, B^i,
│   │                                      gamma_ij (30); resample_chunks: 100
│   ├── job.batch                          gp-default, 4 nodes x 25, 12.4 min (job 7563)
│   └── HR_newcool_30M_flux_resampled_3d_xz_BH1/   51 files, 100 its, 4.6 GB
├── theta_sigma_omega_kappa/Frames/        the xy movie's make_frames.py with the
│   │                                      physics unchanged; box about bh1_uv
│   │                                      (no trajectory table), lines (u.B, v.B)
│   ├── movie_config.yaml                  limits re-measured: [0, 2.5], +-1,
│   │                                      sigma [1e-4, 1e3]
│   ├── job.batch                          gp-default, 1 node x 32 ranks (job 7565)
│   ├── test_congruence_scalars.py         copied; ALL OK
│   ├── check_intersection.py (+ .out)     vs the xy movie's data on the shared
│   │                                      line z = 0 through BH1; ALL OK
│   └── frames__HR_newcool_30M_flux_3d_xz_BH1/   100 PNGs + ..._3d_xz_BH1.mp4
├── ratios_current/Frames/                 the xy ratios movie, same geometry edits;
│   │                                      |J|/|B| [3e-3, 3], rho_0 [1e-9, 1e-2],
│   │                                      sigma [1e-4, 1e3], S/k [0.1, 10] kept
│   ├── check_intersection.py (+ .out)     alpha, d_k alpha (d_y alpha normal to
│   │                                      the cut), J^i/|B|, the ratios; ALL OK
│   ├── test_congruence_scalars.py, test_current_density.py   ALL OK
│   └── job.batch                          gp-default, 32 ranks (job 7568)
├── ratios_current_smoothed/Frames/        + derivatives smoothed, sigma 0.2 M in
│                                          (x, z); test_smoothing.py ALL OK (job 7569)
└── Bz_Bphi/Frames/                        2x2: B^z/|B|, B^phi/|B| in the xy plane
                                           (derivatives_xy data) and in the cut,
                                           lines on all four; phi about BH1's spin
                                           axis (vertical through BH1); new
                                           make_frames.py, test_components.py
                                           ALL OK (job 7583)
```

- **The whole 100-iteration resample took 12.4 min** (xy run: 48 min, before
  TOC sharing and on a ±40 M plane): 95% of TOCs shared, and derivative slices
  ~2 s once a rank has opened an iteration's files (the first variable ~140 s
  cold).
- **Frame stride 11 aliases with the orbit** (11 x 51.2 M = 563 M ≈ half the
  ~1077 M period): frames 0, 11, ..., 99 all have BH1 near y = 0. Irrelevant
  for the xy box, but a plane with a fixed normal turns relative to the binary,
  so for it sample every frame or a stride of 5.
- **Comparing two planes on their shared line separates the derivative
  conventions sharply only for smooth fields.** The metric's derivatives,
  ∂_y γ_ij included (normal to the cut, in the xy plane), beat a mirrored cut
  and a cut with ∂/∂(u, v, n) in place of ∂/∂(x, y, z) by 3x10³–2x10⁴. The B
  derivatives and the scalars, which vary on the grid scale, manage only
  2–10x - a 10x bar fails for them although their medians are 3.4–5.5x below
  the row-to-row variation. Components that vanish on z = 0 by symmetry
  (d_z γ_xx, d_x γ_zz, ...) need an absolute round-off test, not one relative
  to the row variation.
- **The cut is 4–10x quieter than the xy BH1 box** (medians Σ 0.10, Ω 0.22,
  κ 0.16, |Θ| 0.12 /M; 95th percentiles 2.4 / 2.4 / 1.1 / 0.75): it runs
  through the funnel and corona, where the field is ordered. σ > 10 on 48% of
  the box (max 889). Hence the `*_box40` limits, and the snapshot σ range.
- **The cut is twist- and curvature-dominated** (the funnel): shear beats
  twist on 35% of pixels and curvature on 44% (xy box 60% / 78%); |J|/|B| ~6x
  weaker (median 0.042 /M); ρ₀ down to ~2e-9 in the funnel. Smoothing at
  0.2 M halves the sign flips of (Σ²−Ω²)/(Σ²+Ω²) here too (18% → 8%; xy
  32% → 16%). On the spin axis κ_B dips (straight field line), so Σ_B/κ_B
  shows a thin line at u = 0.
- **Funnel winding** (`Bz_Bphi`): in the cut B^φ < 0 above the equator and
  > 0 below at every frame (median 96% of funnel pixels, |x − x_BH1| < 3 M,
  2 < |z| < 10 M), B^z/|B| median 0.80 - the trailing toroidal field of an
  upward field on a hole spinning about +z (BH1's spin is along +z, a1x = a1y
  = 0). B^φ is NaN on the cut's axis column (grey line).
- **A norm is a weak yardstick for the derivative convention:** |J|/|B| beat
  the wrong-convention cut by only 1.9x (mostly twist here, |J_∥| = Ω_B/2π),
  the signed J^i/|B| by 3.1–4.6x, ∂_y α by 1.6x10⁴. Test signed components.

### `derivatives_xz/ratios_current`: funnel mask on ρ₀ (2026-10-19)

Re-rendered in place many times the same day (user); the current movie draws
the boundary of a **funnel mask** on the ρ₀ panel, cell by cell:

    mask = 0 where (Σ_B² − Ω_B²)/(Σ_B² + Ω_B²) ≥ 0 OR σ ≤ 1 OR B^z ≤ 0,
    mask = 1 where none holds (ratio < 0, σ > 1, B^z > 0: the funnel),

from the panels' own arrays (ratio unsmoothed), drawn as the mask's 0.5
contour (white, 0.6 pt + 0.35 pt black edge, legend with the definition).
`mask_criterion()` / `funnel_mask()` in `make_frames.py`; `MASK_*` (0 / 1 /
0, `MASK_NEIGHBOURS` 0) and `CONTOUR_*` in `movie_config.yaml`;
`CONTOUR_PANELS: {}` gives the 2026-10-07 movie back. Mask = 1 on 42 / 47 /
60% of the cropped box at t = 0 / 2560 / 5068.8; the line follows the funnel
walls. The last render ran on `debug` (24 ranks, 2.8 min): gp-default was
full and the user said "run on debug".

Tried and reverted the same day, each one config line (mask = 1 at the
three times): `MASK_SQUEEZE_MIN: 0.1` (43 / 48 / 61%); `MASK_NEIGHBOURS: 4`
- 0 only if the cell and its 4 face neighbours meet the criterion, an
`ndimage.binary_erosion` with `border_value=1`, the funnel grown by one cell,
~40% fewer contour vertices (49 / 52 / 65%); the latter with
`MASK_SQUEEZE_MIN: -0.25` (44 / 48 / 62%). The mask was first asked with
"and", corrected to "either" before the render. `test_contours.py` →
`tests.out` ALL OK for every variant (criterion on hand-made cells incl.
thresholds and NaN; neighbour option on small grids incl. a corner; mask =
independent recomputation, shifted copies when neighbours are on; vertices
on 0.5 to 9e-15, transposed array off by 0.5; style).

History: first contours of the ratio itself at −0.9 / −0.75 / −0.5 / 0, then
−0.1 alone (`{rho_b: squeeze}` + those levels). Those mesh the disk and
sheath, where the ratio flips sign cell to cell - the mask boundary does
not, because σ and B^z are smooth there and decide via "or". At 1.1 pt the
four-level mesh hid ρ₀; 0.6 pt since.

### Two-plane movies about BH1 (2026-10-17)

Both rows of each, xy (`derivatives_xy`) over the xz cut (`derivatives_xz`),
the `Bz_Bphi` layout and machinery (centre = the cut's `bh1_position`), 2 x 3;
no new resample. Field lines (`STREAM_PANELS`, panel key -> colour) only on
the b^2/2P panels of the first, none in the second (user, 2026-10-17; the
first renders, jobs 7709/7710, had them on every panel). Each Frames/ has make_frames.py,
movie_config.yaml, job.batch (gp-default, 32 ranks, ~5 / ~1 min), README.md,
tests + `tests.out` (ALL OK), and the PNGs + mp4.

```
$A/bernoulli_sigma_invbeta_BH1/Frames/     Be = -h_tot u_t - 1 (RdBu_r, linear +-0.1),
                                           sigma (log [0.01, 10]),
                                           b^2/2P (log [0.01, 100]); job 7713
$A/shear_current_rho_BH1/Frames/           Sigma_B, |J|/|B| (linear [0, 5] /M),
                                           rho_0 (jet, [1e-10, 1e-1]); job 7712
```

- **Be uses h_tot = 1 + ε + P/ρ₀ + b²/ρ₀** (user's choice after review; their
  first draft had b²/(2ρ₀), which drops the field's energy density).
  `BE_B2_FACTOR` switches it. u_t = W(−α + β_i v^i).
- **`vel_i` is the Valencia velocity**: W from it matches `w_lorentz` to a
  median 6e-6 (xy) / 1e-4 (cut), against 3e-2 / 3e-1 as u^i/u^0. Up to ~10%
  apart where W ≈ 2 in the funnel (separate interpolation of W and v^i).
- `test_planes.py` (same file in both) checks every panel's quantity on the
  shared line z = 0 through BH1: 0.11–0.18 of the neighbouring-row change.
- Lines on a jet ρ₀ panel should be white: cyan vanishes at ρ₀ ~ 1e-7.
- `srun python make_frames.py` fails here (mpi4py, no PMI): test frames go
  through `sbatch` + `mpirun -n 1`, writing under the Frames dir (compute
  nodes cannot see the login node's /tmp).

---

## 8. Single-snapshot cuts through BH1 (t = 3020.8 M, 2026-10-06)

One 3D output, iteration 241664 (frame 59 of `derivatives_xy`), cut along
planes through BH1 and plotted in boxes centred on it. The recipe generalises
to any snapshot and body.

```
$A/single_snapshot_BH1_3020/                         README.md has the details
├── Resample/
│   ├── config_30M_HR_q1_flux_3d_xz_BH1_3020.yaml    xz through BH1: normal -y,
│   │                                                u_axis x; +-20 M, 400x400;
│   │                                                21 standard vars (no lcool)
│   ├── job.batch                                    debug, 21 ranks, 75 s cold
│   ├── config_30M_HR_q1_flux_3d_xzrot_m45_BH1_3020.yaml   the same cut turned
│   │                                                -45 deg about z: normal
│   │                                                [-1,-1,0], u_axis [1,-1,0]
│   ├── job_xzrot_m45.batch                          11 s (warm toc_cache)
│   ├── config_..._xzrot_m45_BH1_3020_derivs.yaml    + d_k alp, B^i, gamma_ij
│   │                                                (30 files, same label/dir), 11 s
│   └── HR_newcool_30M_flux_resampled_3d_{xz,xzrot_m45}_BH1_3020/
└── Plot/
    ├── plot_snapshot.py          rho_0 | sigma + lines | B^z/|B|, row 1 xy (frame
    │                             59 of derivatives_xy, cropped about BH1), row 2
    │                             the cut (CUT_DIR); field lines (u.B, v.B) from
    │                             the file's plane axes
    ├── plot_config_7M.yaml       +-7 M, lines 1.2x thicker (0.4554), density 6.0
    ├── plot_config_xzrot_m45_7M.yaml   ... with the rotated cut
    ├── check_intersection.py     the cut vs xy on their shared line -> ALL OK
    └── ratios/                   the ratios figure ((S^2-O^2)/(S^2+O^2), S_B/k_B,
                                  |J|/|B|, rho_0, sigma + lines, B^z/|B|) entirely
                                  in the rotated cut, +-7 M, smoothed + unsmoothed:
                                  make_frames.py of ratios_current_smoothed_box40
                                  run on one frame (+ OUT_NAME/OUT_EXTS, plane
                                  axes); test_cut_derivatives.py
```

### Cutting a plane through a body at one time

```yaml
plane:
  kind: trajectory          # positions from the run's own trajectory table
  file: auto
  time_offset: auto
  origin: bh1
  normal: [0.0, -1.0, 0.0]  # xz; for xz turned by -45 deg about z: [-1, -1, 0]
  u_axis: x                 #                                     and [1, -1, 0]
iteration_min: 241664       # = iteration_max: one iteration
iteration_max: 241664
```

The output stores `bh1_position`, `plane_u_axis`/`plane_v_axis` per iteration,
so the plot script places the xy crop and projects B without any manual AST_t0
bookkeeping.

- **A trajectory plane's v axis is n × u**, and the normal's sign is kept as
  given: `normal: y`, `u_axis: x` puts -z up. Pick the normal so that n × u
  is +z (`[0, -1, 0]` for xz). `inspect_plane.py ... --times 3020.8 3020.8 1`
  shows the frame in seconds, before any data is read. Static planes do not
  have this problem: they keep the 2D (x, z) convention.
- **Derivatives in a tilted cut** are still w.r.t. the simulation x, y, z (the
  invariants do not care). Checked on the rotated cut: (∂_x f − ∂_y f)/√2 and
  ∂_z f match centred differences of f along u and z to ~1e-5 of the max
  (metric, lapse); a wrong convention is off by ~1e-2.
- **Two planes on different grids never agree exactly on their shared line.**
  Judge them against how much the field changes between neighbouring grid rows:
  both cuts agree with the xy plane to within that on 92–97% of the line (99.5%
  for γ_xx), worst within 1 M of the puncture; the mirrored cut misses by
  13–16,000x more (`check_intersection.py`).
- **Streamplot `density` is relative to the data's extent, not the axes'.**
  Crop every panel's data to the box before drawing lines, or a panel fed a
  larger array (the ±20 M cut in a ±7 M box) gets sparser lines than its
  neighbours.
- **`srun` vs mpi4py**: `plot_snapshot.py` runs under `srun`, `ratios/
  make_frames.py` does not (§1, fourth trap).

### What the snapshot shows

- **σ in the meridional cuts reaches ~900** in the polar funnel (95th
  percentile 166), so the movies' σ scale [1e-4, 10] saturates a third of the
  panel; all snapshot figures use [1e-4, 1e3]. ρ₀ uses the xy_plane /
  corotating_meridional range [1e-10, 5e-2].
- **Twist and curvature dominate the rotated cut**, unlike the equatorial
  plane: the funnel (|u| ≲ 2) is (Σ²−Ω²)/(Σ²+Ω²) ≈ −1 with Σ_B/κ_B < 1, its
  sheath shear-dominated, the disk mixed. Smoothed, shear beats twist on 40%
  of the ±7 M box and curvature on 39% - against 58% and 73% in the smoothed
  BH1-box movie (unsmoothed: 41% / 47% here, 60% / 78% there). Smoothing
  (σ = 0.2 M) removes the speckle, not the pattern.
- At ±7 M the 0.1 M sampling (140 samples across) shows as blocks. The AMR
  near the hole is probably finer than that (not checked) - see §9 item 5.

### Rerunning

```bash
S=$A/single_snapshot_BH1_3020
cd $S/Resample && sbatch job.batch                   # xz, ~75 s cold
sbatch job_xzrot_m45.batch && sbatch job_xzrot_m45_derivs.batch   # rotated, ~11 s each
cd $S/Plot && python plot_snapshot.py                # +-20 M, xz   (~40 s, 1 core;
python plot_snapshot.py plot_config_7M.yaml          #  srun -p debug is fine)
python plot_snapshot.py plot_config_xzrot_m45_7M.yaml
python check_intersection.py && python check_intersection.py plot_config_xzrot_m45_7M.yaml
cd ratios && sbatch --wait -p debug -n 1 --mem=8G --wrap "source ~/anaconda3/etc/profile.d/conda.sh && \
  conda activate etk-analysis && python make_frames.py ratios_config.yaml && \
  python make_frames.py ratios_config_unsmoothed.yaml && python test_cut_derivatives.py"
```

### The BH2 counterpart: `single_snapshot_BH2_4403` (t = 4403.2 M, 2026-10-07)

The same figures around BH2 at iteration 352256 (frame 86): the ±20 M and ±7 M
snapshot figures and the ratios figure (smoothed and unsmoothed). The cut is
the **coordinate plane y = 0**, not a plane through the hole and not a rotated
cut. At this time the binary lies almost along x, with BH2 at
(14.7806, −0.2183, 0), so y = 0 passes 0.218 M from both holes. README.md has
the details.

- **A fixed plane that still records the bodies:** use `kind: trajectory` with
  `origin: [0, 0, 0]`, `normal: [0, -1, 0]`, `u_axis: x`. The frame is the
  static xz plane, but the files carry `bh*_position` and `bh*_uv`. The box
  centre is then `bh2_uv`, not (0, 0). `plot_snapshot.py` there has a `BODY`
  setting and crops the cut about `<BODY>_uv`, which is (0, 0) for a cut
  through the hole, so it also works for BH1. `check_intersection.py` takes
  the shared line from the cut's `plane_origin`.
- **Use the 2D grid when the plane is a 2D-output plane.** The cut is
  ±40 M at 800 × 800, the grid of `2d_analysis/xz_plane`, so it can be checked
  against the simulation's own 2D output (`check_2d_xz.py`): all 15 shared
  variables agree to ≤1.3e-13 of their maximum. Both boxes about BH2 fit
  inside it.
- One gp-default node (it was idle) runs both resample configs in one job: 81 s
  for the plain variables cold, then 12 s for the 30 derivatives with a warm
  `toc_cache`.
- The scratchpad `/tmp` is node-local, so an `sbatch -o` or script path there
  fails on a compute node. Stage throwaway files under `/lagoon`.
- Results: the BH1 colour scales fit unchanged (σ max 889). The funnel is
  twist- and curvature-dominated, as around BH1. Smoothed, shear beats twist
  on 42% of the ±7 M box and curvature on 46%.

---

## 10. Planes about each hole of a precessing binary (3C 186 LR, 2026-10-08)

`/lagoon/michailchabanov/Analysis/BBH_production/K-configs-LR/3C_186_LR/3d_analysis/rho_BH_planes/`
(README.md has the details): `rho_b` in the xz plane through each hole, the
plane through each hole **perpendicular to its spin** (turning as the spin
precesses), and the xy plane through each hole (z = z_BH(t)), all ±10 M; the
meridional plane through both holes and z
(v = z, the holes moving within it) and the orbital plane (normal L̂, inertial
in-plane axes), both ±40 M; a 2 x 2 movie plus one movie per plane, one colour
scale.
Trajectory `flip_20M_3C186_prod.h5` - spins nearly in the orbital plane
(tilted up to ~117° from z), precessing, orbit not planar.

- **A trajectory table can merge inside the 3D output.** Here the holes
  coincide at t = 17456 M while 3D output runs to 18074 M; the pipeline refuses
  a trajectory plane once the separation is 0 (even with `normal:` given), so
  set `iteration_max` to the last output before it. In the last outputs
  before that the table's spins jump to the remnant's (tilt 76° → 13°).
- **The spin plane's in-plane axes come from a plane module**: `normal: spin1`
  with `u_axis: module` and `u(t) = ẑ × ŝ(t)`, so v = n × u is the projected
  +z and the plane is seen from the spin's north pole. The module imports
  `plane_motion.Trajectory` (on `sys.path` when run by `resample_3d.py`) so
  that u is interpolated exactly as the normal. A fixed `u_axis` axis cannot
  do this: the normal precesses all round z.
- **Choosing a moving plane's in-plane axes with a module.** The same trick
  sets any frame the presets lack: the meridional plane (`second_vector:
  meridional`) with `origin: module` = the COM at z = 0 and `u_axis: module` =
  the horizontal part of d gives v = z exactly, so the holes move within the
  plane instead of being pinned to v = 0 by `u_axis: separation`. For a plane
  that must *not* co-rotate, `normal: orbital` with `u_axis: x` keeps u as
  close to x as the precessing plane allows and the binary orbits in it.
- **Pipeline flaw: the normal's sign-continuity fix mirrors undersampled
  planes.** `_make_normal_continuous` flips n whenever consecutive normals are
  > 90° apart. That is right for a slowly turning plane, wrong for one turned
  by more than 90° between outputs: a plane holding the separation does that
  once the orbital period drops below ~4x the 3D cadence (here 12 of 340
  steps, up to 159°), and v = n × u then comes out as −z on stretches of
  frames (7 here). The data are right, the orientation mixed; the movie's
  `make_frames.py` turns those frames over (v → −v). Not fixed in
  `plane_motion.py` - see §9.
- **Cost on this LR run** (64 files per variable per iteration, like the HHR
  run): one variable x 341 iterations, 801² samples, on 128 ranks: 6.6 min with
  cold TOCs, then 46-65 s per further plane with every TOC in `toc_cache`; the
  meridional plane on 32 `debug` ranks, 3.4 min.

---

## 11. Spheres from the 3D output (2026-10-17/18)

Code: commit `e7f526a` on `integrals` (pushed 2026-10-17), after `ba0b3da`
(the §7 two-plane movie notes). Movie and run directories are on /lagoon,
outside the repo.

`Resample/resample_sphere.py` (`Resample/README.md`, "Sampling spheres"): a
uniform (θ, φ) grid - θ cell-centred, φ periodic - at one or several radii,
each point from the finest level that covers it. A thin subclass of the 3D
plane backend: `_add_component` now walks `spec.windows(...)` (index tuples
into the output) instead of the plane's single (u, v) rectangle, so a
`SphereSpec` (`sphere_geom.py`: exact shell test + bounding-ball cap, φ
windows wrapping at 0) slots in. Centre / pole / φ = 0 are a plane's origin /
normal / u axis, so `plane_motion` builds the moving frame (`centre: bh1`,
`pole: spin1`, `phi_zero: separation`, `module` ...). Several radii: one pass,
one file, `data[it, r, θ, φ]` chunked per (it, r) sheet; one radius:
`data[it, θ, φ]`. `refinement_level: yes` (default for spheres) writes the
supplying level per point from metadata alone; planes have it too (default
no). `sphere_tools.py` reads it back (always 4D), gives r̂/θ̂/φ̂, exact cell
areas, √γ.

Verified (scratch tests, it = 204800 / 208896, cross output-0018/0019):
plane xy incl. `dBx_dx`, the moving xz plane about BH1 incl. its frame
datasets, and 2D xy all **bit-identical** to production after the refactor
(chunked + merged); sphere values = trilinear interpolation by hand from the
raw `*.xyz.file_*.h5` (finest owning component) to ≤ 3e-15 of the rms; r = 3
sheet of a [2.5, 3, 4] run = the 1-radius run to ≤ 3e-14 of the rms (not
bit-identical: the union hyperslab moves the interpolation's block origin);
level 8 everywhere for r ≤ 4 about BH1; net/unsigned flux ≤ 3e-5.
`python Resample/test_sphere_geom.py` → ALL OK (7200 random boxes, no grid
point missed).

- **Compare sign-changing fields against their rms**, not pointwise: B^i
  and its derivatives cross zero, where a round-off difference of 1e-16 of the
  rms is a "relative" 1e-11.
- **Rank counts that divide `resample_chunks`** keep a rank on the same
  iterations of every variable (geometry cache, shared TOCs); 50 ranks for
  100 chunks gives each rank two iterations of all variables.

$A/sphere_BH1/ (production, HHR flux run): r = 3 M about BH1, pole +z,
φ = 0 along +x, 300 x 600 (one sample per level-8 cell), the 22
`derivatives_xy` variables + `refinement_level`, 100 iterations; README.md,
`Resample/test_sphere.py` → `tests.out` (ALL OK: raw-file interpolation at
four times ≤ 3e-15 of the rms, level 8 everywhere, net/unsigned flux ≤ 8e-5).

- **Cost:** job 7717, 2 gp nodes x 25 ranks, 2300 slices in 439 s; per slice
  median 7 s (1.3-62 s, the cold first slice per rank and iteration being the
  slow one), every TOC from `toc_cache`. A sphere slice touches ~61 of 4608
  components (~50 MiB; three radii 2.5-4 M: ~90, ~145 MiB) against ~420 and
  ~85 MiB for the 800² xy plane; cold slices cost the same ~30 s either way
  (metadata), so several radii per run are nearly free.

$A/sphere_BH1/mollweide/Frames/ (movie, 2026-10-17): ρ₀, σ, b²/2P over
B^z/|B|, B^φ/|B|, B^r/|B| on the whole sphere, Mollweide (longitude φ from +x
at the centre, latitude 90° − θ, seen from outside; white dot = direction to
BH2), the last plane movies' colour scales; job 7719 and a re-render after the
seam fix below, one gp node, < 1 min each. Tests (ALL OK): projection on synthetic grids, the quantities against
`Bz_Bphi`'s formulas, and the equator against the xy plane (correlation 0.99
vs ~0 mirrored/turned).

$A/sphere_BH1_r7/ (2026-10-18): the same at r = 7 M - 700 x 1400 (still one
sample per level-8 cell, and level 8 everywhere: BH1's finest box has
half-width 9 M), the 22 variables + `refinement_level`; job 7724, 1081 s on
2 gp nodes x 25 ranks, 14 GB; ~230 components / ~400 MiB per slice, median
20 s - at this radius the reads, not the metadata, set the cost. Tests ALL OK
(raw-file interpolation ≤ 1.5e-15 of the rms, net/unsigned flux ≤ 3.7e-5).
Movie `mollweide/Frames/`: the r = 3 script with the radius read from the
files, < 2 min; equator vs xy plane correlation 0.996 / 0.994 (log ρ₀ / B^z).
A panel is ~800 px for 1400 longitude cells, so the PNGs show about every
second cell; average 2 x 2 before plotting if that ever looks grainy.
`mollweide_rescaled/Frames/` (user, 2026-10-18): the same movie with σ on
log [0.1, 10] and b²/2P on log [1, 100] - only `SIGMA_VMIN` / `IBETA_VMIN`
and the output names differ from `mollweide/`; tests ALL OK.

- **Mollweide in matplotlib:** `pcolormesh` on cell *edges* spanning exactly
  [−π, π] x [−π/2, π/2]; the sphere cell on the ±180° seam is split between
  both map edges (its column appears twice) so nothing is drawn outside the
  ellipse. Compare the seam and the poles with a tolerance: 300·(π/300) is
  not π, and a seam on a cell boundary otherwise yields a zero-width cell.

---

## 12. Shell integrals split by field-line connectivity (2026-10-20)

**Where we left off (2026-10-20 ~12:50 EDT): production job 7793 running,
not tracked (user logged off).** The code is committed on branch `integrals`
(`9865c5f`, these notes in the commit after it), not pushed. Next session:
check the run (below), then look at the results with the user.

**Goal (user, 2026-10-20):** what the `integrals` branch is for - volume
integrals over spherical shells, in production to track flux tubes through
them, **with the holes' magnetospheres separated out**: every point of the
integration domain (not of the whole box) is classified by whether its field
line connects to a horizon. Decisions, all the user's:
- shell **r = 4-5 M about BH1**, spherical (pole z = BH1's spin, centre
  `bh1_position` from the trajectory, translated, not boosted);
- the three integrals of `~/in_plane_flux_tube_diagnostics.md`, with 𝓑ⁱ =
  √γ Bⁱ on flat unit vectors: (a) Φ_φ = (1/2π)∫|𝓑·φ̂|/**R** d³x - user wrote
  "/r", chose the cylindrical R (the half-plane flux) when asked;
  (b) Φ_r = ∫|𝓑·r̂| d³x / (r_out − r_in), spherical r̂ (flux through
  spheres); (c) Φ_z = ∫|𝓑·ẑ| d³x / 2h with **h = r_out = 5 M** (all planes
  crossing the shell; a ζ-binned profile is stored so a smaller h can be
  applied later);
- raw AMR data, no resampling; Python folder `Integrals/` (not the C++
  `CactusVolumeIntegrals_*` tool in the run tree - **no validation against
  it**: "self-consistent tests/checks are sufficient");
- the tracer as a **standalone** package (`FieldLines/`), "for now only the
  parts we need for the integral work" (no footpoint maps, voxel masks, CLI);
- **record both horizon sizes** (analytic Kerr-Schild spheroid 1.0x and 1.1x,
  ≈ AHFinderDirect's coordinate size) and **keep per-point classes for every
  iteration** (both "yes", 2026-10-20).

### Code (see each folder's README)

- `Resample/amr_blocks.py` - `Snapshot`: the component hierarchy of one
  iteration (geometry once, from one variable) and raw slab reads of any
  variable, TOC sharing; `Component.owned_box()` = interior grown by half a
  cell (the backends' ownership rule).
- `FieldLines/` - `amr_field.AMRVectorField` (B on the AMR blocks meeting a
  box, packed for numba, lookup grid → finest owning block),
  `horizons.Horizons.kerr_schild` (spheroid ρ²/(r₊²+a²) + ζ²/r₊² < 1; m = 0.5,
  χ = 0.9: 0.718 / 0.847 M), `tracer.trace` (RK4, step 0.5 x local dx,
  trilinear, both ways; ends `BH1`/`BH2`/`exit`/`cap`/`null`; class = the
  unordered pair; `extra_scales` gives the ends with enlarged horizons from
  the same lines, exactly). `test_tracer.py` ALL OK.
- `Integrals/shell_cells.py` - shell points with exact AMR weights (level L
  minus what level L+1 owns; cut cells by 8³ sub-points jointly with the finer
  boxes), integrands, per-class sums. `test_shell_cells.py` ALL OK.
- `Integrals/shell_integrals.py` - MPI driver (iterations round-robin, numba
  threads per rank); `check_tracer.py` - tracer variants on a seed subset;
  `integrals_tools.py` - `load_integral`, `group(res, "horizon" | "no_horizon"
  | "free" | "undetermined" | "BH1" | "total", horizon_scale=1.1)`,
  `load_points`; `test_driver.py` (synthetic end-to-end: classes, 1.2x
  horizon, restart reads ends, settings change retraces, layout) ALL OK.

### Output (`$A/shell_BH1_r4-5/HHR_newcool_30M_flux_shell_BH1_r4-5/`)

- `Phi_phi|Phi_r|Phi_z|volume__<label>.h5`: `data` (n_iter, n_hscale = 2,
  n_class = 15), `horizon_scales` [1.0, 1.1], `classes` attribute
  (`BH1-exit`, `exit-exit`, ...); `Phi_z` also `profile` (n_iter, 2, 15, 40),
  `z_edges` (ζ bins of 0.25 M). **Written only at the end, by rank 0.**
- `tracing__<label>.h5`: points/mean length per class, volume check, levels,
  timings, horizons, `traced` (1 traced, 0 ends read).
- `points__<label>__it<it>.h5`, one per iteration, written as each finishes
  (tmp + rename, so existing = complete): positions, weights, levels, `ends`
  (n, 2), `xends` (n, 2, 1) - **~4 MB** (ends compress to 0.1 MB); for
  `save_points: [204800]` also lengths, end points, integrands (~400 MB).
- **Restart / re-integration:** a rerun reads the ends of every iteration
  whose `points__` file has the same settings (JSON in its `settings`
  attribute) and the identical points, so it recomputes the sums in ~1 min
  per iteration instead of tracing (~10 min on 32 cores). If 7793 died:
  `cd $A/shell_BH1_r4-5 && sbatch -p gp-default job.batch` - only the
  missing iterations are traced. New integrands/masks: same mechanism.

### The run

`$A/shell_BH1_r4-5/`: `config_shell_BH1_r4-5.yaml`, `job.batch` (4 gp
nodes x 4 ranks x 8 threads, 12 h limit; log `job.out`/`job.err`). Job 7793
started 12:42 EDT on gp-n[0035-0038]; 100 iterations (t = 0-5068.8);
measured ~5,700 seeds/s per 8-thread rank → ~27 min per iteration, 7 rounds
→ expected done **~16:00 EDT 2026-10-20**. Success:
`grep Done job.out` → `Done: 100 iteration(s)`; `ls HHR_*/points__*.h5 | wc -l`
→ 100; five `*__HHR_newcool_30M_flux_shell_BH1_r4-5.h5` files. Also in the
directory: `check.batch`/`check.out` (the variant check below) and
`test_t2560/` + `config_..._t2560.yaml` + `test_t2560.batch` (the one-iteration
test, written by the code before `xends`/`settings` existed - its 398 MB
`points__` file will not be reused; deletable).

### Measured at t = 2560 M (it 204800; one node, 32 threads)

8,719,294 shell points, all level 8; volume 255.5162 vs exact (+1e-7);
3,432 near-axis cells. Shell + integrands 27-36 s, tracing field (±12 M box,
all levels) 8.3 GB in 31-38 s, tracing 537 s (13,100 seeds/s); peak 10.4 GB.
Classes: `BH1-exit` 4,100,586 points (46.9% of the volume), `exit-exit`
4,615,859, capped 2,849; no `BH1-BH1` loops, nothing reaches BH2; median
half-line 22 M. Integrals (code units): Φ_φ 0.1315 (magnetosphere 60.9%),
Φ_r 1.9267 (64.7%), Φ_z 0.1655 (68.1%). Φ_r lies between the r = 3 and 7 M
sphere fluxes (2 x upper hemisphere 1.65 / 2.07, §9 item 9) - plausible, not a
test.

Tracer variants (`check_tracer.py`, same 2e5 random seeds): step/2 → 0.065%
of seeds change class, integrals equal to 4 digits; length cap 400 M →
0.022%; box ±16 M → 0.19% (exit-exit → exit-cap; magnetosphere share
unchanged); **horizons x1.1 → 1.7%**, magnetosphere 46.9 → 48.6% of the
volume, 64.6 → 66.2% of Φ_r - the one setting that matters, hence both
recorded.

### Findings worth not rediscovering

- **Shell boundary cells:** counting cut cells whole or not by their centre
  left up to 1.5% volume error (tests), first order in dx and not averaging
  out where a sphere runs parallel to grid planes - the poles of a shell
  about a grid axis, or a refinement boundary just inside a sphere (level
  faces at 4.625 vs r = 5: a whole layer of coarse cells cut by both). Fixed
  by sub-sampled fractions taken jointly with the finer level's boxes; with
  m = 8 (multiple of 4: refinement edges fall on quarter cells) the volume
  error is ≤ 1e-4 and the analytic integrals agree to ≤ 2e-4.
- **Near the pole axis** |𝓑·φ̂|/R is singular for any field with a component
  across the axis; one point 1e-7 off the axis made Φ_φ 3,700x too large.
  Cells within 4 dx of the axis average it over 8³ sub-points (exact cell
  integral of the piecewise-constant data); what remains is first order in
  dx, ~1e-3 for a uniform field across the axis.
- A test field B_φ ∝ 1/R (a line current) cannot test the axis treatment:
  constant-per-cell data cannot represent it there. Use a field regular on
  the axis (B_φ ∝ R) and a uniform field across it.

### Not done / offered

- Look at the results with the user: Φ(t) per group, both horizon scales.
- Self-consistency of Φ_r against the sphere resampler (radii 4.05..4.95,
  mean of the unsigned sphere flux) - not run.
- φ- and R-binned profiles (space-time diagrams of the note), the β/ρ tube
  criterion Θ of the note on top of the classes, other integrands: no
  tracing needed - the restart mechanism reads the ends from the `points__`
  files, only the variables are read again (~1 min per iteration).
- Other shells (radii, about BH2 or the binary) and the horizon calibration
  (∇·B) need new traces.

---

## 9. Open items

1. ~~Open a PR `corot` → `main`~~ merged 2026-10-16 (PR #3, `9558ba2`).
2. Dimensionless diagnostics: (Σ²−Ω²)/(Σ²+Ω²) and Σ_B/κ_B are done -
   `ratios_current` (BH1), `ratios_current_smoothed`, the whole-plane
   `ratios_current_smoothed_box40`, the rotated cut at t = 3020.8 (§8), and
   the xz plane about BH2 at t = 4403.2 (§8). Σ_B Δx (resolution) not done.
   BH2 versions of the movies if wanted. The xz-through-BH1 plane
   (`derivatives_xz`, §7) has the congruence movie and both ratios movies.
3. Integrated shear along field lines (eq. 57) — needs field-line tracing;
   the tracer exists now (`FieldLines/`, §12), integrating along a line is
   not built.
4. ~~3D backend speed: stop re-parsing TOCs~~ done 2026-10-05 (`share_tocs`,
   `toc_cache`; −29 to −34% per cold slice). Left: the 64 file opens per
   variable per slice (8–15 s) and the first variable's metadata pass at an
   iteration (geometry, 9–14 s) - the geometry cache could be persisted the
   same way as the TOCs. Fine-to-coarse skipping: tried, no gain (§7 Findings).
5. Single-snapshot cuts at finer sampling (e.g. 0.025-0.05 M over ±7 M, both
   planes, the xy one resampled anew rather than cropped): offered, not done.
   Each cut is a ~1 min `debug` job with the warm `toc_cache`.
6. ~~it = 81920 discrepancy~~ resolved (§7 Findings).
7. ~~scratch_theta_convergence~~ and the empty `out_bench_a/` in the repo:
   deleted 2026-10-05.
8. `plane_motion._make_normal_continuous` should not flip a normal that turned
   by more than 90° because the plane did (§10): e.g. keep the sign of a
   reference direction (n_raw itself for `second_vector` planes) and warn,
   rather than flipping v. Until then, check `plane_v_axis` of any plane
   turning > 90° per output. The same holds for a sphere's pole (§11):
   check `sphere_pole` of any pole turning > 90° per output.
9. **In-plane flux-tube integrals** (`~/in_plane_flux_tube_diagnostics.md`;
   the reason for the `integrals` branch) - read and checked 2026-10-17,
   **not started**; paused for the movies and the sphere resampler. The note:
   detect in-plane tubes by the field component they carry, not by shape;
   inside a tube mask Θ average the unsigned flux through half-planes
   (toroidal tubes), cylinders (radial) and horizontal planes (vertical), one
   number per time each; calibrate Θ with ∇·B = 0 (a drop in horizon flux
   must reappear as vertical flux through the midplane); φ-t and R-t
   diagrams; test the shear lag 1/(qΩ) between radial and toroidal events.
   What was established for the HHR flux run:
   - The integrals are exact with the Cartesian 𝓑ⁱ = √γ Bⁱ projected on the
     flat R̂, φ̂, ẑ (the R of the coordinate change cancels): only √γ is
     needed from the metric.
   - The midplane version needs no resample: `2d_analysis/xy_plane/Resample`
     has 397 frames (every 12.8 M, t = 0-5068.8, ±40 M) with B^i, γ_ij, P,
     ρ₀, b², 1/β. The 3D xy plane has 100 frames, 51.2 M apart.
   - The in-plane field does not vanish at z = 0 (medians of the flat-norm
     ratios about the origin, t = 512 / 2560 / 4608): 3 < R < 10 M:
     |B_R|/|B| 0.56-0.63, |B_φ|/|B| 0.57-0.65, |B^z|/|B| 0.25-0.32;
     10 < R < 40 M: 0.29-0.50, 0.67-0.78, 0.18-0.32. Tubes just off the
     midplane are still missed; a slab would need several 3D planes.
   - Shear lag 1/(qΩ) = R^1.5/1.5 (Keplerian about the total mass): 21 M at
     R = 10, 60 M at 20, 110 M at 30, so the 2D cadence resolves it for
     R ≳ 11 M, the 3D one for R ≳ 25 M; inside the cavity it does not apply.
   - Horizon calibration: not checked whether
     `CactusAnalysis_BH1/BH2/Surface_integrals_BH*.asc` holds the horizon
     flux, or `CactusAnalysis_Cavity30M` an outer-sphere flux. The sphere
     resampler (§11) now gives fluxes directly. Upper-hemisphere √γ B^r flux
     about BH1, code units, at t = 0 / 1280 / 2560 / 3840 / 5068.8:
     r = 3 M 0.946 / 0.559 / 0.825 / 0.659 / 0.514, r = 7 M 0.952 / 0.769 /
     1.035 / 0.893 / 0.566; net/unsigned ≤ 8e-5 and ≤ 4e-5
     (`sphere_BH1*/Resample/test_sphere.py`). Only these five times were
     printed; the full Φ(t) is a few lines with `sphere_tools`.
   - Proposed first step (awaiting the user's go): one script on the 2D xy
     data giving Φ̄_R, Φ̄_φ, Φ̄_z(t), the tube fractions and the φ-t / R-t
     diagrams, with Θ = (β < β_c) ∧ (ρ₀ < ε x the azimuthal median ρ₀) and
     the regions about the holes excluded; the horizon calibration after.
     Also open: whether `2d_analysis/flux_tubes_scalars_xy/` (2026-09-28) is
     the earlier tracking approach the note replaces - not yet opened.
   - **Built 2026-10-20 as shell integrals, §12** - the brainstorm's
     questions settled there: an `Integrals/` folder in Python on the raw
     AMR points, postcactus' file layer through `Resample/amr_blocks.py`,
     per-quantity HDF5 files like `Resample/`, each volume counted once by
     exact weights of the finest owning level. The production geometry is
     spherical shells about a hole with the magnetosphere separated by
     field-line connectivity, not the midplane annulus/slab with a β/ρ mask
     proposed above (that remains possible: Θ can be applied on top of the
     classes). The run tree's `CactusVolumeIntegrals_*/` is Lorenzo
     Ennoggi's C++ volume-integral code (spherical volumes, 0.847 M spheres
     cut about each hole, pointwise masks); the user does not want it as a
     validation reference.
10. Spheres (§11). Done: r = 3 M and r = 7 M about BH1 (two separate runs,
    `sphere_BH1/`, `sphere_BH1_r7/`), each with the Mollweide movie; r = 7 M
    also with the narrower σ [0.1, 10] / b²/2P [1, 100] scales
    (`mollweide_rescaled/`). Offered, not done: r = 3 M with those scales (the
    same two-line config change), a map that turns with the binary (BH2 at
    the centre: `CENTRE_LON` per frame from `bh2_position`), the same spheres
    about BH2, further radii in one run (one pass, `radius: [...]` - cheaper
    than separate runs), and the full flux time series Φ(t) for item 9.
11. Field lines connected to BH1's horizon (user, 2026-10-19). **Superseded
    2026-10-20 by §12**: the user wants the classification at the points of
    the integration domain only, so the tracer works on the raw AMR blocks
    (no resampled ±10 M box at 0.05 M) and classifies the shell's points;
    the 2026-10-19 plan's horizon-seeded footpoint maps and the 3D voxel
    mask were **not built**. Still parked: cleaning the threshold funnel mask
    of §7 (axis-connected component + filled holes; the axis-connected
    pieces hold 91-98% of the mask-1 area at t = 0 / 2560 / 5068.8, the rest
    80-314 specks, 47-226 one-cell holes) - the field-line classification on
    an xz plane's points (`tracer.trace` with plane points as seeds) would
    give the threshold-free funnel instead. The analytic horizons vs
    AHFinderDirect (`output-*/Miscellaneous/BH_diagnostics.ah1.gp`: BH1 at
    t ≈ 2421 has coordinate radius 0.786-0.936 M, mean 0.868, ~9-10% above
    the analytic 0.718 / 0.847 M; why is not checked) is why §12 records the
    classes for 1.1x horizons too.
12. **NEXT: §12** - check job 7793, then the results with the user.
