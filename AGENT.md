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
about each hole of the precessing 3C 186 run.

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

### Four traps in batch scripts

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

## 9. Open items

1. ~~Open a PR `corot` → `main`~~ merged 2026-10-16 (PR #3, `9558ba2`).
2. Dimensionless diagnostics: (Σ²−Ω²)/(Σ²+Ω²) and Σ_B/κ_B are done -
   `ratios_current` (BH1), `ratios_current_smoothed`, the whole-plane
   `ratios_current_smoothed_box40`, the rotated cut at t = 3020.8 (§8), and
   the xz plane about BH2 at t = 4403.2 (§8). Σ_B Δx (resolution) not done.
   BH2 versions of the movies if wanted. The xz-through-BH1 plane
   (`derivatives_xz`, §7) has the congruence movie and both ratios movies.
3. Integrated shear along field lines (eq. 57) — needs field-line tracing.
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
   turning > 90° per output.
