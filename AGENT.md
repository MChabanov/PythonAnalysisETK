# Operating notes

Practical knowledge for *running* this pipeline on a cluster, as opposed to
what the code does — that is in `Resample/README.md` and `Frames/README.md`,
which stay the reference for options, schema and physics. This file holds the
things that were learned the hard way: batch-environment setup, how to tell a
job succeeded where the usual tools are missing, how to verify a moving-plane
run after the fact, and the movie-encoding recipes.

Written after the co-rotating meridional run described in §3, which is the
worked example the rest of the file refers to.

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

So the resample step - hours on 16 ranks - belongs on `gp-default`, and it is
worth the wait. But the **frame/movie steps are minutes**, and queuing those
behind a whole-node allocation is the wrong trade: send them to `debug`.
Measured: the 100-frame `corotating_meridional_Bfield` render started within
20 s and finished in about 30 s on 16 ranks, sharing the node with another
user's `lm-largemem` job.

Two courtesies, since the node is shared: do **not** pass `--exclusive`, and
keep `--ntasks-per-node` modest. `debug` sets `DefMemPerCPU=2048`, so 16 ranks
get 32 GB by default - ample for rendering, where a rank holds only a couple of
800x800 float64 slices at a time, but something to raise explicitly if you ever
run a memory-hungry step there.

### Three traps in batch scripts

- **`$0` is not the script.** SLURM runs a spool copy, so `$0` is
  `/var/spool/slurmd/jobNNNNN/slurm_script` and `dirname "$0"` is meaningless.
  Use `$SLURM_SUBMIT_DIR` — SLURM starts the script in the submission
  directory, so bare `PWD` is right too.
- **`/tmp` is node-local.** `#SBATCH --output=/tmp/...` writes on the compute
  node and is invisible from the login node. Log to a shared filesystem.
- **A queued job holds a snapshot of the script.** SLURM copies it at submit
  time, so editing the file does not change an already-queued job — cancel and
  resubmit.

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

## 7. Field-line congruence diagnostics — where the work stands (2026-10-05)

Goal: maps of the expansion Θ_B, shear Σ_B, twist Ω_B and curvature κ_B of the
magnetic field-line congruence, `~/magnetic_field_line_congruence.tex`,
"Practical evaluation" section. **Equation numbers:** the compiled manuscript's
E74/E78/E79/E80 are eqs. 73/77/78/79 of the tex compiled standalone (one
extra numbered equation precedes the section in the manuscript).

### Directory structure

```
~/PythonAnalysisETK/Resample/              (repo, branch corot)
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
└── ratios_current/Frames/                 BH1 movie (2x3): (S^2-O^2)/(S^2+O^2),
    ├── make_frames.py                     S_B/k_B, |J|/|B| / rho_0, sigma + lines,
    │                                      B^z/|B| (no lines). congruence_scalars
    │                                      copied unchanged + current_density()
    ├── movie_config.yaml                  44 input paths, measured ranges
    ├── job.batch                          debug, 40 ranks, ~2 min (job 7523)
    ├── test_congruence_scalars.py, test_current_density.py   both print ALL OK
    ├── README.md                          definitions, units, identities, ranges
    └── frames__HR_newcool_30M_flux_3d_xy/ 100 PNGs + ratios_current_BH1_...mp4
```

Panels: Θ_B, Σ_B, Ω_B / κ_B, σ = b²/(ρ₀h) + lines, B^z/|B| + lines. The last two
copy `../2d_analysis/xy_sigma_BH1` and `xy_Bz_overnormB_BH1` (lines 1.15x
thicker, lw 0.3795). Box: 20 M following the BH from the trajectory table.
6 fps, because 3D output gives 100 frames against 397 for the 2D movies.

### State

- The derivative code is committed on `corot` as `adb9b7a` and pushed
  (`corot` = `origin/corot`); not merged into `main`.
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
cd $A/derivatives_xy/theta_sigma_omega_kappa/Frames && sbatch job.batch   # ~5 min
cd $A/derivatives_xy/theta_sigma_omega_kappa_BH2/Frames && sbatch job.batch
cd $A/derivatives_xy/Resample && sbatch job_dalp.batch     # ~5 min, 1 gp node
cd $A/derivatives_xy/ratios_current/Frames && sbatch job.batch            # ~2 min
```

Movie knobs live in `movie_config.yaml`: `CONG_SCALE` (log/linear),
`CONG_VMIN/VMAX`, `THETA_LINTHRESH/THETA_VMAX` (log), `CONG_LIN_VMAX`,
`THETA_LIN_VMAX` (linear), `THETA_FORM` (divfree/projector), `BH`,
`STREAM_*`. Typical values in the BH1 box: medians Σ 1.1, Ω 0.9, κ 0.5,
|Θ| 0.25 /M; 95th percentiles 8.6 / 8.4 / 5.3 / 3.6; 99.5th 20–32.

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
  TOCs (KeyError fallback, warning) - output still identical. The
  `derivatives_xy` configs now point at `$A/toc_cache_HHR_newcool_30M_flux_3d.pkl`
  (empty until the next run); other 3D configs can share it.
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
  quantities); nothing is smoothed. In the BH1 box shear beats twist on 60% of
  pixels and curvature on 78%.
- `compare_output.py` reports an **elementwise** relative difference, which
  blows up at zero crossings (gxy, B). Normalise by the field's global maximum
  instead before calling something a mismatch.
- `lcool` is NaN everywhere in the xy output, 2D and 3D alike.
- **YAML:** `1.0e2` parses as a string (YAML 1.1 wants `1.0e+2`). The movie
  scripts now cast limits with float(); older BH1 configs got away with it.
- Scratch test jobs must run from `/lagoon` — the session scratchpad is under
  node-local `/tmp` (§1). sympy is not in `etk-analysis` (base Anaconda has it);
  the congruence test is numerical and needs none.

### Open items

1. Open a PR `corot` → `main` when ready (`corot` is pushed).
2. Dimensionless diagnostics: (Σ²−Ω²)/(Σ²+Ω²) and Σ_B/κ_B are done
   (`ratios_current`, BH1); Σ_B Δx (resolution) not done. BH2 version if wanted.
3. Integrated shear along field lines (eq. 57) — needs field-line tracing.
4. ~~3D backend speed: stop re-parsing TOCs~~ done 2026-10-05 (`share_tocs`,
   `toc_cache`; −29 to −34% per cold slice). Left: the 64 file opens per
   variable per slice (8–15 s) and the first variable's metadata pass at an
   iteration (geometry, 9–14 s) - the geometry cache could be persisted the
   same way as the TOCs. Fine-to-coarse skipping: tried, no gain (Findings).
5. ~~it = 81920 discrepancy~~ resolved (Findings).
6. ~~scratch_theta_convergence~~ and the empty `out_bench_a/` in the repo:
   deleted 2026-10-05.
