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
├── Resample/job.batch                  16 ranks, `short`, 5:45:00
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
per rank, and the real figure was lower — but the `short` partition's 6 h cap
leaves little room to be wrong in the other direction.

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
