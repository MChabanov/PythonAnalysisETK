# Volume integrals

Flux integrals over a spherical shell, computed on the raw AMR grid points of
the 3D output (no resampling) and split by where the field line through each
point ends, so that the holes' magnetospheres can be told apart from the rest
of the field. The integrals are those of `~/in_plane_flux_tube_diagnostics.md`,
for a shell r_in ≤ r < r_out about a hole (here BH1, 4-5 M):

| Output | Definition | Meaning |
| ------ | ---------- | ------- |
| `Phi_phi` | (1/2π) ∫ \|𝓑·φ̂\| / R d³x | mean unsigned flux through the meridional half-planes φ = const |
| `Phi_r`   | (1/(r_out − r_in)) ∫ \|𝓑·r̂\| d³x | mean unsigned flux through the spheres r = const |
| `Phi_z`   | (1/2h) ∫ \|𝓑·ẑ\| d³x, h = r_out | mean unsigned flux through the planes ζ = const crossing the shell |
| `volume`  | ∫ d³x | coordinate volume |

with 𝓑ⁱ = √γ Bⁱ projected on the flat unit vectors of the frame: centre c,
pole ẑ (d = x − c, ζ = d·ẑ, R = distance from the pole axis, φ̂ = ẑ × d / R).
They are exact for coordinate surfaces - 𝓑ⁱ is divergence-free with the
coordinate divergence - so √γ is all the metric needed. The frame is
translated to the hole, not boosted.

| File | Purpose |
| ---- | ------- |
| `shell_integrals.py` | The MPI driver: per iteration, the shell's points and integrands, the tracing field, the line through every point, the sums per class; output files. |
| `shell_cells.py` | The shell's grid points and their exact AMR volume weights, the integrands, the per-class sums. numpy only. |
| `check_tracer.py` | One iteration, a random subset of the shell's points: how the classes and the integrals per group change with the tracer settings (step, length cap, box, horizon size). |
| `integrals_tools.py` | Read the output; sum classes into groups (`total`, `horizon`, `BH1`, `no_horizon`, `free`, `undetermined`), per horizon size; `load_points` for one iteration's per-point classes. |
| `config_example_shell.yaml` | Documented template config. |
| `test_shell_cells.py` | Synthetic hierarchies, fields with closed-form integrals: `python test_shell_cells.py` → ALL OK. |
| `test_driver.py` | The driver end to end on synthetic data: classes, enlarged horizon, restart from the `points__` file, output layout: `python test_driver.py` → ALL OK. |

The tracer is `../FieldLines/`; reading raw components is
`../Resample/amr_blocks.py`; the SimDir scan, the iteration query, the TOC cache
and the hole's frame from the trajectory are the resampling pipeline's own.

## Use

```bash
mpirun -n <N> python shell_integrals.py config_shell.yaml       # see job.batch in the run directory
python check_tracer.py config_shell.yaml --iteration 204800     # one rank, all cores
```

```python
from integrals_tools import load_integral, group
phi = load_integral("Phi_r__<label>.h5")
t = phi["times"]
total, magnetosphere, rest = (group(phi, g) for g in ("total", "horizon", "no_horizon"))
magnetosphere_big = group(phi, "horizon", horizon_scale=1.1)
```

On this cluster launch through `mpirun` (see `job.batch` in the run
directory): a plain-`python` MPI singleton is bound to one core, and the numba
threads then share it. The scripts print threads and cores at start.

Measured (HHR flux run, t = 2560 M, shell 4-5 M about BH1): 8.7e6 points, all
level 8; shell points and integrands ~30 s, tracing field 8.3 GB in ~35 s,
tracing 537 s on 32 threads (≈ 5,700 points/s per 8-thread rank); 10.4 GB peak
per rank.

## Output

One file per quantity, `<quantity>__<label>.h5`: `data` (n_iter, n_hscale,
n_class), `horizon_scales`, `iterations`, `times`, `shell_centre`,
`shell_pole`, `bh1_position`, `bh2_position` (n_iter, 3); for `Phi_z` also
`profile` (n_iter, n_hscale, n_class, n_bin), the flux through the planes
ζ = const bin by bin (its mean over the bins is `data`), and `z_edges`.
Attributes: `classes` (the class names, e.g. `BH1-exit`), `ends`, the
definition, and every setting. `tracing__<label>.h5` holds per iteration the
points and mean line length per class, the summed and exact shell volume,
points per level, horizons, timings and `traced` (0 where the ends were read).
These are written at the end of the run.

Per iteration, as soon as it is done: `points__<label>__it<it>.h5` - every
point's position, volume, level and line ends (`ends`, and `xends` for the
extra horizon scales), a few MB; for the iterations in `save_points` also
line lengths, end points and integrands (~400 MB).

**Horizon sizes.** `horizon_scales` = `[horizon_scale] +
extra_horizon_scales`: the lines are traced with the first, and each records
where it first enters the enlarged spheroids - where it would have ended with
them, so the classes for every size come from one trace, exactly
(`tracer.trace(extra_scales=...)`).

**Restarting, re-integrating.** An iteration whose `points__` file holds ends
traced with the same settings (its `settings` attribute) from the identical
points is not traced again: the integrals are recomputed from the stored ends,
about a minute instead of ten. A run that stopped is submitted again as it is;
so is one with new integrands or masks.

Classes are unordered pairs of the two ends of the line through a point:
`BH1`/`BH2` (enters that horizon), `exit` (leaves the tracing box), `cap`
(reaches the length cap; undetermined), `null` (B = 0). The groups of
`integrals_tools.group` sum them: `horizon` = any end on a horizon (the
magnetospheres), `no_horizon` = the rest, split into `free` (exit both ways)
and `undetermined`.

## Volumes on the AMR grid

Each grid point stands for its dual cell (side dx, Carpet is vertex-centred),
over which the data are constant. A point of level L counts with the part of
its cell that level L+1 does not own, so the levels tile space exactly; a
coarse cell straddling a refinement boundary is split, not given to one level.
Cells a bounding sphere cuts count with the fraction of 8³ sub-points inside
the shell (and outside the finer level). Counting them whole or not at all by
the centre leaves a first-order error that does not average out where a
sphere runs parallel to grid planes (the poles of a shell about a grid axis,
or a refinement boundary near a sphere): up to 1.5% of the volume in the
tests, against ≤ 1e-4 with the fractions. Within 4 cells of the pole axis
|𝓑·φ̂|/R is averaged over the sub-points, since 1/R varies strongly there.

## Tests (`test_shell_cells.py`)

A three-level hierarchy (Carpet layout, components split along x, three ghost
zones), the shell off-centre and its pole tilted so that it crosses both
refinement boundaries; √γ ≠ 1. Weights of all levels tile the coarsest owned
box and each level's box to round-off; uniform field along and normal to the
pole, B_φ ∝ R and radial field against their closed-form integrals (errors
≤ 2e-4 at the coarser, ≤ 6e-5 at the finer spacing); the pole axis on (and
1e-7 off) grid points; per-class sums adding up to the unmasked ones; the
profile's mean equal to Phi_z.
