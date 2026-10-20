# Field-line tracer

Trace magnetic field lines through the raw AMR output of a 3D run and classify
every seed point by where its line ends: on a horizon, outside the tracing box,
at the length cap, or at a null of B. Built for the shell integrals in
`../Integrals/` (which leave out the holes' magnetospheres), but independent of
them: give it seeds, a field and horizons.

| File | Purpose |
| ---- | ------- |
| `amr_field.py` | `AMRVectorField`: a 3-vector field on AMR blocks, packed for numba, with the lookup grid that finds the finest block claiming a point. `from_snapshot` loads B^i of a box from Carpet output (through `../Resample/amr_blocks.py`); `from_function` samples a function on a synthetic hierarchy (tests). |
| `horizons.py` | `Horizons`: absorbing spheroids; `Horizons.kerr_schild` builds each hole's analytic Kerr-Schild horizon from its position, spin vector a = S/m and mass. |
| `tracer.py` | The numba kernels (block lookup, trilinear interpolation, RK4) and the Python API: `trace(field, horizons, seeds, ...)` → `TraceResult` (end codes both ways, line lengths, end points), `classify`, `sample`. |
| `test_tracer.py` | Analytic fields with known lines: `python test_tracer.py` → ALL OK. |

## Use

```python
import sys; sys.path[:0] = ["~/PythonAnalysisETK/FieldLines", "~/PythonAnalysisETK/Resample"]
from amr_blocks import Snapshot                  # Resample/
from amr_field import AMRVectorField
from horizons import Horizons
import tracer

snap = Snapshot(sim.grid.xyz, it, "Bx")          # the grid hierarchy of one iteration
field = AMRVectorField.from_snapshot(snap, ["Bx", "By", "Bz"], c - 12, c + 12)
hz = Horizons.kerr_schild([pos1, pos2], [a1, a2], [m1, m2])
res = tracer.trace(field, hz, seeds, step=0.5, max_length=200.0)
res.ends          # (n, 2): end along B, end against B (codes into res.legend)
res.classes()     # unordered pair index into res.class_names(): "BH1-exit", ...

res = tracer.trace(field, hz, seeds, extra_scales=(1.1,))
res.classes(0)    # the classes had the horizons been 1.1x larger - same lines
```

## What it does

* **Field.** B^i is read on every component whose owned box meets the tracing
  box (all levels, ghost zones included, a few points beyond the box). At any
  point the finest component that owns it supplies the value, trilinearly -
  the ownership rule and `interp_order: 1` of the resampling pipeline. The
  shape of a field line does not depend on the normalisation of B, so no
  metric is needed.
* **Lines.** dx/dl = ±B/|B| in coordinate arc length, RK4 with a fixed step of
  `step` times the local grid spacing (finest level at the start of each step),
  followed both ways from each seed.
* **Ends.** A horizon when the line enters its spheroid (tested after every
  step), `exit` on leaving the box or the data, `cap` at `max_length` per
  half-line, `null` where |B| is zero or not finite. A seed's class is the
  unordered pair of its two ends.
* **Horizons.** For the superposed Kerr-Schild metric of the AnalyticalSpacetime
  runs, each horizon is the spheroid ρ²/(r₊² + a²) + ζ²/r₊² = 1 about the
  trajectory position, spin along a; boost (v ≈ 0.09) ignored. m = 0.5,
  χ = 0.9: polar semi-axis 0.718 M, equatorial 0.847 M.
* **Enlarged horizons.** `extra_scales` records, per half-line, the first
  enlarged spheroid it enters: up to there the line is the same curve, so
  that is its end with the larger horizons - identical to tracing with them
  (tested), at no extra tracing cost.
* **Threads.** numba `prange` over the seeds, in a shuffled order (costs range
  from a few steps to the cap). The result does not depend on the order, the
  batching or the thread count.

## Tests (`test_tracer.py`)

On one- and two-level synthetic hierarchies with Carpet's ghost-zone layout
and a unit-sphere horizon: trilinear interpolation exact for a linear field,
the finest level supplying each point; uniform field (lines through R < 1 hit
the sphere one way), dipole (lines close on the sphere unless they leave the
box about the equator), split monopole (one end on the sphere, polarity of the
ends), toroidal circles (both halves capped, R conserved), zero field (null);
the dipole invariant L = r³/R² conserved to the footpoints, its drift falling
at second order in dx; Kerr-Schild semi-axes and the inside test, also for a
tilted spin; independence of seed order and batching; the enlarged-horizon
ends equal to tracing with the enlarged horizons.
