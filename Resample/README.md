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
| `resample_2d_data_postcactus.py`| MPI-parallel resampler — **postcactus** backend.       |
| `resample_2d_data_kuibit.py`    | MPI-parallel resampler — **kuibit** backend (same out).|
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
never holds more than one 2D slice in memory.

Output goes to `output_dir` as `<variable>__<label>.h5`, e.g.
`rho_b__dU_10_15_linear_HR.h5`.

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
not a bug — pick whichever backend matches the environment you have available.
kuibit additionally reads openPMD output and is the maintained option going
forward.

### Environments used here

- postcactus: available in the `base` conda env (Python 3.8).
- kuibit: needs Python ≥ 3.9; installed in a dedicated `kuibit` conda env
  (`conda create -n kuibit python=3.11 && pip install kuibit h5py mpi4py pyyaml`).
