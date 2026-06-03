# Movie frame rendering

Render per-iteration PNG frames (and optionally an mp4) from the resampled 2D
HDF5 files produced by `../Resample`, MPI-parallel over frames.

Replaces the old `movie_maker.py` / `movie_maker_parallel.py` (kept in
`../Archive`): no SimDir scanning, no monolithic pickles, no temp-file
staging — each rank lazily reads only the HDF5 slices for its own frames.

## Usage

`make_frames.py` is one file in three clearly-marked sections:

1. **SETTINGS** — edit these: input files (`FIELDS`), unit conversions,
   frame selection, output/ffmpeg options, figure layout, contour levels,
   and the `PANELS` list (one entry per subplot).
2. **PLOTTING** — the matplotlib code for a single frame (`render_frame`).
   Edit when you need something the panel options don't cover.
3. **MACHINERY** — HDF5 access, MPI loop, ffmpeg. No need to touch.

```bash
mpirun -n 16 python make_frames.py     # parallel: frames round-robin over ranks
python make_frames.py                  # serial fallback (no mpi4py needed)

# with a config YAML (e.g. written by Notebooks/analysis_2D.ipynb):
mpirun -n 16 python make_frames.py movie_config.yaml
```

Any rank count works; output PNGs are numbered contiguously so the ffmpeg
pattern always matches, even with `FRAME_STRIDE > 1`.

## Config YAML and notebook hand-off

`Notebooks/analysis_2D.ipynb` is the place to *choose* a plot: it imports this
module, previews frames with the real `render_frame()`, and writes
`movie_config.yaml` containing every movie parameter (the key set is
`CONFIGURABLE` in `make_frames.py`: input files, units, frame selection,
output/ffmpeg options, figure layout, contour specs). Passing that file on the
command line overrides the SETTINGS section — so those values never need
editing in two places.

**The one thing that must be kept in sync by hand is `PANELS`** (data lambdas,
norms, colormaps, contour wiring): lambdas can't travel through YAML. When a
figure is final in the notebook, copy its panel dicts into `build_panels()`
here.

## Defining panels

Each `PANELS` entry is a dict:

```python
{
    "title": r"low$;~t-t_\mathrm{mer}=$ {t:.3f} ms",   # {t},{t_raw},{it},{idx}
    "ref": "pi",                                # field giving coords/times
    "data": lambda f, i: f.data("pi", i) / f.data("P", i),   # any numpy expr
    "cmap": "PRGn",
    "norm": colors.SymLogNorm(vmin=-1, vmax=1, linthresh=1e-4, base=10),
    "colorbar": True,
    "time_offset": T_MERGE,                     # subtracted before {t}
    "contours": [ {"data": ..., "ref": "rho", "levels": [...],
                   "colors": [...], "linestyles": [...], "linewidths": 1.3} ],
}
```

- `f.data(name, i)` reads field `name` at frame `i`, so panels can show any
  combination of fields (ratios, `abs(w - 1)`, …).
- Title placeholders are substituted without disturbing LaTeX braces.
- For a multi-simulation comparison (the old `movie_maker_parallel.py` use
  case) add each simulation's files to `FIELDS` under distinct names
  (`rho_tnt`, `rho_low`, …), set `FIG_ROWS`/`FIG_COLS`, and add one panel per
  simulation with its own `time_offset` (merger time).

## Frame alignment

Frames are aligned across fields **by index**: frame `k` uses the `k`-th
stored iteration of every field, and the shortest field sets the frame count
(same convention as the old scripts). When mixing simulations, resample them
with consistent iteration lists.
