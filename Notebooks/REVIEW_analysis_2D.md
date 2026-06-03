# Review: `Notebooks/analysis_2D.ipynb`

Reviewed 2026-06-03 against the current repository tooling
(`Resample/` HDF5 pipeline, `Frames/make_frames.py`).

---

## 1. What the notebook does (components)

| Section | Purpose |
| --- | --- |
| **Imports / backend** | postcactus + matplotlib setup. |
| **Load simulation directories** | Four `SimDir`s (`tnt`, `low`, `med`, `high`) + hardcoded merger times. |
| **Constants** | Unit conversions (geometric → km / ms / MeV / cgs), colormaps, contour levels, font sizes. |
| **A) Scalar data** | Loads `ts.max` / `ts.min` time series (`rho_b`, `pi`, `temp` max; `alp` min) per simulation; plots maximum-density / maximum-temperature evolution and minimum lapse. |
| **Grid settings** | An 800×800, ±50 M `RegGeom` (only relevant to resampling, unused by the plots). |
| **Read pickle files** | Loads 16 monolithic pickles (4 variables × 4 simulations) entirely into RAM. |
| **Iteration bookkeeping** | Builds `all_it` / `all_times` per simulation; `find_it(t)` maps a *t − t_merge* in ms to the per-simulation iteration (first time ≥ target). |
| **Publication figure** | 2×2 panel: Π/p (SymLog, PRGn) for `tnt` vs `high` on top, T in MeV (Log, magma) below, ρ contours overlaid, selective colorbars, saved as `pip_temp.pdf`. |
| **Movie section** | A serial frame loop (`range(700, MINLEN, 3)`) producing 4-sim ρ comparison PNGs — explicitly marked "Not updated with settings above". |

The interactive parts (scalar exploration, `find_it`, the one-off publication
figure) are exactly what a notebook is for. The movie section is not — that
job now belongs to `Frames/make_frames.py`.

---

## 2. Bugs and errors

1. **`data_dict` is silently destroyed.** Section A.2 builds `data_dict`
   (scalar time series); the pickle-loading cell then reuses
   `data_dict = pickle.load(r)` as a loop temporary, overwriting it with the
   last pickle read. Any scalar plot re-run after loading pickles will fail or
   silently plot wrong content. The notebook only works when executed strictly
   top-down, once.
2. **`g` is shadowed twice.** `g` is the `RegGeom` ("Grid settings"), then a
   pickle file handle (`with open(...) as g:`). The same shadowing existed in
   the old `movie_maker_parallel.py`. Execution-order-dependent breakage.
3. **The movie cell references `clrs`, which is never defined** in this
   notebook (only `clrs1` / `clrs2` exist) → instant `NameError`. Consistent
   with its "Not updated" comment — the cell is dead as written.
4. **The movie cell saves to `./rho_plots/`, which is never created** →
   `savefig` raises on the first frame even after fixing `clrs`.
5. **Suptitle mixes simulations.** The figure suptitle uses
   `all_times[key][this_it["tnt"][1]]` where `key` is still `'high'` from the
   last panel but the index comes from `tnt` — i.e. *high*'s time list indexed
   with *tnt*'s position. The merger times differ by ~ms so the printed
   t − t_merge can be subtly wrong.
6. **`MINLEN = 2170` is hardcoded** directly under a cell that prints the four
   actual lengths (2217/2226/2170/2200). Forgetting to update it for a new
   dataset silently truncates or (if too large) crashes the movie loop. Should
   be `MINLEN = min(len(v) for v in all_it.values())`.
7. **`mpl.use('Agg')` followed by `%matplotlib inline`** — the Agg call is at
   best pointless in a notebook and at worst suppresses figure display if the
   magic cell isn't run; pick one (the magic).
8. **Trailing `plt.xticks(...)/plt.yticks(...)`** act on the *last active
   axes*, which after the colorbar calls is a colorbar axis — not the panels.
   Harmless here, but it doesn't do what it looks like it does.
9. **Movie frames aligned by index** (`iii` into per-simulation iteration
   lists with different lengths/cadences) — same assumption as the old
   scripts; correct only when all four simulations were output with identical
   iteration spacing.

---

## 3. Efficiency

1. **~10s of GB of RAM for eager pickle loading.** 16 monolithic pickles
   (4 vars × 4 sims × ~2200 iterations × 800×800 float64 ≈ 5 MB/slice) are
   fully unpickled up front, even when a session only ever touches a handful
   of iterations. This is the single biggest cost — in load time *and* memory
   — and the reason the resampled-HDF5 format exists: `h5py` slices lazily,
   so reading one frame costs one frame.
2. **`find_it` is an O(N) linear scan per call** over each simulation's time
   list. With the HDF5 `times` arrays this is one `np.searchsorted` call.
3. **Four `SimDir` scans at import time**, needed only by the scalar section.
   Fine interactively, but worth knowing they re-scan on every kernel restart
   (postcactus SimDir caching or pickling `ts.max/min` series would avoid it).
4. **The serial movie loop** renders ~490 frames one by one in the kernel —
   the exact workload `Frames/make_frames.py` now does MPI-parallel, with
   contiguous numbering and ffmpeg assembly.
5. `shading='gouraud'` in the publication figure with 800×800 input is
   noticeably slower to rasterize than `'nearest'` at `dpi=500`; only keep it
   if the smoothing is wanted in print.

---

## 4. Readability and future usability

1. **Opaque variable naming:** `pickle_dict`, `pickle_dict_2`,
   `pickle_dict_3`, `pickle_dict_rho` hold Π, P, T, ρ respectively — but only
   the path strings reveal that. A single `fields[sim][var]` mapping (or the
   `read_data.py` accessors) removes the guesswork.
2. **4× copy-pasted panel blocks** in both the figure and movie cells,
   differing only in `key` and axis position — the same duplication that was
   factored into a panel loop in `make_frames.py`. A 10-line
   `plot_panel(ax, sim, ...)` helper would shrink the figure cell by ~75% and
   make style experiments (the notebook's main purpose!) one-line changes.
3. **Dead weight:** duplicated grid bounds (`mini_*`/`maxi_*` vs
   `min_*`/`max_*`), unused constants (`vmin/vmax = 5/0.5`, `linthresh`,
   `plotting_styles`, `color_stream`, `dens_stream0`, `cont`, `rhosat`), the
   unused `RegGeom`, and large blocks of commented-out alternatives. Trimming
   these would roughly halve the "Definitions" noise.
4. **Hardcoded cluster paths** in two places (SimDirs + 16 pickle paths).
   One `DATA_DIR` + label-based file naming (as the Resample pipeline already
   produces) collapses 40 lines of path dictionaries into ~6.
5. **Old data format:** the notebook is the last consumer of the legacy
   monolithic pickles. Porting the loading cell to the resampled HDF5 +
   `Resample/read_data.py` removes the format dependency, the RAM blowup, and
   bugs (1)–(2) in one move, and gives `iterations`/`times` arrays for free
   (which also fixes (6) and simplifies `find_it`).
6. **Repository hygiene:** the notebook is stored with outputs embedded
   (~760 KB). Stripping outputs before committing (e.g. `nbstripout`) keeps
   diffs reviewable.

---

## 5. Recommended workflow (resample → notebook → frames)

Your pipeline — *resample first, explore in the notebook, render the final
movie with Frames* — is exactly the right division of labor, and the pieces
are now built to hand off to each other; the trick is to make the notebook
speak the same two languages as its neighbors. **Upstream:** load data via
`Resample/read_data.py` (or `h5py` directly) instead of pickles, so
exploration is lazy and instant — open the per-variable `.h5` files once,
use `np.searchsorted(times, t_merge + t_ms*MILLIS)` to jump to a physical
time, and slice only the frames you're looking at; nothing is decoded until
you index it, so even a 16-variable, 4-simulation session stays at megabytes
of RAM. **Downstream:** prototype each plot in the *shape make_frames already
consumes* — a panel dict with a `data=lambda f, i: ...` expression, a norm, a
cmap, and a contour list. You can even do this literally: `import
make_frames as M` in the notebook (add `../Frames` to `sys.path`), point
`M.FIELDS` at your files, build a trial `M.PANELS`, call
`M.render_frame(M.Fields(M.FIELDS), idx, 0)` for a single frame, and display
the saved PNG with `IPython.display.Image` — what you see is *pixel-identical*
to what the movie will contain. Then promoting a finished plot to a movie is
a copy-paste of the panel dict and constants into `make_frames.py`'s SETTINGS
block, followed by `mpirun -n N python make_frames.py` — no re-translation of
plotting code, no drift between the notebook figure and the final animation.
The one convention to respect end-to-end: resample all simulations you intend
to compare with the same iteration stride and box, since both the notebook's
index bookkeeping and the frame renderer align datasets by frame index.
