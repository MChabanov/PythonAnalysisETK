# CarpetX / PlanesX openPMD Visualization Scripts

Utilities for inspecting and plotting CarpetX/PlanesX openPMD output. The main
focus is sparse AMR data written as BP5/HDF5, where the declared mesh extent can
be much larger than the data that was actually written.

## Core Rule

Do not load the full declared extent of a sparse AMR mesh unless you have already
verified it is small and dense.

For PlanesX/openPMD output, a mesh can declare the full refined domain while only
writing the AMR boxes that intersect the plane. Reading the full extent can pull
unwritten padding and produce garbage values. The safe pattern is:

1. Iterate `available_chunks()`.
2. Clip cell-centered fill rows/columns when needed.
3. Read each written chunk with `load_chunk(offset, extent)`.
4. Composite chunks coarse-to-fine on a plotting canvas.

This is the behavior used by the current plotters and diagnostics.

## ADIOS2 BP5 Directories

BP5 output may appear as a directory ending in `.bp5`, not as a single regular
file. This is normal for ADIOS2 parallel output. Pass the `.bp5` directory path
directly to `openpmd_api` or to these scripts; `openpmd_api` handles the metadata
and `data.*` files internally.

## Plane Tag Naming

PlanesX plane files and mesh names use a tag of the form:

```text
<plane>_<normal-axis>_<sign><integer-digits>p<fraction-digits>
```

Examples:

- `xy_z_pos0012p500` means an `xy` plane at `z = +12.5`.
- `xz_y_neg0003p000` means an `xz` plane at `y = -3.0`.

The number of integer and fractional digits is configurable in Einstein Toolkit
with `planes_int_precision` and `planes_frac_precision`, so the parser accepts
variable-width tags such as `xy_z_pos12p5` and `xy_z_pos0012p500`.

Mesh names may also include a two-character in-plane centering suffix before
`patch`/`lev`, for example `_cv` or `_vc`. The plotting scripts strip that suffix
from the group name and use the openPMD component name as the plotted variable
label whenever possible.

## Files

| File | Purpose |
| --- | --- |
| `plot_2d_planes.py` | Main PlanesX 2D plotter. Reads written chunks, resamples to a uniform canvas, and composites AMR levels coarse-to-fine. |
| `plot_3d_slices.py` | 3D CarpetX slice plotter. Loads one-cell-thick slabs from chunks that intersect the requested slice. |
| `openpmd_common.py` | Shared openPMD, plotting, chunk-reading, colormap, and movie helpers. |
| `quick_diagnose.py` | Fast directory scan with optional metadata-only mesh inspection. |
| `inspect_chunks.py` | Shows declared extent vs actual written chunks for one series/variable. |
| `show_mesh_names.py` | Lists mesh names, components, shapes, and naming patterns. |
| `plot_openpmd_plane.py` | Minimal chunk-safe reference plotter, kept temporarily while the main plotter settles. |
| `test_single_plot_v2.py` | Known successful hard-coded reference used to guide the main 2D plotter. |

## Dependencies

Required:

```bash
pip install openPMD-api numpy matplotlib
```

Recommended:

```bash
pip install scipy imageio imageio-ffmpeg
```

`scipy` enables linear interpolation. Without it, the plotters fall back to
nearest-neighbor sampling. `imageio` and ffmpeg are only needed for movies.

## Typical Workflow

Start with a quick scan:

```bash
python XScripts/quick_diagnose.py /path/to/output hydrobasex_rho --inspect
```

Inspect mesh names if variable matching is unclear:

```bash
python XScripts/show_mesh_names.py /path/to/output/file.it00000000.bp5
```

Inspect actual chunks for a variable:

```bash
python XScripts/inspect_chunks.py /path/to/output/file.it00000000.bp5 hydrobasex_rho
```

Plot 2D PlanesX output:

```bash
python XScripts/plot_2d_planes.py /path/to/planes \
  --variable hydrobasex_rho \
  --plane xy \
  --nxny 1024 \
  --out-dir planes_frames \
  --no-movie
```

Plot only a physical region:

```bash
python XScripts/plot_2d_planes.py /path/to/planes \
  --variable hydrobasex_rho \
  --plane xy \
  --xmin -500 --xmax 500 \
  --ymin -500 --ymax 500 \
  --nx 1200 --ny 800 \
  --vmin 1e-8 --vmax 1e-3 \
  --cmap plasma \
  --out-dir planes_frames \
  --no-movie
```

Plot one BP5 series directly:

```bash
python XScripts/plot_2d_planes.py /path/to/parfile.xy_z_pos0000p000.it00000000.bp5 \
  --variable hydrobasex_rho \
  --out-dir planes_frames \
  --no-movie
```

Plot a 3D slice:

```bash
python XScripts/plot_3d_slices.py /path/to/3d_output \
  --variable hydrobasex_rho \
  --axes xy \
  --slice-value 0.0 \
  --nxny 1024 \
  --vmin 1e-8 --vmax 1e-3 \
  --cmap plasma \
  --out-dir slices_frames \
  --no-movie
```

## 2D Plotter Notes

`plot_2d_planes.py` is the production version of the successful
`test_single_plot_v2.py` workflow:

- extent comes from actual written chunks on the coarsest available level;
- each chunk is mapped into world coordinates;
- chunks are interpolated or sampled onto a fixed `--nxny` by `--nxny` canvas;
- finer AMR levels overwrite coarser levels;
- output filenames include the input series basename to avoid collisions between
  planes at the same iteration.

Useful options:

```bash
--variable TEXT      variable/mesh/component substring to plot
--tag TAG            exact plane tag, e.g. xy_z_pos0012p500
--plane xy           select one plane family: xy, xz, or yz
--normal-axis z      select by plane normal axis
--elevation 12.5     select by parsed plane elevation
--xmin X --xmax X    physical x extent to plot
--ymin Y --ymax Y    physical y extent to plot
--nx NX             output points in x
--ny NY             output points in y
--nxny N            set both --nx and --ny, default 1024
--vmin V            color scale minimum
--vmax V            color scale maximum
--cmap NAME         matplotlib colormap name, default plasma
--method linear     linear interpolation, falls back to nearest without scipy
--method nearest    nearest-neighbor sampling
--scale log         logarithmic color scale, default
--scale linear      linear color scale
--no-movie          write PNG frames only
```

Matplotlib styling for each panel is centralized in `plot_panel()` inside
`plot_2d_planes.py`. Edit that function for axis labels, titles, aspect ratio,
tick styling, colorbar placement, or other per-plot matplotlib customization.

## 3D Plotter Notes

`plot_3d_slices.py` is intended for full 3D CarpetX output. It assumes record
axis order `(z, y, x)`, which matches the existing CarpetX plotting conventions
used here.

Unlike the old version, it does not assemble a full 3D array. For each requested
slice it:

- finds chunks that intersect `--slice-value`;
- reads only a one-cell-thick slab from each intersecting chunk;
- composites levels coarse-to-fine on a uniform canvas.

Use `--axes xy,xz,yz` to select slice planes. If more than one variable matches
and `--all-variables` is not passed, the script plots the first matching label
and reports that choice.

## Diagnostics

`quick_diagnose.py` is the first-pass tool. It accepts either a directory or one
openPMD series path. With `--inspect`, or when a variable is provided, it opens
one 2D and/or 3D sample and reports mesh count, parsed plane tag, parseable AMR
names, components, and shapes without loading field data.

`inspect_chunks.py` is the most important data-safety diagnostic. It reports how
much of each declared mesh is actually written. Use it when plots look wrong or
when a file appears enormous.

`show_mesh_names.py` is useful when `--variable` does not match what you expect.

## Notes on `read_full()`

`openpmd_common.OpenPMDField.read_full()` exists only as a convenience for cases
where the written chunk bounding box is known to be manageable. It still may
allocate a large dense array. Prefer `read_chunks()` or script-specific
chunk/slab iteration for production plotting.
