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
python XScripts/quick_diagnose.py /path/to/output --inspect
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
  --nxny 1024 \
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
--nxny N            output canvas size, default 1024
--method linear     linear interpolation, falls back to nearest without scipy
--method nearest    nearest-neighbor sampling
--scale log         logarithmic color scale, default
--scale linear      linear color scale
--no-movie          write PNG frames only
```

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

`quick_diagnose.py` is the first-pass tool. With `--inspect`, it opens one 2D
and/or 3D sample and reports mesh count, parseable AMR names, components, and
shapes without loading field data.

`inspect_chunks.py` is the most important data-safety diagnostic. It reports how
much of each declared mesh is actually written. Use it when plots look wrong or
when a file appears enormous.

`show_mesh_names.py` is useful when `--variable` does not match what you expect.

## Notes on `read_full()`

`openpmd_common.OpenPMDField.read_full()` exists only as a convenience for cases
where the written chunk bounding box is known to be manageable. It still may
allocate a large dense array. Prefer `read_chunks()` or script-specific
chunk/slab iteration for production plotting.
