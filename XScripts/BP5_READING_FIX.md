# BP5 Data Corruption Fix: Per-Chunk Reading

## The Problem

When reading BP5 files with openpmd_api by loading the full declared extent, data values were corrupted with impossibly high values (4.358e+276) and all AMR levels showed identical maximums with exactly 8 NaN values.

**Root Cause**: The declared extent is the full refined domain (ballooning to ~3000²), but only the AMR boxes are actually written. Loading the full extent pulls unwritten padding as garbage data.

## The Solution

Read PlanesX/CarpetX openPMD plane series **per-chunk rather than loading the full declared extent**.

### How It Works

- Opens the series, picks the last iteration (or `--iteration N`)
- For each mesh (= one patch/level, `_levNN` in the name), loops `rc.available_chunks()` and loads only the written boxes at their offsets
- Clips the cell-centred fill row/column (declared shape is the vertex count N, but only N-1 cells were written) — derived from `position == 0.5`, a no-op for ADIOS2 box chunks but essential for HDF5 which reports the whole extent as one chunk
- Builds world-coordinate cell edges from `grid_global_offset + grid_spacing + position` and draws each chunk with pcolormesh, coarse-to-fine so finer levels sit on top, on a shared color scale

### Usage

```bash
python plot_openpmd_plane.py sim.xy_z0.it%08T.bp5                 # first component, last iteration
python plot_openpmd_plane.py sim.xy_z0.it%08T.bp5 hydrobasex_rho_lev00 --save rho.png
python plot_openpmd_plane.py sim.xy_z0.it%08T.h5 --iteration 1024
```

**Note**: The pattern uses the literal `%08T` iteration placeholder (quote it in the shell if needed). Component names are the lowercased `thorn_var` form, e.g. `hydrobasex_rho_lev00`; omit the arg to auto-pick the first component in each mesh.

### Two Caveats for Reuse

1. **Relies on `available_chunks()`** — Works for ADIOS2 and HDF5; for sparse AMR it's what keeps memory bounded
2. **Assumes 2D plane axis ordering `[b, a]`** — That's what the PlanesX writer uses; fine for any PlanesX/CarpetX plane output, but adjust axis labels if pointing at full 3D meshes

## Reference Implementation

See `TestPlanesX/test/plot_openpmd_plane.py` for the working standalone script demonstrating this approach.
