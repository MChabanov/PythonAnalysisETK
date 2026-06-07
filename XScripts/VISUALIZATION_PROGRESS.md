# CarpetX 2D/3D Data Visualization Scripts - Progress Report
**Date**: 2026-06-07  
**Status**: Analysis phase complete, ready for implementation  
**Token usage**: ~41% of session budget

## Task Summary
Create reusable Python visualization scripts for EinsteinToolkit CarpetX 2D and 3D data, compare tool applicability across dimensions, and save progress when reaching 80% token usage.

## Key Findings

### 1. Existing Code Structure

#### 2D Visualization (PlanesX output)
- **Location**: `../CarpetX/TestPlanesX/test/`
- **Reader**: `plane_readers/openpmd_reader.py` - reads openPMD plane files (ADIOS2 .bp/.bp5 or HDF5 .h5)
- **Data structure**: Pre-extracted 2D planes written at simulation runtime
- **Features**: 
  - Handles multi-level AMR data
  - Supports per-mesh attributes (gridSpacing, gridGlobalOffset, planeCoordinate)
  - Geometry-aware (normal axis + in-plane axes)
  - Test verification: `verify_planes.py` validates plane data analytically

#### 3D Visualization (CarpetX 3D output)
- **Location**: `../AsterX-Docs/scripts/plot2d_BNS_rho.py`
- **Approach**: Extract 2D slices from full 3D AMR data
- **Key features**:
  - Reads 3D openPMD outputs
  - Composites multiple refinement levels (coarse → fine, fine overwrites interior only)
  - Extracts planes along coordinate axes
  - Uses SciPy RegularGridInterpolator for interpolation
  - Custom edge-eroding logic to hide refinement boundaries
  - Supports movie generation (MP4/GIF)

#### Common I/O Library
- **openPMD API**: Both use `openpmd_api` for file reading
- **Data format**: openPMD (ADIOS2 or HDF5 backend)

### 2. Tool Applicability Analysis

#### **Shared/Reusable Components** ✓
1. **openPMD file reading**: Both use `openpmd_api.Series()`
2. **Metadata extraction**: gridSpacing, gridGlobalOffset, time attributes
3. **Matplotlib styling**: LaTeX support, colormaps, normalization (LogNorm)
4. **Movie generation**: imageio + FFmpeg for frame assembly
5. **Canvas composition**: Both handle multi-level/multi-patch data aggregation

#### **Dimension-Specific Components** ✗
1. **Slice extraction logic**: 
   - 2D: Already sliced in file (just read)
   - 3D: Must extract planes from volumetric data
2. **Interpolation**: 
   - 2D: Simple coordinate lookup
   - 3D: Full RegularGridInterpolator for multiple levels
3. **Refinement boundary handling**:
   - 2D: May not need edge erosion (pre-extracted)
   - 3D: Needs careful boundary blending

### 3. Data Writer Details

#### CarpetX writer (`CarpetX/src/io_openpmd.cxx`)
- Writes full 3D AMR data
- Creates hierarchy of mesh objects with level/patch metadata
- Stores gridSpacing, gridGlobalOffset, time attributes

#### PlanesX writer (`PlanesX/src/openpmd_planes.cxx`)
- Writes pre-extracted 2D plane files
- Adds plane-specific metadata: `planeNormalAxis`, `planeElevation`, `planeCoordinate`
- Handles multiple planes per iteration
- Geometry-free format (coordinates embedded in file)

---

## Implementation Plan

### Phase 1: Common Utilities Module
**File**: `Xscripts/openpmd_common.py`
- Shared matplotlib config (styling, colormap setup)
- openPMD file discovery and metadata extraction
- Canvas composition base class
- Movie generation utilities

### Phase 2: 2D Visualization Script
**File**: `Xscripts/plot_2d_planes.py`
- Reads PlanesX openPMD output
- Generates per-plane slices
- Supports multiple variables and colormaps
- Movie generation option

### Phase 3: 3D Visualization Script
**File**: `Xscripts/plot_3d_slices.py`
- Reads full 3D CarpetX openPMD output
- Extracts slices along user-specified axes
- Handles AMR level composition (with edge erosion)
- Optional interpolation to uniform grid
- Movie generation option

### Phase 4: Comparison/Analysis Tool
**File**: `Xscripts/compare_2d_3d_tools.md`
- Document findings on tool overlap
- Provide migration guide (2D↔3D)
- Note incompatibilities and workarounds

---

## Next Steps

1. **Create `Xscripts/` directory** if missing
2. **Implement `openpmd_common.py`** with shared utilities
3. **Implement `plot_2d_planes.py`** (simpler, good warm-up)
4. **Implement `plot_3d_slices.py`** (reuses common module)
5. **Test both scripts** with sample data (if available locally)
6. **Document results** in comparison report

---

## Known Constraints & Notes

- Both formats depend on `openpmd_api` (non-standard dependency)
- PlanesX output is typically single-level (no complex AMR composition needed)
- 3D slicing requires careful interpolation to avoid artifacts
- Movie generation optional but depends on ffmpeg/imageio
- Edge erosion (refinement boundary hiding) only needed for 3D compositing

---

## Proceed When Ready

All exploratory analysis complete. Ready to start implementation in `Xscripts/`.
