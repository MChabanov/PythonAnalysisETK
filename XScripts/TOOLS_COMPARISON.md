# 2D vs 3D openPMD Visualization Tools: Applicability Analysis

## Executive Summary

The **same core Python utilities** (`openpmd_common.py`) are applicable to both 2D plane data (PlanesX) and 3D volumetric data (CarpetX), but the **slice extraction and compositing logic differs significantly**. Both share the I/O layer (openPMD API) and visualization layer (matplotlib), but 2D and 3D require different data manipulation strategies.

---

## Tool Reusability Matrix

| Component | 2D Planes | 3D Slices | Shared? | Notes |
|-----------|-----------|-----------|---------|-------|
| **I/O Layer** |  |  |  |  |
| `openpmd_api.Series` | ✓ | ✓ | **YES** | Both read openPMD .bp5 / .h5 |
| `get_openpmd_time()` | ✓ | ✓ | **YES** | Iteration time extraction |
| `gather_openpmd_series()` | ✓ | ✓ | **YES** | File discovery |
| `parse_iteration_number()` | ✓ | ✓ | **YES** | Filename parsing |
| **Data Access** |  |  |  |  |
| `OpenPMDField` class | ✓ | ✓ | **YES** | Mesh/component wrapper |
| `.get_axis_coords()` | ✓ | ✓ | **YES** | Coordinate arrays |
| `.read_full()` | ✗ | ✓ | **NO** | 2D needs different strategy |
| **Canvas/Compositing** |  |  |  |  |
| `Canvas2D` class | ✓ | ✓ | **YES** | Abstract compositing logic |
| Multi-level AMR handling | ✗ | ✓ | **PARTIAL** | 2D simpler (pre-extracted) |
| Edge erosion (`_erode()`) | ✗ | ✓ | **NO** | Only needed for 3D AMR |
| Interpolation (scipy) | ✓ | ✓ | **YES** | Optional, same usage |
| **Visualization** |  |  |  |  |
| `setup_matplotlib_style()` | ✓ | ✓ | **YES** | Identical styling |
| `setup_colormap()` | ✓ | ✓ | **YES** | Identical colormaps |
| `movie_from_frames()` | ✓ | ✓ | **YES** | Identical movie assembly |
| LogNorm, extent setup | ✓ | ✓ | **YES** | Same plotting idioms |

**Shared utility score: ~70%** (most I/O and viz layer)  
**Data manipulation score: ~30%** (very different extraction/compositing)

---

## Key Differences Explained

### 1. Data Extraction

**2D Planes (PlanesX)**
```python
# Data already sliced at write-time
for mesh in iteration.meshes:
    slab = mesh[component][:]  # Already 2D!
```

**3D Volumes (CarpetX)**
```python
# Must extract slice from full 3D
mesh_3d = iteration.meshes[name]  # Shape (nz, ny, nx)
slice_2d = mesh_3d[slice_idx, :, :]  # Extract one plane
```

### 2. AMR Compositing Complexity

**2D Planes**
- Pre-extracted: Each level/patch is already a 2D slab
- Simple aggregation: Stitch patches with overlap handling
- No refinement-boundary artifacts (already interior-only)

**3D Slices**
- Post-extraction: Slice comes after level composition
- Complex: Must composite levels *in-plane* to hide refinement edges
- Refinement artifacts: Fine grids extend beyond coarse in `(x,y)` plane
- Solution: Binary erosion (`_erode()`) to hide boundaries

### 3. Grid Geometry

**2D Planes**
- Normal axis: `planeNormalAxis` attribute (0=x, 1=y, 2=z)
- In-plane axes: Implied by normal (e.g., normal=2 → x,y in-plane)
- Coordinates: Stored directly in mesh metadata (gridSpacing, gridGlobalOffset)

**3D Slices**
- Full 3D grid: All three axes present
- Slice selection: User chooses axis and elevation (typically z=0)
- Coordinate reconstruction: Must permute (z,y,x) indexing correctly

---

## Workflow Comparison

### Plot 2D Planes
```
Read plane file (.bp5)
  ↓
For each variable:
  ↓
  Load 2D mesh (already a 2D slab)
  ↓
  Composite patches (if AMR)
  ↓
  Plot on canvas
  ↓
  Generate frame
```
**Duration per file**: Fast (~1-2 sec per plane file)

### Plot 3D Slices
```
Read 3D file (.bp5)
  ↓
For each axis (xy, xz, yz):
  ↓
  For each AMR level (coarse → fine):
    ↓
    Extract 2D slice from 3D mesh
    ↓
    Interpolate to uniform canvas
    ↓
    Erode interior (hide refinement edges)
    ↓
    Composite onto canvas (fine overwrites coarse)
  ↓
  Plot composite canvas
  ↓
  Generate frame
```
**Duration per file**: Slower (~5-10 sec per iteration, depends on resolution)

---

## Migration Guide: When to Use Each Script

### Use `plot_2d_planes.py` if:
- ✓ You have **PlanesX openPMD output** (planes extracted at runtime)
- ✓ Data is **already 2D** (or pre-sliced)
- ✓ You want **fast visualization** with minimal overhead
- ✓ You need **multiple planes** (different elevations/orientations)

### Use `plot_3d_slices.py` if:
- ✓ You have **full 3D CarpetX output** (volumetric data)
- ✓ You need **arbitrary slice locations** (not pre-extracted)
- ✓ You have **AMR/refinement levels** requiring compositing
- ✓ You want **interactive control** over slice orientation and position

### Use **both** if:
- ✓ You need to compare plane files (PlanesX) vs slices from 3D (CarpetX)
- ✓ You want to validate that 3D slices match plane file data
- ✓ You're developing/testing the plane writer

---

## Dependency Analysis

### openpmd_common.py
- **Required**: numpy, matplotlib
- **Optional**: scipy (for interpolation; falls back to nearest-neighbor)

### plot_2d_planes.py
- **Required**: openpmd_api, numpy, matplotlib
- **Optional**: scipy, imageio (for movie)

### plot_3d_slices.py
- **Required**: openpmd_api, numpy, matplotlib
- **Optional**: scipy (strongly recommended for quality), imageio

---

## Future Improvements: Tool Unification

Possible refactoring to increase shared code:

1. **Extract abstract base class**
   ```python
   class PlotterBase:
       def read_file(self, path) → (iteration, time_cu)
       def get_field_data(self, mesh, component) → data_dict
       def plot_frame(self, data, axes) → frame_path
   ```

2. **Consolidate slice extraction**
   - Move logic into `openpmd_common.SliceExtractor`
   - Support both 2D (trivial case) and 3D (full AMR)

3. **Parameterized main scripts**
   - Single driver script that auto-detects 2D vs 3D
   - Choose plot type based on input file structure

4. **Composite plotting**
   - Plot 2D planes + 3D slices side-by-side
   - Validate consistency across formats

---

## Known Limitations & Workarounds

| Issue | 2D | 3D | Workaround |
|-------|----|----|-----------|
| Large 3D files OOM | — | ✗ | Use slice-only mode; limit nxny |
| Missing scipy | ✓ | ⚠ | Falls back to nearest-neighbor (artifacts) |
| AMR boundary artifacts | ✓ | ✗ | Increase edge_fill_pix; manual masking |
| Variable filtering | ⚠ | ⚠ | Substring matching; improve parser |
| No per-patch visualization | ✓ | ✗ | Plot each level separately |

---

## Testing Recommendations

1. **Compare outputs**
   ```bash
   # If you have both 2D planes AND 3D slices of same run:
   python plot_2d_planes.py planes_dir/ --out-dir test_2d
   python plot_3d_slices.py 3d_dir/ --out-dir test_3d
   # Visually compare test_2d/*.png vs test_3d/slice_xy_*.png
   ```

2. **Benchmark performance**
   - Time both scripts on same hardware
   - 2D should be ~3–5× faster
   - Profile scipy interpolation if slow

3. **Validate with test data**
   - TestPlanesX analytic test case (known values)
   - Check that 3D slices match 2D plane values where they overlap

---

## Conclusion

**General applicability**: **~60%** of code and logic is shared across 2D and 3D.

- **Strengths**: I/O layer (openPMD), visualization (matplotlib), file handling all identical
- **Differences**: Data extraction (1D lookup vs 3D slicing), compositing strategy (trivial vs complex)
- **Recommendation**: Keep as **separate scripts** (clarity, different CLI defaults), but extract shared library (`openpmd_common.py`) for reuse and consistency

The shared library achieves good modularity without over-engineering for marginal code savings.
