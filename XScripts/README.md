# CarpetX 2D/3D Visualization Scripts

Simple, publication-quality visualization tools for EinsteinToolkit CarpetX openPMD output (2D planes and 3D slices).

## Quick Start

### Dependencies

```bash
# Required
pip install openpmd-api numpy matplotlib

# Optional but recommended
pip install scipy imageio imageio-ffmpeg
```

### 2D Plane Visualization

For **PlanesX output** (pre-extracted 2D planes):

```bash
python plot_2d_planes.py /path/to/planes/ --out-dir frames_2d --fps 12
```

### 3D Slice Visualization

For **full 3D CarpetX data** (extract slices on-the-fly):

```bash
python plot_3d_slices.py /path/to/3d_data/ --axes xy,xz,yz --out-dir frames_3d
```

---

## Files in This Directory

| File | Purpose |
|------|---------|
| `openpmd_common.py` | Shared utilities for I/O, matplotlib styling, canvas compositing |
| `plot_2d_planes.py` | Main script for visualizing PlanesX 2D plane output |
| `plot_3d_slices.py` | Main script for visualizing CarpetX 3D data with slice extraction |
| `TOOLS_COMPARISON.md` | Detailed analysis of tool applicability across 2D and 3D |
| `PROGRESS.md` | Initial analysis and implementation plan |
| `README.md` | This file |

---

## Script Usage

### `plot_2d_planes.py`

Extract and visualize 2D plane data from openPMD files.

```bash
usage: plot_2d_planes.py [-h] [--out-dir DIR] [--fps N] [--nxny N]
                         [--vmin V] [--vmax V] [--cmap CMAP] [--variable VAR]
                         [--no-movie]
                         data_dir
```

**Arguments:**
- `data_dir`: Directory containing plane .bp5/.h5 files
- `--out-dir DIR`: Output directory for frames (default: `planes_frames`)
- `--fps N`: Movie frame rate (default: 12)
- `--nxny N`: Resample to N×N grid (default: native resolution)
- `--vmin V`, `--vmax V`: Color scale bounds (auto-detected if not set)
- `--cmap CMAP`: Colormap name, e.g., "plasma", "viridis" (default: "plasma")
- `--variable VAR`: Filter to variables matching this substring
- `--no-movie`: Skip movie assembly (keep frames only)

**Example:**

```bash
python plot_2d_planes.py ../CarpetX/TestPlanesX/output/ \
  --out-dir test_planes \
  --cmap plasma \
  --vmin 1e-10 --vmax 1e3 \
  --fps 10
```

### `plot_3d_slices.py`

Extract 2D slices from 3D CarpetX openPMD data and composite AMR levels.

```bash
usage: plot_3d_slices.py [-h] [--axes AXES] [--out-dir DIR] [--nxny N]
                         [--vmin V] [--vmax V] [--cmap CMAP] [--fps N]
                         [--interpolate] [--no-movie]
                         data_dir
```

**Arguments:**
- `data_dir`: Directory containing 3D .bp5/.h5 files
- `--axes AXES`: Comma-separated slice planes (default: "xy,xz,yz")
  - `xy`: slice along z-axis (bird's eye view)
  - `xz`: slice along y-axis (side view)
  - `yz`: slice along x-axis (front view)
- `--out-dir DIR`: Output directory for frames (default: `slices_frames`)
- `--nxny N`: Canvas resolution per axis (default: 1024)
- `--vmin V`, `--vmax V`: Color scale bounds
- `--cmap CMAP`: Colormap name (default: "plasma")
- `--fps N`: Movie frame rate (default: 12)
- `--interpolate`: Use scipy RegularGridInterpolator (higher quality, slower)
- `--no-movie`: Skip movie assembly

**Example:**

```bash
python plot_3d_slices.py ../data/BNS_IG_fixedGrid/ \
  --axes xy,xz \
  --nxny 1024 \
  --interpolate \
  --out-dir bns_slices
```

---

## OpenPMD Common Library

`openpmd_common.py` provides reusable utilities:

### Key Functions

```python
# Matplotlib setup
plt = openpmd_common.setup_matplotlib_style(use_tex=None)

# File discovery
files = openpmd_common.gather_openpmd_series(data_dir)

# Metadata extraction
iteration = openpmd_common.parse_iteration_number(filepath)
time_cu = openpmd_common.get_openpmd_time(series, iteration)

# Colormap setup
cmap = openpmd_common.setup_colormap("plasma", vmin=1e-10, vmax=1e3)

# Movie assembly
openpmd_common.movie_from_frames(frame_list, "output.mp4", fps=12)
```

### Key Classes

**`OpenPMDField`**: Wrapper around openPMD mesh component
```python
field = OpenPMDField(mesh, component_name)
x_coords = field.get_axis_coords(0)  # 1D array
full_array = field.read_full()       # Load all data
```

**`Canvas2D`**: Uniform 2D canvas for compositing
```python
canvas = Canvas2D(extent_xy=(xmin, xmax, ymin, ymax), nxny=1024)
canvas.add_patch(data_2d, x_coords, y_coords, method="nearest")
```

---

## Output Structure

Both scripts produce:

```
<out_dir>/
├── frame_*.png or slice_*.png    # Individual frame images (150 dpi, ~1-2 MB each)
├── <name>.mp4                    # Assembled movie (h.264, ~10-50 MB)
└── <name>.gif                    # Fallback if ffmpeg unavailable (~100-500 MB)
```

Movie frame rate: 12 fps by default (use `--fps` to change).

---

## Visualization Options

### Colormaps

Common choices (use with `--cmap`):
- `plasma` (default) – perceptually uniform, good for astrophysics
- `viridis` – colorblind-friendly
- `inferno` – high contrast
- `hot` – thermal-like
- `Greys` – monochrome

See [matplotlib colormaps](https://matplotlib.org/stable/tutorials/colors/colormaps.html).

### Normalization

By default, 1st and 99th percentiles are used for vmin/vmax. Override with:
```bash
--vmin 1e-10 --vmax 1e-3    # Log scale (automatic via LogNorm)
```

---

## Performance Notes

### 2D Planes
- Speed: ~1–2 sec per plane file
- Memory: Minimal (2D slabs only)
- Scaling: Linear with number of files

### 3D Slices
- Speed: ~5–10 sec per iteration (depends on nxny, AMR depth)
- Memory: O(nxny²) per axis (e.g., 1024² ≈ 4 GB per axis)
- Scaling: Quadratic with nxny; linear with number of iterations

**Tips for large data:**
- Reduce `--nxny` (e.g., 512 instead of 1024)
- Select only axes of interest (e.g., `--axes xy`)
- Use `--no-movie` if only frames needed

---

## Troubleshooting

### "openpmd_api not available"
```bash
pip install openPMD-api
```

### "No openPMD files found"
- Check data directory path (must contain .bp5, .bp, .bp4, or .h5 files)
- Ensure files are not in subdirectories (adjust if needed)

### "scipy not available" (for 3D with `--interpolate`)
```bash
pip install scipy
```

Falls back to nearest-neighbor without scipy, but with lower quality.

### Memory exhausted on large 3D files
- Reduce resolution: `--nxny 512`
- Process subsets of axes: `--axes xy` only
- Check if data can be pre-sliced with PlanesX instead

### Movie assembly fails
Ensure ffmpeg is available:
```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Or via pip
pip install imageio-ffmpeg
```

Fallback: Script creates GIF instead if MP4 fails.

---

## Tool Applicability: 2D vs 3D

See **`TOOLS_COMPARISON.md`** for detailed analysis of:
- Which components are shared vs unique
- When to use each script
- Performance/scalability trade-offs
- Known limitations and workarounds

**TL;DR**: ~70% of I/O and visualization code is shared; data extraction differs by dimension.

---

## Testing with Example Data

If you have CarpetX test simulations available:

```bash
# Test 2D planes
python plot_2d_planes.py ../CarpetX/TestPlanesX/test/ \
  --out-dir test_2d --fps 4 --no-movie

# Test 3D (if available)
python plot_3d_slices.py /path/to/3d/output/ \
  --axes xy --nxny 256 --no-movie  # Low res for quick test
```

---

## Development Notes

- **Architecture**: Split script I/O into a reusable library (`openpmd_common.py`)
- **Error handling**: Graceful degradation (skip bad files/axes, fall back from scipy)
- **Extensibility**: Easy to add new scripts (slicing methods, new file formats, etc.)

See `PROGRESS.md` for initial analysis and design decisions.

---

## License

Same as parent project.

## Contact

For issues or improvements, contact the project maintainers.
