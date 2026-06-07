# Update Summary: Per-Chunk Reading Logic Applied

## Scripts Updated

### 1. **test_single_plot_v2.py** (MAJOR UPDATE)
Migrated from declared-extent reading to per-chunk safe reading.

#### Before
```python
# ✗ Used full declared extent
x_coarse = coarse_field.get_axis_coords(0)  # Assumes all cells written
y_coarse = coarse_field.get_axis_coords(1)
arr = field.read_full()  # Old: loaded full extent as one blob
```

#### After
```python
# ✓ Uses actual chunks
for ch in coarse_field.record.available_chunks():
    off = np.array([int(v) for v in ch.offset])
    ext = np.array([int(v) for v in ch.extent])
    # Compute world coordinates from chunk position
    y0 = ggo[0] + (off[0] + pos[0]) * gsp[0]
    x0 = ggo[1] + (off[1] + pos[1]) * gsp[1]
    # Track actual extent bounds
    x_min = min(x_min, x0); x_max = max(x_max, x0 + ext[1]*gsp[1])

# Per-chunk interpolation
for data, off, ext in field.read_chunks(series):
    y_chunk = y0 + np.arange(ext[0]) * gsp[0]
    x_chunk = x0 + np.arange(ext[1]) * gsp[1]
    # Interpolate chunk with correct coordinates
    interp = RegularGridInterpolator((y_chunk, x_chunk), ...)
```

#### Benefits
1. **Correct extent detection** — Uses actual written chunks, not declared extent
2. **Per-chunk interpolation** — Each chunk has correct coordinate system
3. **Sparse AMR safe** — Doesn't load unwritten padding as garbage
4. **Memory efficient** — Iterates chunks instead of assembling full array

---

### 2. **inspect_chunks.py** (NEW - Reference Implementation)
Diagnostic tool showing data structure using per-chunk reading.

**Usage:**
```bash
python inspect_chunks.py <filepath> [variable_pattern]
python inspect_chunks.py sim.xy_z0.it%08T.bp5 rho
```

**Output:**
- Declared extent vs. actual written points
- Sparsity percentage
- Per-chunk layout (offset, extent, point count)

---

### 3. **openpmd_common.py**
Added `read_chunks()` generator to `OpenPMDField` class.

```python
def read_chunks(self, series=None):
    """Yield (data, offset, extent) for each written chunk (ADIOS2/HDF5 safe)."""
    for ch in self.record.available_chunks():
        # ... per-chunk loading logic ...
        yield data, off, ext
```

Updated `read_full()` to use chunks internally for safe assembly.

---

### 4. **plot_2d_planes.py**
Updated `read_plane_file()` to use per-chunk iteration with world coordinates.

```python
for data, off, ext in field.read_chunks(series):
    # Compute world-coordinate edges
    nb, na = ext[0], ext[1]
    y0 = ggo[0] + (off[0] + pos[0]) * gsp[0]
    x0 = ggo[1] + (off[1] + pos[1]) * gsp[1]
    # Build coordinate arrays
    y_coords = y0 + np.arange(nb) * gsp[0]
    x_coords = x0 + np.arange(na) * gsp[1]
```

---

### 5. **plot_3d_slices.py**
Updated `read_full()` internally (uses chunks now); still works for slicing.

---

## Key Principle

**Problem**: Declared extent is the full refined domain, but only AMR boxes are written.
**Solution**: Use `available_chunks()` to load only written regions with correct world coordinates.

## Testing Recommendation

Run diagnostic first to understand data structure:
```bash
python inspect_chunks.py /path/to/data.bp5 hydrobasex_rho
```

Then use plotting scripts with confidence that they handle sparse AMR correctly.
