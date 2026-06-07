# ADIOS2 Directory Format Guide

## What You Have

Your `.bp5` "files" are actually **ADIOS2 parallel directories**:

```
/lagoon/.../dev-trial7-long-plane/
├── parfile.xy_z_pos0000p000.it00000000.bp5/  ← DIRECTORY (512 processes)
│   ├── data.0          ← Process 0's data chunk
│   ├── data.1          ← Process 1's data chunk
│   ├── data.2
│   ├── ...
│   ├── data.511        ← Process 511's data chunk
│   ├── md.0            ← Metadata from process 0
│   ├── md.idx          ← Metadata index
│   ├── mmd.0           ← Master metadata (read this!)
│   └── profiling.json  ← Performance info
│
├── parfile.xy_z_pos0000p000.it00000512.bp5/  ← Next iteration
├── parfile.xz_y_pos0000p000.it00000000.bp5/  ← Different plane
└── ...
```

**Key point**: It's a directory, not a file. openpmd_api handles this transparently.

---

## How It Works

When you run:
```python
import openpmd_api as io
series = io.Series("/path/to/parfile.xy_z_pos0000p000.it00000000.bp5/", io.Access.read_only)
```

The openpmd_api:
1. ✓ Finds the `.bp5` directory
2. ✓ Reads `mmd.0` (master metadata)
3. ✓ Aggregates chunks from all `data.*` files
4. ✓ Presents data as if it were a single file

You don't need to do anything special—just pass the directory path!

---

## Usage With Visualization Scripts

### Method 1: Direct (Recommended)
```bash
# Pass the directory containing the .bp5 directories
python XScripts/plot_2d_planes.py /lagoon/michailchabanov/frontier/production_trial/dev-trial7-long-plane/
```

The scripts will find all `.bp5` directories and process them.

### Method 2: Explicit Path
```bash
# Or be explicit if you want only certain planes
python XScripts/plot_2d_planes.py /lagoon/michailchabanov/frontier/production_trial/dev-trial7-long-plane/ \
  --variable "xy"  # Filter to xy planes only (if needed)
```

---

## Diagnostic Commands

### Quick Scan
```bash
python XScripts/quick_diagnose.py /lagoon/michailchabanov/frontier/production_trial/dev-trial7-long-plane/
```

This now handles `.bp5` directories correctly and will:
- ✓ Find all 2D plane directories (xy, xz, yz)
- ✓ Count them
- ✓ Suggest visualization commands

### Inspect One Directory
```bash
python XScripts/show_mesh_names.py /lagoon/michailchabanov/frontier/production_trial/dev-trial7-long-plane/parfile.xy_z_pos0000p000.it00000000.bp5/
```

Pass the **directory path** (note the trailing slash matters for some tools).

---

## Performance Notes

**ADIOS2 directories are efficient:**
- ✓ Each process writes independently (no contention)
- ✓ Easy to read in parallel
- ✓ openpmd_api handles aggregation transparently
- ✓ You can read individual chunks if needed (advanced)

**Reading overhead:**
- First read: Slight overhead to aggregate 512 chunks
- Subsequent reads: Data cached, should be fast
- Total time: Usually <1 sec per plane file

---

## Common Issues & Solutions

### "Permission denied" reading a chunk
- Check: All `data.*` files readable by your user
- Fix: They should be readable if you own them

### "Only reading partial data"
- Check: All 512 chunks present (data.0 through data.511)
- Fix: If files are missing, the simulation might have crashed

### "openpmd_api says invalid file"
- Check: The directory name ends in `.bp5` (or `.bp4`, `.bp`, `.h5`)
- Fix: Some edge cases—try the full path with trailing `/`

---

## File Size Estimation

With 512 processes:
- Each `data.N` file: ~few MB to hundreds MB (depends on mesh size)
- Total `.bp5` directory: Size of 512 × single chunk
- Example: If each chunk is 10 MB, total is ~5 GB per iteration

**For visualization**: openpmd_api reads only what's needed, so memory usage is reasonable.

---

## Advanced: Reading Without openpmd_api

If you ever need raw access to chunks:
```bash
# List what's in a chunk
h5dump /path/to/.bp5/data.0  # If using HDF5 backend

# Or with ADIOS tools (if installed)
bpdump /path/to/.bp5/mmd.0
```

But you shouldn't need this—use the visualization scripts!

---

## Recommended Workflow

1. **Scan your data:**
   ```bash
   python XScripts/quick_diagnose.py /lagoon/.../dev-trial7-long-plane/
   ```

2. **Inspect a sample file:**
   ```bash
   python XScripts/show_mesh_names.py /lagoon/.../parfile.xy_z_pos0000p000.it00000000.bp5/
   ```

3. **Visualize:**
   ```bash
   python XScripts/plot_2d_planes.py /lagoon/.../dev-trial7-long-plane/ --out-dir frames
   ```

---

## TL;DR

- `.bp5` directories are normal (parallel ADIOS2 format)
- openpmd_api handles them transparently
- Pass the directory path to all tools
- Everything should "just work" ✓
