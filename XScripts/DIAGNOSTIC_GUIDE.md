# Data Diagnostic Guide

Three tools to understand your openPMD data structure:

## 1. Full Directory Diagnostic (Start Here!)

**Best for:** First-time setup, understanding what data you have

```bash
python XScripts/diagnose_data.py /path/to/simulations/
```

This does everything:
- ✓ Scans entire directory tree for openPMD files
- ✓ Classifies as 2D (planes) or 3D (volumes)
- ✓ Inspects all mesh structures
- ✓ Reports compatibility with visualization scripts
- ✓ Suggests next steps

**Example output:**
```
2D DATA (5 files)
  BNS_IG.xy_z_pos0000p000.it00000000.bp5
    Category: 2D (PlanesX)
    Iteration: 0
    Meshes: 140
    Patterns found:
      • Vector components: 105 (75%)
      • AMR (lev_patch): 35 (25%)
    ⚠ Compatible with visualization: 35/140 (25%)
    Sample mesh names:
      - rho_lev0_patch0_x
      - rho_lev0_patch0_y
      - rho_lev0_patch0_z

3D DATA (2 files)
  BNS_IG.it00000000.bp5
    Category: 3D (AMR structure)
    Iteration: 0
    Meshes: 12
    Patterns found:
      • AMR (lev_patch): 12 (100%)
    ✓ Compatible with visualization: 12/12 (100%)

SUMMARY
Total readable files: 7
Total meshes found: 284

Data types available:
  ✓ 2D planes
  ✓ 3D volumes

You can use BOTH visualization scripts:
  - python XScripts/plot_2d_planes.py <2d_dir>
  - python XScripts/plot_3d_slices.py <3d_dir>
```

---

## 2. Single File Mesh Inspection

**Best for:** Understanding a specific file's structure

```bash
python XScripts/show_mesh_names.py /path/to/file.bp5
```

Shows:
- All mesh names
- Components in each mesh
- Data shapes
- Naming patterns found
- Recommendations

**Use when:** Diagnostic says "Not compatible", need to see actual mesh names

---

## 3. Quick Directory Check

**Best for:** Quick sanity check if a directory is ready to visualize

```bash
python XScripts/check_plane_dir.py /path/to/data/
```

Shows:
- ✓ Files discovered
- ✓ openpmd_api available
- ✓ Sample mesh names
- ✓ Compatibility percentage

---

## Typical Workflow

### Step 1: Understand Your Data
```bash
python XScripts/diagnose_data.py /path/to/simulations/
```
This tells you:
- What 2D/3D files you have
- Which are compatible
- What needs fixing

### Step 2: If "Not Compatible", Inspect Details
```bash
python XScripts/show_mesh_names.py /path/to/problematic/file.bp5
```
This shows:
- Actual mesh names
- Patterns found
- How to fix it

### Step 3: Visualize
```bash
# For 2D planes
python XScripts/plot_2d_planes.py /path/to/2d_data/ --out-dir output

# For 3D volumes
python XScripts/plot_3d_slices.py /path/to/3d_data/ --out-dir output
```

---

## Common Diagnostic Results

### ✓ "100% compatible"
**What:** All mesh names match expected patterns  
**Action:** Ready to visualize immediately
```bash
python XScripts/plot_2d_planes.py <dir> --out-dir output
```

### ⚠ "25-75% compatible"
**What:** Some meshes recognized, others not  
**Likely cause:** Vector components (rho_x, rho_y, rho_z)  
**Action:** Still works! Script groups them automatically
```bash
python XScripts/plot_2d_planes.py <dir> --out-dir output
```

### ✗ "0% compatible"
**What:** Mesh names completely unrecognized  
**Likely causes:**
- Completely different naming scheme
- Custom writer format
- Data that's not standard PlanesX/CarpetX  

**Action:** Run detailed inspection
```bash
python XScripts/show_mesh_names.py <file.bp5>
# Share output so we can add support
```

---

## Mesh Name Patterns

### Expected (Good)
```
rho_lev0_patch0              ← Standard AMR
pressure_lev1_patch2         ← Different level/patch
hydrobasex_rho_patch0_lev0   ← Alternate format
```

### Partially Expected (Still OK)
```
rho_lev0_patch0_x            ← With vector component
rho_lev0_patch0_y
rho_lev0_patch0_z
```

### Unexpected (Needs Investigation)
```
mesh_0_0_0                   ← Numerical only
field_12345                  ← Flat naming
component_scalar             ← Generic names
```

---

## If You Find Incompatibilities

1. **Run detailed inspection:**
   ```bash
   python XScripts/show_mesh_names.py <file.bp5>
   ```

2. **Share the output** so we can:
   - Add a new mesh name pattern
   - Update the script to handle your data
   - Provide custom configuration

3. **Or manually specify** mesh mappings in the script

---

## File Organization Best Practices

For visualization to work best:

**Good:**
```
simulations/
├── 2d_planes/
│   ├── sim.xy_z_pos0.it00000.bp5
│   ├── sim.xy_z_pos0.it00100.bp5
│   └── ...
└── 3d_data/
    ├── sim.it00000.bp5
    ├── sim.it00100.bp5
    └── ...
```

**Also OK:**
```
simulations/
├── sim.xy_z_pos0.it00000.bp5
├── sim.xy_z_pos0.it00100.bp5
├── sim.it00000.bp5
├── sim.it00100.bp5
└── ...
```

The diagnostic will find them either way!

---

## Token Usage Note

- `diagnose_data.py`: Quick scan, ~1% token per 100 files
- `show_mesh_names.py`: Single file, <1% token
- `check_plane_dir.py`: Single directory, <1% token

All safe to run repeatedly without token concerns.
