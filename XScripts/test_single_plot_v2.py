#!/usr/bin/env python3
"""Test: Plot 2D plane with user-specified resampling and linear interpolation."""

import sys
import os
import time
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

import openpmd_common as opc

# Timing utilities
class Timer:
    def __init__(self, name=""):
        self.name = name
        self.start_time = None
        self.elapsed = 0.0

    def __enter__(self):
        self.start_time = time.time()
        if self.name:
            print(f"⏱ {self.name}...", end=" ", flush=True)
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start_time
        if self.name:
            print(f"({self.elapsed:.3f}s)")

# Configuration
DATA_DIR = "/lagoon/michailchabanov/frontier/production_trial/dev-trial7-long-plane/parfile.xy_z_pos0000p000.it00000000.bp5"
VARIABLE = "hydrobasex_rho"  # Which variable to plot
OUTPUT_RESOLUTION = 1024  # Resample to N x N grid
OUTPUT_FILE = "test_rho_xy_z_resampled.png"

print(f"Test Plot v2: 2D Plane with Resampling")
print("=" * 80)
print(f"File:        {os.path.basename(DATA_DIR)}")
print(f"Variable:    {VARIABLE}")
print(f"Resolution:  {OUTPUT_RESOLUTION} × {OUTPUT_RESOLUTION}")
print(f"Output:      {OUTPUT_FILE}")
print("=" * 80 + "\n")

overall_timer = Timer()
overall_timer.__enter__()

# Setup matplotlib
with Timer("Setup matplotlib"):
    opc.setup_matplotlib_style()

# Verify the file/directory exists
with Timer("Verify path"):
    filepath = DATA_DIR
    if not (os.path.isfile(filepath) or os.path.isdir(filepath)):
        print(f"✗ Path not found: {filepath}")
        sys.exit(1)
    print(f"Found: {os.path.basename(filepath)}")

# Read with openpmd_api
try:
    import openpmd_api as io
except ImportError:
    print("ERROR: openpmd_api not available")
    sys.exit(1)

with Timer("Open openPMD file"):
    try:
        series = io.Series(filepath, io.Access.read_only)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

with Timer("Read iteration metadata"):
    it = list(series.iterations)[0]
    itobj = series.iterations[it]
    time_cu = opc.get_openpmd_time(series, it)

print(f"Iteration: {it}")
if time_cu:
    print(f"Time: {time_cu:.6g} code units\n")

# Find meshes for this variable
print(f"Looking for variable: {VARIABLE}")
target_meshes = {}

with Timer("Scan meshes"):
    for mesh_name in sorted(itobj.meshes):
        if VARIABLE in mesh_name:
            import re
            m = re.search(r"lev(\d+)", mesh_name)
            if m:
                level = int(m.group(1))
                mesh = itobj.meshes[mesh_name]
                components = list(mesh)
                if components:
                    target_meshes[level] = (mesh_name, mesh, components[0])

if not target_meshes:
    print(f"✗ No meshes found for {VARIABLE}")
    sys.exit(1)

print(f"Found {len(target_meshes)} level(s): {sorted(target_meshes.keys())}\n")

# Create uniform canvas and composite all levels
print(f"Creating uniform {OUTPUT_RESOLUTION}×{OUTPUT_RESOLUTION} grid...")

# First pass: determine extent from actual chunks in coarsest level
with Timer("Determine extent from chunks"):
    coarse_level = min(target_meshes.keys())
    _, coarse_mesh, coarse_comp = target_meshes[coarse_level]
    coarse_field = opc.OpenPMDField(coarse_mesh, coarse_comp)

    # Find actual extent from chunks (not declared extent)
    ggo = coarse_field.offset   # grid global offset
    gsp = coarse_field.spacing  # grid spacing
    pos = coarse_field.position # cell position

    x_min, x_max = np.inf, -np.inf
    y_min, y_max = np.inf, -np.inf

    for ch in coarse_field.record.available_chunks():
        off = np.array([int(v) for v in ch.offset])
        ext = np.array([int(v) for v in ch.extent])

        # World coordinates for this chunk
        y0 = ggo[0] + (off[0] + pos[0]) * gsp[0]
        x0 = ggo[1] + (off[1] + pos[1]) * gsp[1]
        y1 = y0 + ext[0] * gsp[0]
        x1 = x0 + ext[1] * gsp[1]

        x_min = min(x_min, x0)
        x_max = max(x_max, x1)
        y_min = min(y_min, y0)
        y_max = max(y_max, y1)

    if x_min == np.inf:
        print("ERROR: No chunks found in coarse level")
        sys.exit(1)

    print(f"  Actual extent: x=[{x_min:.6e}, {x_max:.6e}]")
    print(f"                 y=[{y_min:.6e}, {y_max:.6e}]")

# Create uniform canvas
canvas = np.full((OUTPUT_RESOLUTION, OUTPUT_RESOLUTION), np.nan, dtype=np.float64)
grid_x = np.linspace(x_min, x_max, OUTPUT_RESOLUTION)
grid_y = np.linspace(y_min, y_max, OUTPUT_RESOLUTION)

# Composite levels coarse → fine with per-chunk interpolation
print("\nCompositing levels with linear interpolation (per-chunk)...")

with Timer("Composite all levels"):
    for level in sorted(target_meshes.keys()):
        with Timer(f"  Level {level}"):
            mesh_name, mesh, comp = target_meshes[level]
            field = opc.OpenPMDField(mesh, comp)

            ggo = field.offset
            gsp = field.spacing
            pos = field.position

            chunk_count = 0
            for data, off, ext in field.read_chunks(series):
                chunk_count += 1

                # World coordinates for this chunk
                y0 = ggo[0] + (off[0] + pos[0]) * gsp[0]
                x0 = ggo[1] + (off[1] + pos[1]) * gsp[1]

                # Build per-chunk coordinate arrays
                y_chunk = y0 + np.arange(ext[0]) * gsp[0]
                x_chunk = x0 + np.arange(ext[1]) * gsp[1]

                # Interpolate chunk to uniform canvas using scipy
                try:
                    from scipy.interpolate import RegularGridInterpolator
                    # Create interpolator for this chunk
                    valid = np.where(np.isfinite(data), data, np.nan)
                    interp = RegularGridInterpolator((y_chunk, x_chunk), valid,
                                                   bounds_error=False, fill_value=np.nan)
                    # Evaluate on uniform grid
                    YY, XX = np.meshgrid(grid_y, grid_x, indexing="ij")
                    points = np.stack([YY, XX], axis=-1)
                    interpolated = interp(points)

                    # Mask for valid data
                    valid_mask = np.isfinite(interpolated)

                    # Fine levels overwrite coarse
                    canvas[valid_mask] = interpolated[valid_mask]

                except ImportError:
                    # Fallback: nearest neighbor per chunk
                    jj = np.searchsorted(y_chunk, grid_y)
                    ii = np.searchsorted(x_chunk, grid_x)
                    jj = np.clip(jj, 0, len(y_chunk)-1)
                    ii = np.clip(ii, 0, len(x_chunk)-1)
                    sub = data[np.ix_(jj, ii)]
                    valid_mask = np.isfinite(sub)
                    canvas[valid_mask] = sub[valid_mask]

            print(f"    Processed {chunk_count} chunk(s)")

# Plot
print("\nCreating visualization...")

with Timer("Create figure"):
    fig, ax = plt.subplots(figsize=(10, 9), constrained_layout=True)

with Timer("Compute colorscale"):
    extent = (grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max())
    vmin = np.nanpercentile(canvas, 1)
    vmax = np.nanpercentile(canvas, 99)
    print(f"  vmin={vmin:.3e}, vmax={vmax:.3e}")
    print(f"  NaN pixels: {np.isnan(canvas).sum()} / {canvas.size}")

with Timer("Draw image"):
    norm = LogNorm(vmin=vmin, vmax=vmax)
    cmap = opc.setup_colormap("plasma")
    im = ax.imshow(canvas, origin="lower", extent=extent, norm=norm, cmap=cmap, interpolation="none")

with Timer("Add labels"):
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    title = f"{VARIABLE} ({OUTPUT_RESOLUTION}×{OUTPUT_RESOLUTION})"
    if time_cu:
        title += f", t={time_cu:.3e} CU"
    ax.set_title(title)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(VARIABLE)

with Timer("Save PNG"):
    plt.savefig(OUTPUT_FILE, bbox_inches="tight", dpi=150)
    print(f"  Saved: {OUTPUT_FILE}")

plt.close()
series.close()

# Summary
overall_timer.__exit__(None, None, None)
print("\n" + "=" * 80)
print(f"✓ Complete in {overall_timer.elapsed:.2f}s")
print(f"✓ Resampled {len(target_meshes)} levels to {OUTPUT_RESOLUTION}×{OUTPUT_RESOLUTION}")
print("=" * 80)
