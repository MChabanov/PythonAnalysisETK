#!/usr/bin/env python3
"""Quick test: plot a single variable from one plane file with timing."""

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

overall_timer = Timer()
overall_timer.__enter__()

# Configuration
DATA_DIR = "/lagoon/michailchabanov/frontier/production_trial/dev-trial7-long-plane"
PLANE_TYPE = "xy_z_pos0000p000"  # Which plane to plot
ITERATION = "it00000000"  # Which snapshot
VARIABLE = "hydrobasex_rho"  # Which variable to plot
OUTPUT_FILE = "test_rho_xy_z.png"

print(f"Test Plot: Single Variable Visualization")
print("=" * 80)
print(f"Directory:   {DATA_DIR}")
print(f"Plane:       {PLANE_TYPE}")
print(f"Snapshot:    {ITERATION}")
print(f"Variable:    {VARIABLE}")
print(f"Output:      {OUTPUT_FILE}")
print("=" * 80 + "\n")

# Setup matplotlib
with Timer("Setup matplotlib"):
    opc.setup_matplotlib_style()

# Find the file
with Timer("Find file") as t_find:
    import glob
    pattern = os.path.join(DATA_DIR, f"{PLANE_TYPE}.{ITERATION}.bp5")
    files = glob.glob(pattern)

if not files:
    print(f"✗ File not found matching: {pattern}")
    sys.exit(1)

filepath = files[0]
print(f"Found file: {os.path.basename(filepath)}\n")

# Read with openpmd_api
try:
    import openpmd_api as io
except ImportError:
    print("ERROR: openpmd_api not available")
    sys.exit(1)

with Timer("Open openPMD file") as t_open:
    try:
        series = io.Series(filepath, io.Access.read_only)
    except Exception as e:
        print(f"ERROR opening file: {e}")
        sys.exit(1)

with Timer("Read iteration metadata"):
    it = list(series.iterations)[0]
    itobj = series.iterations[it]
    time_cu = opc.get_openpmd_time(series, it)

print(f"Iteration: {it}")
if time_cu:
    print(f"Time: {time_cu:.6g} code units\n")

# Find meshes matching the variable
print(f"Looking for variable: {VARIABLE}")
print(f"Available meshes containing '{VARIABLE}':")

with Timer("Scan meshes"):
    target_meshes = []
    for mesh_name in sorted(itobj.meshes):
        if VARIABLE in mesh_name:
            mesh = itobj.meshes[mesh_name]
            components = list(mesh)
            print(f"  ✓ {mesh_name}")
            print(f"    Components: {components}")
            target_meshes.append((mesh_name, mesh, components))

if not target_meshes:
    print(f"✗ No meshes found containing '{VARIABLE}'")
    series.close()
    sys.exit(1)

print(f"\nFound {len(target_meshes)} mesh(es)\n")

# Composite all levels for this variable
print("Reading AMR levels...")
data_by_level = {}

with Timer("Read all mesh levels"):
    for i, (mesh_name, mesh, components) in enumerate(target_meshes):
        # Extract level from name
        import re
        m = re.search(r"lev(\d+)", mesh_name)
        if not m:
            continue
        level = int(m.group(1))

        # Read first component (usually scalar)
        comp = components[0]
        try:
            with Timer(f"  → Read level {level}"):
                field = opc.OpenPMDField(mesh, comp)
                arr = field.read_full()
                x = field.get_axis_coords(0)
                y = field.get_axis_coords(1)
                data_by_level[level] = (arr, x, y)
                print(f"    Shape: {arr.shape}")
        except Exception as e:
            print(f"  ✗ Level {level}: {e}")

if not data_by_level:
    print("✗ No data loaded")
    series.close()
    sys.exit(1)

# Simple composite: use finest level data
finest_level = max(data_by_level.keys())
arr, x, y = data_by_level[finest_level]
print(f"\nUsing level {finest_level} (finest available)")
print(f"Data shape: {arr.shape}")
print(f"Data range: {np.nanmin(arr):.6e} to {np.nanmax(arr):.6e}")

# Plot
with Timer("Create figure"):
    fig, ax = plt.subplots(figsize=(10, 9), constrained_layout=True)

with Timer("Compute colorscale"):
    extent = (x.min(), x.max(), y.min(), y.max())
    vmin = np.nanpercentile(arr, 1)
    vmax = np.nanpercentile(arr, 99)
    print(f"    vmin={vmin:.6e}, vmax={vmax:.6e}")

with Timer("Draw image"):
    norm = LogNorm(vmin=vmin, vmax=vmax)
    cmap = opc.setup_colormap("plasma")
    im = ax.imshow(arr, origin="lower", extent=extent, norm=norm, cmap=cmap, interpolation="none")

with Timer("Add labels and colorbar"):
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    title = f"{VARIABLE} (level {finest_level})"
    if time_cu:
        title += f", t={time_cu:.6g} CU"
    ax.set_title(title)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(VARIABLE)

with Timer("Save PNG"):
    plt.savefig(OUTPUT_FILE, bbox_inches="tight", dpi=150)
    print(f"    Saved: {OUTPUT_FILE}")

plt.close()
series.close()

# Final summary
overall_timer.__exit__(None, None, None)
print("\n" + "=" * 80)
print(f"✓ Complete in {overall_timer.elapsed:.2f}s")
print("=" * 80)
