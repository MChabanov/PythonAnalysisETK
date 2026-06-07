#!/usr/bin/env python3
"""Inspect raw data content: count points and show statistics per level."""

import sys
import os
import time
import numpy as np

# Configuration
DATA_DIR = "/lagoon/michailchabanov/frontier/production_trial/dev-trial7-long-plane/parfile.xy_z_pos0000p000.it00000000.bp5"
VARIABLE = "hydrobasex_rho"

print("Raw Data Inspector")
print("=" * 80)
print(f"File: {os.path.basename(DATA_DIR)}")
print(f"Variable: {VARIABLE}")
print("=" * 80 + "\n")

# Read with openpmd_api
try:
    import openpmd_api as io
except ImportError:
    print("ERROR: openpmd_api not available")
    sys.exit(1)

try:
    print("Opening file...", end=" ", flush=True)
    t0 = time.time()
    series = io.Series(DATA_DIR, io.Access.read_only)
    print(f"({time.time()-t0:.2f}s)")
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)

it = list(series.iterations)[0]
itobj = series.iterations[it]

print(f"Iteration: {it}\n")

# Find all levels for this variable
print(f"Scanning meshes for '{VARIABLE}'...\n")

data_per_level = {}

for mesh_name in sorted(itobj.meshes):
    if VARIABLE not in mesh_name:
        continue

    # Extract level
    import re
    m = re.search(r"lev(\d+)", mesh_name)
    if not m:
        continue
    level = int(m.group(1))

    try:
        t_start = time.time()
        mesh = itobj.meshes[mesh_name]
        comp_name = list(mesh)[0]
        rc = mesh[comp_name]

        # Get shape WITHOUT loading data
        shape = tuple(int(s) for s in rc.shape)
        num_points = np.prod(shape)

        # Now load the data
        rc.load_chunk()
        series.flush()
        arr = np.asarray(rc)

        # Statistics
        valid = np.isfinite(arr)
        num_valid = np.sum(valid)
        num_invalid = np.sum(~valid)

        if num_valid > 0:
            data_min = np.min(arr[valid])
            data_max = np.max(arr[valid])
            data_mean = np.mean(arr[valid])
        else:
            data_min = data_max = data_mean = np.nan

        t_elapsed = time.time() - t_start

        data_per_level[level] = {
            'shape': shape,
            'num_points': num_points,
            'num_valid': num_valid,
            'num_invalid': num_invalid,
            'data_min': data_min,
            'data_max': data_max,
            'data_mean': data_mean,
            'time': t_elapsed,
        }

        print(f"Level {level}:")
        print(f"  Mesh name: {mesh_name}")
        print(f"  Shape: {shape}")
        print(f"  Total points: {num_points:,}")
        print(f"  Valid points: {num_valid:,} ({100*num_valid/num_points:.1f}%)")
        print(f"  Invalid/NaN: {num_invalid:,} ({100*num_invalid/num_points:.1f}%)")
        print(f"  Data range: [{data_min:.6e}, {data_max:.6e}]")
        print(f"  Data mean: {data_mean:.6e}")
        print(f"  Read time: {t_elapsed:.3f}s")
        print()

    except Exception as e:
        print(f"Level {level}: ERROR - {e}\n")
        continue

series.close()

# Summary
print("=" * 80)
print("SUMMARY")
print("=" * 80)

if data_per_level:
    levels = sorted(data_per_level.keys())
    print(f"\nLevels found: {levels}")
    print(f"Total levels: {len(levels)}\n")

    # Find any problematic data
    print("Data Quality Check:")
    for level in levels:
        info = data_per_level[level]
        pct_valid = 100 * info['num_valid'] / info['num_points']
        pct_invalid = 100 * info['num_invalid'] / info['num_points']

        status = "✓" if pct_valid > 95 else "⚠"
        print(f"  {status} Level {level}: {pct_valid:.1f}% valid, {pct_invalid:.1f}% NaN/inf")

        # Check for unreasonably large values
        if info['data_max'] > 1e10:
            print(f"      WARNING: Maximum value {info['data_max']:.3e} seems very large!")
        if info['data_min'] < -1e10:
            print(f"      WARNING: Minimum value {info['data_min']:.3e} seems very negative!")

    print(f"\nTotal read time: {sum(d['time'] for d in data_per_level.values()):.2f}s")
else:
    print("ERROR: No data found for this variable")
    sys.exit(1)
