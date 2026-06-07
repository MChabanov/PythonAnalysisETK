#!/usr/bin/env python3
"""Diagnose ADIOS2 reading issues: check file structure and data integrity."""

import sys
import os
import struct
import numpy as np

DATA_DIR = "/lagoon/michailchabanov/frontier/production_trial/dev-trial7-long-plane/parfile.xy_z_pos0000p000.it00000000.bp5"

print("ADIOS2 Structure Diagnostic")
print("=" * 80)
print(f"Path: {DATA_DIR}\n")

# Check file structure
if not os.path.isdir(DATA_DIR):
    print(f"✗ Not a directory: {DATA_DIR}")
    sys.exit(1)

print("Directory contents:")
items = sorted(os.listdir(DATA_DIR))
total_size = 0
for item in items:
    path = os.path.join(DATA_DIR, item)
    if os.path.isfile(path):
        size = os.path.getsize(path)
        total_size += size
        size_mb = size / (1024**2)
        print(f"  {item:20s} {size_mb:10.2f} MB")

print(f"\n  Total: {total_size / (1024**3):.2f} GB\n")

# Try reading with openpmd_api
try:
    import openpmd_api as io
except ImportError:
    print("ERROR: openpmd_api not available")
    sys.exit(1)

print("Reading with openpmd_api:")
try:
    series = io.Series(DATA_DIR, io.Access.read_only)
    print("  ✓ Series opened successfully")

    it = list(series.iterations)[0]
    itobj = series.iterations[it]
    print(f"  ✓ Iteration {it} readable")

    # Check a simple mesh
    for mesh_name in sorted(itobj.meshes)[:1]:
        mesh = itobj.meshes[mesh_name]
        print(f"  ✓ First mesh: {mesh_name}")

        # Check metadata
        try:
            spacing = mesh.get_attribute("gridSpacing")
            offset = mesh.get_attribute("gridGlobalOffset")
            print(f"    Grid spacing: {spacing}")
            print(f"    Grid offset: {offset}")
        except:
            print("    (no grid metadata)")

        # Try reading a small chunk
        comp = list(mesh)[0]
        rc = mesh[comp]
        shape = tuple(int(s) for s in rc.shape)
        print(f"    Component: {comp}")
        print(f"    Shape: {shape}")

        # Read first few values
        rc.load_chunk()
        series.flush()
        arr = np.asarray(rc)

        print(f"    dtype: {arr.dtype}")
        print(f"    First 5 values: {arr.flat[:5]}")
        print(f"    Last 5 values: {arr.flat[-5:]}")
        print(f"    Min: {np.nanmin(arr):.6e}, Max: {np.nanmax(arr):.6e}")

        # Check if values look reasonable
        if np.nanmax(arr) > 1e100:
            print(f"\n    ⚠ WARNING: Max value is suspiciously large!")
            print(f"    This suggests data corruption or reading error")
            print(f"    Possible causes:")
            print(f"    - ADIOS2 parallel aggregation bug")
            print(f"    - Byte order mismatch (endianness)")
            print(f"    - Data type mismatch")
            print(f"    - Incomplete parallel file write")

except Exception as e:
    print(f"  ✗ Error: {e}")
    import traceback
    traceback.print_exc()

series.close()

# Check if individual data chunks are readable
print("\n" + "=" * 80)
print("Checking individual data chunks:")
print("=" * 80 + "\n")

data_files = sorted([f for f in os.listdir(DATA_DIR) if f.startswith('data.')])
if data_files:
    print(f"Found {len(data_files)} data chunk files: data.0 through data.{len(data_files)-1}\n")

    # Try to peek at one chunk
    sample_file = os.path.join(DATA_DIR, data_files[0])
    try:
        size = os.path.getsize(sample_file)
        print(f"Sample chunk (data.0):")
        print(f"  Size: {size / (1024**2):.2f} MB")

        # Read first 32 bytes as raw data
        with open(sample_file, 'rb') as f:
            first_bytes = f.read(32)
            print(f"  First 32 bytes (hex): {first_bytes.hex()}")

            # Try interpreting as float64
            f.seek(0)
            try:
                first_float = struct.unpack('<d', f.read(8))[0]  # Little-endian
                print(f"  First value (LE float64): {first_float:.6e}")
            except:
                pass

            # Try big-endian
            f.seek(0)
            try:
                first_float = struct.unpack('>d', f.read(8))[0]  # Big-endian
                print(f"  First value (BE float64): {first_float:.6e}")
            except:
                pass

    except Exception as e:
        print(f"  Error reading chunk: {e}")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)
print("""
If you see:
1. Very large max values (>1e100) → Data corruption or reading error
2. All levels have same max → Likely ADIOS2 aggregation issue
3. File size mismatch → Incomplete write or aggregation failure

NEXT STEPS:
- Check if SILO files are readable (they're also in the directory)
- Verify simulation didn't crash during output
- Check CarpetX/PlanesX output logs
- Consider re-running the simulation or using a different checkpoint
""")
