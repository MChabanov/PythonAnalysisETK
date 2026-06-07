#!/usr/bin/env python3
"""Check if a directory is compatible with plot_2d_planes.py."""

import sys
import os
import glob
import re

def check_directory(data_dir):
    """Verify a directory has the right files and structure."""
    data_dir = os.path.abspath(os.path.expanduser(data_dir))

    if not os.path.isdir(data_dir):
        print(f"✗ Not a directory: {data_dir}")
        return False

    print(f"Checking: {data_dir}\n")

    # Find openPMD files
    files = []
    for ext in ("bp5", "bp", "bp4", "h5"):
        pattern = os.path.join(data_dir, f"*.it*.{ext}")
        for f in glob.glob(pattern):
            if ".md." not in f and not f.endswith(".dir"):
                files.append(f)

    if not files:
        print("✗ No openPMD plane files found")
        print("  Expected files like: simulation.it00000000.bp5")
        return False

    files.sort()
    print(f"✓ Found {len(files)} plane file(s):")
    for f in files[:5]:
        print(f"  - {os.path.basename(f)}")
    if len(files) > 5:
        print(f"  ... and {len(files) - 5} more")

    # Check if openpmd_api is available
    try:
        import openpmd_api as io
        print("\n✓ openpmd_api available")
    except ImportError:
        print("\n✗ openpmd_api NOT available")
        print("  Install with: pip install openPMD-api")
        return False

    # Try to read first file and check mesh structure
    print("\nChecking first file structure...")
    try:
        import openpmd_api as io
        series = io.Series(files[0], io.Access.read_only)
        iterations = list(series.iterations)

        if not iterations:
            print("✗ No iterations in file")
            series.close()
            return False

        it = iterations[0]
        itobj = series.iterations[it]

        print(f"✓ Iteration {it} found")

        meshes = list(itobj.meshes)
        if not meshes:
            print("✗ No meshes in iteration")
            series.close()
            return False

        print(f"✓ Found {len(meshes)} mesh(es):")

        pattern = re.compile(r"(.+?)_lev(\d+)_patch(\d+)")
        valid_count = 0
        for mesh_name in meshes[:10]:
            m = pattern.match(mesh_name)
            if m:
                var = m.group(1)
                level = m.group(2)
                patch = m.group(3)
                print(f"  ✓ {mesh_name}")
                valid_count += 1
            else:
                print(f"  ⚠ {mesh_name} (doesn't match pattern)")

        if len(meshes) > 10:
            print(f"  ... and {len(meshes) - 10} more")

        if valid_count == 0:
            print("\n⚠ WARNING: No meshes match expected pattern <var>_lev<N>_patch<M>")
            print("  Script may still work, but compositing won't be available")

        series.close()

    except Exception as e:
        print(f"✗ Error reading file: {e}")
        return False

    print("\n" + "=" * 60)
    print("✓ Directory appears compatible with plot_2d_planes.py")
    print("=" * 60)
    print(f"\nUsage:")
    print(f"  python plot_2d_planes.py {data_dir}")
    print(f"  python plot_2d_planes.py {data_dir} --out-dir output_frames")
    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Check if a directory is compatible with plot_2d_planes.py")
        print()
        print("Usage: python check_plane_dir.py <data_dir>")
        print()
        print("Example: python check_plane_dir.py ../CarpetX/TestPlanesX/output/")
        sys.exit(1)

    data_dir = sys.argv[1]
    success = check_directory(data_dir)
    sys.exit(0 if success else 1)
