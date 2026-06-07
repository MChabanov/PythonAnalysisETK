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

        # Test multiple patterns
        patterns = [
            ("Format 1", r"(.+?)_lev(\d+)_patch(\d+)"),
            ("Format 2", r"(.+?)_patch(\d+)_lev(\d+)"),
            ("Format 3", r"(.+?)_lev\d+.*_patch\d+"),
        ]

        matched_names = set()
        unmatched_names = []

        for mesh_name in meshes:
            matched = False
            for fmt_name, pattern_str in patterns:
                if re.search(pattern_str, mesh_name):
                    matched_names.add(mesh_name)
                    matched = True
                    break
            if not matched:
                unmatched_names.append(mesh_name)

        # Show sample of recognized
        print("\n  Recognized meshes (first 5):")
        for mesh_name in sorted(matched_names)[:5]:
            print(f"    ✓ {mesh_name}")

        # Show sample of unrecognized (THIS IS THE KEY INFO)
        print(f"\n  ⚠ UNRECOGNIZED: {len(unmatched_names)}/{len(meshes)} meshes")
        print("\n  Sample unrecognized mesh names (first 10):")
        for mesh_name in unmatched_names[:10]:
            print(f"    - {mesh_name}")

        if len(unmatched_names) > 10:
            print(f"    ... and {len(unmatched_names) - 10} more")

        if len(matched_names) < len(meshes) * 0.5:
            print("\n⚠ MAJOR ISSUE: Most meshes don't match expected naming patterns!")
            print("\nYour mesh names don't follow standard PlanesX format.")
            print("This is EXPECTED if you have component data.")
            print("\nFix: Look at the sample names above and run:")
            print(f"  python XScripts/show_mesh_names.py /path/to/file.bp5")
            print("  to see ALL mesh names and understand the structure")

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
