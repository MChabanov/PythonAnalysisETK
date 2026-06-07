#!/usr/bin/env python3
"""Quick diagnostic: directly scan a single directory for openPMD files."""

import sys
import os
import glob
import re
from collections import defaultdict


def quick_diagnose(data_dir):
    """Fast scan of a single directory."""
    data_dir = os.path.abspath(os.path.expanduser(data_dir))

    if not os.path.isdir(data_dir):
        print(f"ERROR: Not a directory: {data_dir}")
        return 1

    print(f"Quick Diagnostic: {data_dir}\n")

    # Find openPMD files with simple glob
    bp_files = glob.glob(os.path.join(data_dir, "*.bp5")) + \
               glob.glob(os.path.join(data_dir, "*.bp4")) + \
               glob.glob(os.path.join(data_dir, "*.bp")) + \
               glob.glob(os.path.join(data_dir, "*.h5"))

    # Filter out metadata and lock files
    bp_files = [f for f in bp_files if ".md." not in f and not f.endswith(".dir")]
    bp_files = sorted(bp_files)

    if not bp_files:
        print(f"✗ No openPMD files found directly in {data_dir}")
        print("\nLooking for any .bp* or .h5 files...")
        all_files = os.listdir(data_dir)
        bp_like = [f for f in all_files if '.bp' in f.lower() or '.h5' in f.lower()]
        if bp_like:
            print(f"Found {len(bp_like)} .bp*/.h5-like files:")
            for f in sorted(bp_like)[:10]:
                print(f"  - {f}")
            if len(bp_like) > 10:
                print(f"  ... and {len(bp_like) - 10} more")
        return 1

    print(f"Found {len(bp_files)} openPMD files\n")

    # Classify by type
    by_type = defaultdict(list)
    for f in bp_files:
        basename = os.path.basename(f)
        if re.search(r"(xy|xz|yz).*_z|_y|_x.*pos", basename) or \
           re.search(r"\.(xy|xz|yz)_", basename) or \
           "planes" in basename.lower() or \
           re.search(r"_pos\d+", basename):
            by_type["2D"].append(f)
        else:
            by_type["3D"].append(f)

    print("=" * 80)
    if by_type["2D"]:
        print(f"2D PLANE FILES ({len(by_type['2D'])})")
        print("-" * 80)
        # Group by plane type
        planes = defaultdict(list)
        for f in by_type["2D"]:
            basename = os.path.basename(f)
            # Extract plane type (xy, xz, yz)
            m = re.search(r"\.(xy|xz|yz)_", basename)
            plane = m.group(1) if m else "unknown"
            planes[plane].append(basename)

        for plane in sorted(planes.keys()):
            files = planes[plane]
            print(f"\n  {plane.upper()} plane: {len(files)} file(s)")
            for f in sorted(files)[:3]:
                print(f"    - {f}")
            if len(files) > 3:
                print(f"    ... and {len(files) - 3} more")

    if by_type["3D"]:
        print(f"\n3D VOLUME FILES ({len(by_type['3D'])})")
        print("-" * 80)
        for f in sorted(by_type["3D"])[:5]:
            print(f"  - {os.path.basename(f)}")
        if len(by_type["3D"]) > 5:
            print(f"  ... and {len(by_type['3D']) - 5} more")

    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)

    if by_type["2D"]:
        print("\n✓ You have 2D plane files. Inspect one to see mesh structure:")
        sample_2d = by_type["2D"][0]
        print(f"  python XScripts/show_mesh_names.py '{sample_2d}'")

    if by_type["3D"]:
        print("\n✓ You have 3D volume files. Inspect one to see mesh structure:")
        sample_3d = by_type["3D"][0]
        print(f"  python XScripts/show_mesh_names.py '{sample_3d}'")

    if by_type["2D"] and by_type["3D"]:
        print("\n✓ You can use BOTH visualization scripts:")
        print(f"  python XScripts/plot_2d_planes.py '{os.path.dirname(by_type['2D'][0])}'")
        print(f"  python XScripts/plot_3d_slices.py '{os.path.dirname(by_type['3D'][0])}'")

    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Quick diagnostic for openPMD files in a directory")
        print()
        print("Usage: python quick_diagnose.py <directory>")
        print()
        print("Example: python quick_diagnose.py /lagoon/michailchabanov/frontier/...")
        sys.exit(1)

    data_dir = sys.argv[1]
    sys.exit(quick_diagnose(data_dir))
