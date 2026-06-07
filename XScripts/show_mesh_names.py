#!/usr/bin/env python3
"""Show all mesh names and structure in an openPMD file."""

import sys
import os

def show_meshes(filepath):
    """Display all meshes and components in a file."""
    try:
        import openpmd_api as io
    except ImportError:
        print("ERROR: openpmd_api not available; install with: pip install openPMD-api")
        return False

    filepath = os.path.abspath(os.path.expanduser(filepath))
    if not os.path.isfile(filepath):
        print(f"ERROR: File not found: {filepath}")
        return False

    print(f"Reading: {filepath}\n")

    try:
        series = io.Series(filepath, io.Access.read_only)
    except Exception as e:
        print(f"ERROR opening file: {e}")
        return False

    iterations = list(series.iterations)
    if not iterations:
        print("No iterations in file")
        series.close()
        return False

    it = iterations[0]
    itobj = series.iterations[it]

    meshes = list(itobj.meshes)
    print(f"Iteration: {it}")
    print(f"Total meshes: {len(meshes)}\n")

    if len(meshes) > 100:
        print("=" * 80)
        print(f"Too many meshes ({len(meshes)}) to display all.")
        print("=" * 80)

    print("Mesh names and components:")
    print("=" * 80)

    for i, mesh_name in enumerate(meshes):
        mesh = itobj.meshes[mesh_name]
        components = list(mesh)
        shape = mesh[components[0]].shape if components else "?"

        # Truncate long names
        display_name = mesh_name if len(mesh_name) < 70 else mesh_name[:67] + "..."
        print(f"{i+1:4d}. {display_name}")
        print(f"       Components: {components}")
        if hasattr(mesh[components[0]], 'shape'):
            print(f"       Shape: {shape}")
        print()

    series.close()

    # Try to find pattern
    print("=" * 80)
    print("Analyzing naming patterns...\n")

    import re
    patterns = {
        "_lev.*_patch": "Contains _lev and _patch (good for compositing)",
        "gf[0-9]": "Grid function names (gf0, gf1, etc.)",
        "_x$|_y$|_z$": "Component suffixes (x, y, z)",
        "_[xy][z]$|_[yz]$": "Plane/slice designations",
        "_lev": "Contains _lev (AMR level info)",
    }

    found_patterns = {}
    for pattern, desc in patterns.items():
        matches = [m for m in meshes if re.search(pattern, m)]
        if matches:
            found_patterns[pattern] = (desc, len(matches))

    if found_patterns:
        print("Found patterns:")
        for pattern, (desc, count) in sorted(found_patterns.items(), key=lambda x: -x[1][1]):
            pct = 100 * count / len(meshes)
            print(f"  {pct:5.1f}% ({count:3d}) {pattern:20s} - {desc}")
    else:
        print("No clear patterns found in mesh names.")

    print("\n" + "=" * 80)
    print("TIP: If you see mostly unrecognized names:")
    print("  - Mesh names might include vector components (e.g., rho_x, rho_y, rho_z)")
    print("  - Mesh names might be flattened (e.g., rho_lev0_patch0_x_y_z)")
    print("  - These should still be usable - you can group them in the script")
    print("=" * 80)

    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Show all mesh names in an openPMD file")
        print()
        print("Usage: python show_mesh_names.py <file.bp5>")
        print()
        print("Example: python show_mesh_names.py ../data/simulation.it00000000.bp5")
        sys.exit(1)

    filepath = sys.argv[1]
    success = show_meshes(filepath)
    sys.exit(0 if success else 1)
