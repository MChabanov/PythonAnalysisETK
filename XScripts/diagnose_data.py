#!/usr/bin/env python3
"""Comprehensive diagnostic: scan directory, identify 2D/3D data, inspect mesh structures."""

import sys
import os
import glob
import re
from pathlib import Path
from collections import defaultdict


def find_openpmd_files(root_dir, max_depth=3):
    """Recursively find all openPMD files (bp5, bp4, bp, h5) up to max_depth."""
    root_dir = os.path.abspath(os.path.expanduser(root_dir))
    if not os.path.isdir(root_dir):
        print(f"ERROR: Not a directory: {root_dir}")
        return []

    files = []

    try:
        for root, dirs, filenames in os.walk(root_dir):
            # Limit depth
            depth = root[len(root_dir):].count(os.sep)
            if depth > max_depth:
                dirs[:] = []  # Clear dirs in-place to prevent descent
                continue

            for filename in filenames:
                # Skip metadata and lock files
                if ".md." in filename or filename.endswith(".dir"):
                    continue

                # Check for openPMD extensions
                for ext in ("bp5", "bp4", "bp", "h5"):
                    if filename.endswith(f".{ext}"):
                        filepath = os.path.join(root, filename)
                        # Verify it's a file (not a link or special file)
                        if os.path.isfile(filepath):
                            files.append(filepath)
                        break
    except OSError as e:
        print(f"WARNING: Error scanning directory: {e}")
        pass

    return sorted(files)


def classify_file(filepath):
    """Classify as 2D or 3D based on filename.

    Returns: ("2D" | "3D" | "unknown", description)
    """
    basename = os.path.basename(filepath)

    # 2D plane patterns
    if re.search(r"(xy|xz|yz).*_z|_y|_x.*pos", basename):
        return "2D", "PlanesX (plane designation in name)"
    if re.search(r"\.(xy|xz|yz)_", basename):
        return "2D", "PlanesX (plane designation in name)"
    if "planes" in basename.lower():
        return "2D", "Likely PlanesX (contains 'planes')"
    if re.search(r"_pos\d+", basename):
        return "2D", "Likely PlanesX (elevation in name)"

    # 3D patterns
    if "3d" in basename.lower():
        return "3D", "Contains '3D' in filename"
    if re.search(r"_lev\d+_patch\d+", basename):
        return "3D", "AMR structure in name"

    return "unknown", "Cannot determine from filename alone"


def inspect_meshes(filepath):
    """Open file and analyze mesh structure.

    Returns: {
        'success': bool,
        'iteration': int or None,
        'num_meshes': int,
        'mesh_names': [str],
        'patterns': {pattern: count},
        'components': {mesh_name: [components]},
        'shapes': {mesh_name: shape},
        'error': str or None,
    }
    """
    try:
        import openpmd_api as io
    except ImportError:
        return {
            'success': False,
            'error': 'openpmd_api not available',
        }

    try:
        series = io.Series(filepath, io.Access.read_only)
    except Exception as e:
        return {
            'success': False,
            'error': f'Failed to open: {str(e)[:100]}',
        }

    iterations = list(series.iterations)
    if not iterations:
        series.close()
        return {
            'success': False,
            'error': 'No iterations in file',
        }

    it = iterations[0]
    itobj = series.iterations[it]
    meshes = list(itobj.meshes)

    result = {
        'success': True,
        'iteration': it,
        'num_meshes': len(meshes),
        'mesh_names': [],
        'patterns': defaultdict(int),
        'components': {},
        'shapes': {},
        'error': None,
    }

    for mesh_name in meshes:
        result['mesh_names'].append(mesh_name)
        try:
            mesh = itobj.meshes[mesh_name]
            comps = list(mesh)
            result['components'][mesh_name] = comps

            if comps:
                comp = mesh[comps[0]]
                if hasattr(comp, 'shape'):
                    result['shapes'][mesh_name] = tuple(comp.shape)
        except Exception:
            pass

    # Find patterns
    patterns = {
        r"_lev\d+_patch\d+": "AMR (lev_patch)",
        r"_patch\d+_lev\d+": "AMR (patch_lev)",
        r"_lev\d+": "Has levels",
        r"_patch\d+": "Has patches",
        r"_(x|y|z)$": "Vector components",
        r"gf\d+": "Grid functions",
        r"_[xy][z]$": "Plane components",
    }

    for pattern, label in patterns.items():
        count = sum(1 for name in meshes if re.search(pattern, name))
        if count > 0:
            result['patterns'][label] = count

    series.close()
    return result


def format_report(root_dir, files_by_category):
    """Format diagnostic report."""
    print("\n" + "=" * 80)
    print("OPENPMD DATA DIAGNOSTIC REPORT")
    print("=" * 80)
    print(f"Root: {root_dir}")
    print(f"Scanned: {sum(len(f) for f in files_by_category.values())} file(s)\n")

    total_meshes = 0
    total_files_ok = 0

    for category in ("2D", "3D", "unknown"):
        files = files_by_category[category]
        if not files:
            continue

        print(f"\n{category.upper()} DATA ({len(files)} file{'s' if len(files) != 1 else ''})")
        print("-" * 80)

        for filepath, classification, analysis in files:
            basename = os.path.basename(filepath)
            class_type, class_desc = classification

            print(f"\n  {basename}")
            print(f"    Category: {class_type} ({class_desc})")

            if not analysis['success']:
                print(f"    ✗ ERROR: {analysis['error']}")
                continue

            total_files_ok += 1
            total_meshes += analysis['num_meshes']

            print(f"    Iteration: {analysis['iteration']}")
            print(f"    Meshes: {analysis['num_meshes']}")

            if analysis['patterns']:
                print(f"    Patterns found:")
                for label, count in sorted(analysis['patterns'].items(), key=lambda x: -x[1]):
                    pct = 100 * count / analysis['num_meshes']
                    print(f"      • {label}: {count} ({pct:.0f}%)")

            # Check mesh name compatibility
            compatible = 0
            patterns_to_check = [
                r"(.+?)_lev(\d+)_patch(\d+)",
                r"(.+?)_patch(\d+)_lev(\d+)",
            ]
            for mesh_name in analysis['mesh_names']:
                for pattern in patterns_to_check:
                    if re.match(pattern, mesh_name):
                        compatible += 1
                        break

            if compatible > 0:
                pct = 100 * compatible / analysis['num_meshes']
                status = "✓" if pct > 50 else "⚠"
                print(f"    {status} Compatible with visualization: {compatible}/{analysis['num_meshes']} ({pct:.0f}%)")
            else:
                print(f"    ✗ Compatible with visualization: 0/{analysis['num_meshes']}")
                print(f"    Sample mesh names:")
                for name in analysis['mesh_names'][:3]:
                    print(f"      - {name}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total readable files: {total_files_ok}")
    print(f"Total meshes found: {total_meshes}")

    has_2d = len(files_by_category["2D"]) > 0
    has_3d = len(files_by_category["3D"]) > 0

    print(f"\nData types available:")
    print(f"  {'✓' if has_2d else '✗'} 2D planes")
    print(f"  {'✓' if has_3d else '✗'} 3D volumes")

    if has_2d and has_3d:
        print("\n✓ You can use BOTH visualization scripts:")
        print("  - python XScripts/plot_2d_planes.py <2d_dir>")
        print("  - python XScripts/plot_3d_slices.py <3d_dir>")
    elif has_2d:
        print("\n✓ Use 2D visualization:")
        print("  - python XScripts/plot_2d_planes.py <data_dir>")
    elif has_3d:
        print("\n✓ Use 3D visualization:")
        print("  - python XScripts/plot_3d_slices.py <data_dir>")
    else:
        print("\n✗ No usable data files found")

    # Mesh compatibility summary
    print("\n" + "-" * 80)
    print("MESH COMPATIBILITY NOTES")
    print("-" * 80)

    has_incompatible = any(
        analysis['success'] and not any(
            re.match(r"(.+?)_lev(\d+)_patch(\d+)", n) or
            re.match(r"(.+?)_patch(\d+)_lev(\d+)", n)
            for n in analysis['mesh_names']
        )
        for _, _, analysis in sum(files_by_category.values(), [])
    )

    if has_incompatible:
        print("\n⚠ Some files have mesh names that don't match expected patterns.")
        print("  This might be OK! Reasons include:")
        print("    1. Vector components (rho_x, rho_y, rho_z)")
        print("    2. Different naming schemes from custom writers")
        print("    3. Flattened mesh names")
        print("\nTo see detailed mesh names, run:")
        print("  python XScripts/show_mesh_names.py <file.bp5>")
    else:
        print("\n✓ All mesh names follow expected patterns!")

    print("\n" + "=" * 80)


def main():
    if len(sys.argv) < 2:
        print("Comprehensive diagnostic for openPMD simulation data")
        print()
        print("Usage: python diagnose_data.py <root_directory> [--verbose]")
        print()
        print("This script:")
        print("  1. Scans directory tree for all openPMD files")
        print("  2. Classifies as 2D (planes) or 3D (volumes)")
        print("  3. Inspects mesh structures in each file")
        print("  4. Reports compatibility with visualization scripts")
        print()
        print("Example: python diagnose_data.py ~/simulations/")
        print("         python diagnose_data.py ~/simulations/ --verbose")
        return 1

    root_dir = sys.argv[1]
    verbose = "--verbose" in sys.argv

    print(f"Scanning {root_dir}...")
    files = find_openpmd_files(root_dir)

    if not files:
        print(f"✗ No openPMD files found in {root_dir}")
        print()
        print("Looking for files matching: *.bp5, *.bp4, *.bp, *.h5")
        print(f"Excluding: .md.*, *.dir")
        print()
        print("Checking what's in the directory...")
        try:
            items = os.listdir(root_dir)
            bp_files = [f for f in items if '.bp' in f or '.h5' in f]
            if bp_files:
                print(f"Found {len(bp_files)} openPMD-like files:")
                for f in bp_files[:10]:
                    print(f"  - {f}")
            else:
                print("No .bp* or .h5 files found at all")
                print(f"Directory contains: {', '.join(items[:5])}...")
        except Exception as e:
            print(f"Could not list directory: {e}")
        return 1

    print(f"Found {len(files)} file(s), inspecting...")

    # Classify and analyze each file
    files_by_category = defaultdict(list)

    for i, filepath in enumerate(files):
        print(f"  [{i+1}/{len(files)}] {os.path.basename(filepath)}", end="\r")
        classification = classify_file(filepath)
        analysis = inspect_meshes(filepath)
        category = classification[0]
        files_by_category[category].append((filepath, classification, analysis))

    print(" " * 80, end="\r")  # Clear progress line

    # Generate report
    format_report(root_dir, files_by_category)

    return 0


if __name__ == "__main__":
    sys.exit(main())
