#!/usr/bin/env python3
"""Show all mesh names and structure in an openPMD file."""

import sys
import os
import openpmd_common as opc

def show_meshes(filepath):
    """Display all meshes and components in an openPMD file or directory.

    Accepts both:
    - Single files: file.bp5
    - ADIOS2 parallel directories: file.bp5/ (contains data.0, data.1, ...)
    """
    try:
        import openpmd_api as io
    except ImportError:
        print("ERROR: openpmd_api not available; install with: pip install openPMD-api")
        return False

    filepath = os.path.abspath(os.path.expanduser(filepath))
    # Remove trailing slash for consistency
    filepath = filepath.rstrip(os.sep)

    if not (os.path.isfile(filepath) or os.path.isdir(filepath)):
        print(f"ERROR: Path not found: {filepath}")
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
        mesh_info = opc.parse_amr_mesh_name(mesh_name)

        # Truncate long names
        display_name = mesh_name if len(mesh_name) < 70 else mesh_name[:67] + "..."
        print(f"{i+1:4d}. {display_name}")
        print(f"       Components: {components}")
        if hasattr(mesh[components[0]], 'shape'):
            print(f"       Shape: {shape}")
        if mesh_info:
            plane = mesh_info["plane"]
            plane_text = ""
            if plane:
                plane_text = (
                    f", tag={plane['tag']}, plane={plane['plane']}, "
                    f"normal={plane['normal_axis']}, elevation={plane['elevation']}"
                )
            centering = f", centering={mesh_info['centering']}" if mesh_info["centering"] else ""
            print(
                f"       Parsed: group={mesh_info['group']}, "
                f"level={mesh_info['level']}, patch={mesh_info['patch']}"
                f"{centering}{plane_text}"
            )
        print()

    series.close()

    # Try to find pattern
    print("=" * 80)
    print("Analyzing naming patterns...\n")

    parsed = [opc.parse_amr_mesh_name(name) for name in meshes]
    parsed = [info for info in parsed if info is not None]
    print(f"Parseable AMR mesh names: {len(parsed)}/{len(meshes)}")
    tags = sorted({info["plane"]["tag"] for info in parsed if info["plane"]})
    if tags:
        print("Plane tags:")
        for tag in tags[:10]:
            tag_info = opc.parse_plane_tag(tag)
            print(
                f"  - {tag}: plane={tag_info['plane']}, "
                f"normal={tag_info['normal_axis']}, elevation={tag_info['elevation']}"
            )
        if len(tags) > 10:
            print(f"  ... {len(tags) - 10} more")

    centerings = sorted({info["centering"] for info in parsed if info["centering"]})
    if centerings:
        print(f"Centering suffixes: {', '.join(centerings)}")

    print("\n" + "=" * 80)
    print("TIP: Use the component names above with --variable, e.g. hydrobasex_rho.")
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
