#!/usr/bin/env python3
"""Quick metadata diagnostic for openPMD plane/volume directories."""

import argparse
import os
import sys
from collections import defaultdict

import openpmd_common as opc


def classify_series(path):
    if opc.parse_plane_tag(os.path.basename(path.rstrip(os.sep))):
        return "2D"
    return "3D"


def parse_mesh_name(mesh_name):
    return opc.parse_amr_mesh_name(mesh_name) is not None


def inspect_series_metadata(path, variable=None, limit=8):
    try:
        import openpmd_api as io
    except ImportError:
        return {"ok": False, "error": "openpmd_api not available"}

    try:
        series = io.Series(path, io.Access.read_only)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}

    try:
        iterations = list(series.iterations)
        if not iterations:
            return {"ok": False, "error": "no iterations found"}

        it = iterations[0]
        meshes = list(series.iterations[it].meshes)
        recognized = [name for name in meshes if parse_mesh_name(name)]
        unrecognized = [name for name in meshes if name not in recognized]

        samples = []
        matches = []
        for mesh_name in meshes:
            mesh = series.iterations[it].meshes[mesh_name]
            mesh_info = opc.parse_amr_mesh_name(mesh_name)
            comps = list(mesh)
            if variable:
                selected = [
                    comp for comp in comps
                    if opc.mesh_matches_variable(mesh_name, comp, variable, mesh_info)
                ]
            else:
                selected = comps
            if not selected:
                continue

            shape = tuple(mesh[selected[0]].shape) if selected else ()
            label = opc.component_label(mesh_info, selected[0], len(comps)) if selected else ""
            item = (mesh_name, selected, shape, label, mesh_info)
            if len(samples) < limit:
                samples.append(item)
            matches.append(item)

        return {
            "ok": True,
            "iteration": it,
            "mesh_count": len(meshes),
            "recognized_count": len(recognized),
            "unrecognized": unrecognized[:limit],
            "samples": samples,
            "match_count": len(matches),
        }
    finally:
        series.close()


def print_metadata_report(path, variable=None):
    print(f"\nInspecting sample: {os.path.basename(path.rstrip(os.sep))}")
    result = inspect_series_metadata(path, variable=variable)
    if not result["ok"]:
        print(f"  metadata unavailable: {result['error']}")
        return

    tag = opc.parse_plane_tag(os.path.basename(path.rstrip(os.sep)))
    if tag:
        print(
            f"  plane tag: {tag['tag']} "
            f"(plane={tag['plane']}, normal={tag['normal_axis']}, elevation={tag['elevation']})"
        )

    total = result["mesh_count"]
    recognized = result["recognized_count"]
    pct = 100 * recognized / total if total else 0
    print(f"  iteration: {result['iteration']}")
    print(f"  meshes: {total}")
    print(f"  parseable AMR names: {recognized}/{total} ({pct:.0f}%)")
    if variable:
        print(f"  matching '{variable}': {result['match_count']} mesh/component entry(s)")

    print("  sample meshes:")
    for mesh_name, comps, shape, label, mesh_info in result["samples"]:
        print(f"    - {mesh_name}")
        group = mesh_info["group"] if mesh_info else "?"
        centering = f", centering={mesh_info['centering']}" if mesh_info and mesh_info["centering"] else ""
        print(f"      group={group}{centering}, label={label}")
        print(f"      components={comps}, shape={shape}")

    if result["unrecognized"]:
        print("  sample unrecognized names:")
        for name in result["unrecognized"]:
            print(f"    - {name}")


def script_path(script_name):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), script_name)


def quick_diagnose(data_path, variable=None, inspect=False, args=None):
    data_path = os.path.abspath(os.path.expanduser(data_path))
    files = opc.gather_openpmd_series(data_path)
    if args is not None:
        files = opc.filter_series_by_plane(
            files,
            tag=args.tag,
            plane=args.plane,
            normal_axis=args.normal_axis,
            elevation=args.elevation,
        )
    if not files:
        print(f"ERROR: no matching openPMD series found in {data_path}")
        print("Expected names like *.it00000000.bp5, *.it00000000.bp4, or *.it00000000.h5")
        return 1

    by_kind = defaultdict(list)
    for path in files:
        by_kind[classify_series(path)].append(path)

    print(f"Input: {data_path}")
    print(f"Series found: {len(files)}")

    for kind in ("2D", "3D"):
        paths = by_kind[kind]
        if not paths:
            continue
        print(f"\n{kind} candidates: {len(paths)}")
        for path in paths[:8]:
            basename = os.path.basename(path.rstrip(os.sep))
            tag = opc.parse_plane_tag(basename)
            if tag:
                print(f"  - {basename} [{tag['tag']}, elevation={tag['elevation']}]")
            else:
                print(f"  - {basename}")
        if len(paths) > 8:
            print(f"  ... {len(paths) - 8} more")

    if inspect or variable or len(files) == 1:
        if by_kind["2D"]:
            print_metadata_report(by_kind["2D"][0], variable=variable)
        if by_kind["3D"]:
            print_metadata_report(by_kind["3D"][0], variable=variable)

    print("\nSuggested next steps:")
    variable_arg = variable or "<variable>"
    target_arg = data_path
    if by_kind["2D"]:
        sample = by_kind["2D"][0]
        print(f"  python {script_path('show_mesh_names.py')} '{sample}'")
        print(f"  python {script_path('inspect_chunks.py')} '{sample}' {variable_arg}")
        print(f"  python {script_path('plot_2d_planes.py')} '{target_arg}' --variable {variable_arg}")
    if by_kind["3D"]:
        print(f"  python {script_path('plot_3d_slices.py')} '{target_arg}' --variable {variable_arg} --axes xy")

    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_path")
    parser.add_argument("variable", nargs="?", default=None)
    parser.add_argument("--tag", default=None, help="Exact plane tag, e.g. xy_z_pos0012p500")
    parser.add_argument("--plane", choices=("xy", "xz", "yz"), default=None)
    parser.add_argument("--normal-axis", choices=("x", "y", "z"), default=None)
    parser.add_argument("--elevation", type=float, default=None)
    parser.add_argument("--inspect", action="store_true", help="Open one sample and inspect metadata")
    args = parser.parse_args()
    return quick_diagnose(args.data_path, variable=args.variable, inspect=args.inspect, args=args)


if __name__ == "__main__":
    sys.exit(main())
