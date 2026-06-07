#!/usr/bin/env python3
"""Quick metadata diagnostic for openPMD plane/volume directories."""

import argparse
import os
import re
import sys
from collections import defaultdict

import openpmd_common as opc


def classify_series(path):
    basename = os.path.basename(path.rstrip(os.sep)).lower()
    if re.search(r"\.(xy|xz|yz)_", basename):
        return "2D"
    if re.search(r"(xy|xz|yz).*_(x|y|z)_?pos", basename):
        return "2D"
    if "_pos" in basename or "planes" in basename:
        return "2D"
    return "3D"


def parse_mesh_name(mesh_name):
    patterns = (
        r"(.+?)_patch0*(\d+)_lev0*(\d+)(?:$|_)",
        r"(.+?)_lev0*(\d+)_patch0*(\d+)(?:$|_)",
    )
    return any(re.match(pattern, mesh_name) for pattern in patterns)


def inspect_series_metadata(path, limit=8):
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
        for mesh_name in meshes[:limit]:
            mesh = series.iterations[it].meshes[mesh_name]
            comps = list(mesh)
            shape = tuple(mesh[comps[0]].shape) if comps else ()
            samples.append((mesh_name, comps, shape))

        return {
            "ok": True,
            "iteration": it,
            "mesh_count": len(meshes),
            "recognized_count": len(recognized),
            "unrecognized": unrecognized[:limit],
            "samples": samples,
        }
    finally:
        series.close()


def print_metadata_report(path):
    print(f"\nInspecting sample: {os.path.basename(path.rstrip(os.sep))}")
    result = inspect_series_metadata(path)
    if not result["ok"]:
        print(f"  metadata unavailable: {result['error']}")
        return

    total = result["mesh_count"]
    recognized = result["recognized_count"]
    pct = 100 * recognized / total if total else 0
    print(f"  iteration: {result['iteration']}")
    print(f"  meshes: {total}")
    print(f"  parseable AMR names: {recognized}/{total} ({pct:.0f}%)")

    print("  sample meshes:")
    for mesh_name, comps, shape in result["samples"]:
        print(f"    - {mesh_name}")
        print(f"      components={comps}, shape={shape}")

    if result["unrecognized"]:
        print("  sample unrecognized names:")
        for name in result["unrecognized"]:
            print(f"    - {name}")


def quick_diagnose(data_dir, inspect=False):
    data_dir = os.path.abspath(os.path.expanduser(data_dir))
    if not os.path.isdir(data_dir):
        print(f"ERROR: not a directory: {data_dir}")
        return 1

    files = opc.gather_openpmd_series(data_dir)
    if not files:
        print(f"ERROR: no openPMD series found in {data_dir}")
        print("Expected names like *.it00000000.bp5, *.it00000000.bp4, or *.it00000000.h5")
        return 1

    by_kind = defaultdict(list)
    for path in files:
        by_kind[classify_series(path)].append(path)

    print(f"Directory: {data_dir}")
    print(f"Series found: {len(files)}")

    for kind in ("2D", "3D"):
        paths = by_kind[kind]
        if not paths:
            continue
        print(f"\n{kind} candidates: {len(paths)}")
        for path in paths[:8]:
            print(f"  - {os.path.basename(path.rstrip(os.sep))}")
        if len(paths) > 8:
            print(f"  ... {len(paths) - 8} more")

    if inspect:
        if by_kind["2D"]:
            print_metadata_report(by_kind["2D"][0])
        if by_kind["3D"]:
            print_metadata_report(by_kind["3D"][0])

    print("\nSuggested next steps:")
    if by_kind["2D"]:
        sample = by_kind["2D"][0]
        print(f"  python XScripts/show_mesh_names.py '{sample}'")
        print(f"  python XScripts/inspect_chunks.py '{sample}' <variable>")
        print(f"  python XScripts/plot_2d_planes.py '{data_dir}' --variable <variable>")
    if by_kind["3D"]:
        print(f"  python XScripts/plot_3d_slices.py '{data_dir}' --variable <variable> --axes xy")

    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir")
    parser.add_argument("--inspect", action="store_true", help="Open one sample and inspect metadata")
    args = parser.parse_args()
    return quick_diagnose(args.data_dir, inspect=args.inspect)


if __name__ == "__main__":
    sys.exit(main())
