#!/usr/bin/env python3
"""Visualize 2D plane data from PlanesX openPMD output.

Reads plane files extracted at simulation runtime and creates publication-quality
plots with optional movie assembly.

Usage:
    python plot_2d_planes.py <data_dir> [--out-dir DIR] [--fps N] [--nxny N]
                             [--vmin V] [--vmax V] [--cmap CMAP] [--variable VAR]
                             [--no-movie]
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import ticker

import openpmd_common as opc


def read_plane_file(filepath):
    """Read a single openPMD plane file; return (iteration, data_dict, time_cu)."""
    try:
        import openpmd_api as io
    except ImportError:
        print("ERROR: openpmd_api not available; install with: pip install openPMD-api")
        return None, None, None

    try:
        series = io.Series(filepath, io.Access.read_only)
    except Exception as e:
        print(f"ERROR opening {filepath}: {e}")
        return None, None, None

    # Typically one iteration per file
    iterations = list(series.iterations)
    if not iterations:
        series.close()
        return None, None, None

    it = iterations[0]
    itobj = series.iterations[it]
    time_cu = opc.get_openpmd_time(series, it)

    # Collect all meshes by variable name
    data_dict = {}
    try:
        for mesh_name in itobj.meshes:
            mesh = itobj.meshes[mesh_name]
            # Simple strategy: treat each mesh as a variable
            # In practice, parse mesh_name to extract variable label
            for comp in mesh:
                try:
                    field = opc.OpenPMDField(mesh, comp)
                    arr = field.read_full()
                    var_label = f"{mesh_name}:{comp}"
                    data_dict[var_label] = {
                        "array": arr,
                        "x": field.get_axis_coords(0),  # assuming 2D array [y, x]
                        "y": field.get_axis_coords(1),
                        "mesh": mesh,
                    }
                except Exception as e:
                    pass  # Skip variables that don't load
    except Exception as e:
        print(f"WARNING: Error reading meshes from {filepath}: {e}")

    series.close()
    return it, data_dict, time_cu


def plot_plane(ax, data_2d, extent, vmin, vmax, cmap, title=""):
    """Create a 2D image plot on an axis."""
    norm = LogNorm(vmin=vmin, vmax=vmax) if (vmin and vmax) else None
    im = ax.imshow(data_2d, origin="lower", extent=extent,
                   norm=norm, cmap=cmap, interpolation="none")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if title:
        ax.set_title(title)
    return im


def process_plane_file(filepath, args, out_dir):
    """Read and plot a single plane file; return frame path or None."""
    it, data_dict, time_cu = read_plane_file(filepath)
    if it is None or not data_dict:
        print(f"  SKIP (no data): {filepath}")
        return None

    print(f"  Iteration {it}, t={time_cu:.6g} CU" if time_cu else f"  Iteration {it}")

    # Filter to requested variable if specified
    if args.variable:
        matching = {k: v for k, v in data_dict.items() if args.variable.lower() in k.lower()}
        if not matching:
            print(f"    Variable '{args.variable}' not found; available: {list(data_dict.keys())}")
            return None
        data_dict = matching

    # Create figure with one panel per variable
    nvars = len(data_dict)
    fig, axes = plt.subplots(1, nvars, figsize=(5 * nvars, 4.5), constrained_layout=True)
    if nvars == 1:
        axes = [axes]

    frames = []
    for (var_label, var_data), ax in zip(sorted(data_dict.items()), axes):
        arr = var_data["array"]
        x = var_data["x"]
        y = var_data["y"]
        extent = (x.min(), x.max(), y.min(), y.max())

        vmin = args.vmin if args.vmin else np.nanpercentile(arr, 1)
        vmax = args.vmax if args.vmax else np.nanpercentile(arr, 99)

        im = plot_plane(ax, arr, extent, vmin, vmax,
                       opc.setup_colormap(args.cmap), title=var_label)
        plt.colorbar(im, ax=ax, label=var_label)

    # Save frame
    frame_path = os.path.join(out_dir, f"frame_it{it:08d}.png")
    plt.savefig(frame_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return frame_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", help="Directory containing plane .bp*/.h5 files")
    parser.add_argument("--out-dir", default="planes_frames",
                       help="Output directory for frames")
    parser.add_argument("--fps", type=int, default=12,
                       help="Movie frame rate")
    parser.add_argument("--nxny", type=int, default=None,
                       help="Resample to NxN grid (default: native)")
    parser.add_argument("--vmin", type=float, default=None,
                       help="Color scale minimum")
    parser.add_argument("--vmax", type=float, default=None,
                       help="Color scale maximum")
    parser.add_argument("--cmap", default="plasma",
                       help="Colormap name")
    parser.add_argument("--variable", default=None,
                       help="Filter to variable(s) matching this substring")
    parser.add_argument("--no-movie", action="store_true",
                       help="Skip movie assembly")
    args = parser.parse_args()

    opc.setup_matplotlib_style()

    data_dir = os.path.abspath(os.path.expanduser(args.data_dir))
    if not os.path.isdir(data_dir):
        print(f"ERROR: {data_dir} not found")
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)

    # Find all plane files
    files = opc.gather_openpmd_series(data_dir)
    if not files:
        print(f"No openPMD plane files found in {data_dir}")
        sys.exit(1)

    print(f"Found {len(files)} plane file(s)")
    frames = []
    for f in files:
        frame = process_plane_file(f, args, args.out_dir)
        if frame:
            frames.append(frame)

    # Optionally create movie
    if frames and not args.no_movie:
        print(f"\nAssembling {len(frames)} frames into movie...")
        movie_file = os.path.join(args.out_dir, "planes.mp4")
        opc.movie_from_frames(frames, movie_file, fps=args.fps)

    print(f"Done. Output in {args.out_dir}/")


if __name__ == "__main__":
    main()
