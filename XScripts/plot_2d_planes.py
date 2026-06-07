#!/usr/bin/env python3
"""Visualize 2D plane data from PlanesX openPMD output.

Reads plane files extracted at simulation runtime, composites AMR levels,
and creates publication-quality plots with optional movie assembly.

Usage:
    python plot_2d_planes.py <data_dir> [--out-dir DIR] [--fps N] [--nxny N]
                             [--vmin V] [--vmax V] [--cmap CMAP] [--variable VAR]
                             [--edge-fill-pix N] [--no-composite] [--no-movie]
"""

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import ticker

import openpmd_common as opc


def _erode(mask, n=1):
    """Simple binary erosion to hide refinement boundaries."""
    m = mask.astype(bool).copy()
    for _ in range(max(0, int(n))):
        up = np.zeros_like(m)
        up[1:, :] = m[:-1, :]
        down = np.zeros_like(m)
        down[:-1, :] = m[1:, :]
        left = np.zeros_like(m)
        left[:, 1:] = m[:, :-1]
        right = np.zeros_like(m)
        right[:, :-1] = m[:, 1:]
        m = m & up & down & left & right
    return m


def _parse_mesh_name(mesh_name):
    """Extract (var_name, level, patch_id) from mesh name.

    Handles multiple naming conventions:
    - rho_lev0_patch0 (format 1)
    - hydrobasex_rho_patch0_lev0 (format 2)
    - gf100_lev0_patch0 (format 3)

    Returns (var_name, level, patch_id) or (None, None, None) if not parseable.
    """
    # Try pattern 1: <var>_lev<L>_patch<P>
    m = re.match(r"(.+?)_lev(\d+)_patch(\d+)", mesh_name)
    if m:
        return m.group(1), int(m.group(2)), int(m.group(3))

    # Try pattern 2: <prefix>_<var>_patch<P>_lev<L>
    m = re.match(r"(.+?)_patch(\d+)_lev(\d+)", mesh_name)
    if m:
        # Extract var from prefix_var format
        prefix_var = m.group(1)
        patch_id = int(m.group(2))
        level = int(m.group(3))
        return prefix_var, level, patch_id

    # Try pattern 3: just return mesh name as var if it has level/patch info
    m = re.search(r"lev(\d+).*patch(\d+)", mesh_name)
    if m:
        level = int(m.group(1))
        patch_id = int(m.group(2))
        return mesh_name, level, patch_id

    return None, None, None


def read_plane_file(filepath):
    """Read a single openPMD plane file or directory with all levels/patches.

    Handles both:
    - Single files: file.bp5
    - ADIOS2 parallel directories: file.bp5/ (contains data.0, data.1, ...)

    Return (iteration, structured_data, time_cu) where structured_data is:
    {variable_name: {level: {patch_id: (array, x_coords, y_coords)}, ...}}
    """
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

    iterations = list(series.iterations)
    if not iterations:
        series.close()
        return None, None, None

    it = iterations[0]
    itobj = series.iterations[it]
    time_cu = opc.get_openpmd_time(series, it)

    # Organize data by variable, then level, then patch
    structured = {}
    unparseable = []

    try:
        for mesh_name in itobj.meshes:
            mesh = itobj.meshes[mesh_name]
            var_name, level, patch_id = _parse_mesh_name(mesh_name)

            if var_name is None:
                unparseable.append(mesh_name)
                continue

            if var_name not in structured:
                structured[var_name] = {}
            if level not in structured[var_name]:
                structured[var_name][level] = {}

            # Read all components
            for comp in mesh:
                try:
                    field = opc.OpenPMDField(mesh, comp)
                    arr = field.read_full()
                    x = field.get_axis_coords(0)
                    y = field.get_axis_coords(1)
                    structured[var_name][level][patch_id] = (arr, x, y)
                except Exception as e:
                    pass

        if unparseable:
            import sys
            print(f"  Warning: {len(unparseable)} mesh(es) not recognized:", file=sys.stderr)
            for name in unparseable[:3]:
                print(f"    - {name}", file=sys.stderr)

    except Exception as e:
        print(f"WARNING: Error reading meshes from {filepath}: {e}")

    series.close()
    return it, structured, time_cu


def composite_amr_plane(var_levels_patches, edge_fill_pix=3, nxny=None):
    """Composite multiple AMR levels into a single 2D plane.

    var_levels_patches: dict of {level: {patch_id: (array, x_coords, y_coords)}}
    Returns: (composited_array, (y_coords, x_coords)) on uniform canvas
    """
    if not var_levels_patches:
        return None, None

    # Auto-detect extent from coarsest level
    all_x = []
    all_y = []
    for level_dict in var_levels_patches.values():
        for arr, x, y in level_dict.values():
            all_x.extend(x)
            all_y.extend(y)

    if not all_x or not all_y:
        return None, None

    all_x = np.array(all_x)
    all_y = np.array(all_y)

    # Create uniform canvas if nxny specified, otherwise use coarse resolution
    if nxny is None:
        nxny = max(len(all_x), len(all_y))

    canvas = np.full((nxny, nxny), np.nan, dtype=np.float64)
    g_y = np.linspace(all_y.min(), all_y.max(), nxny)
    g_x = np.linspace(all_x.min(), all_x.max(), nxny)

    # Process levels coarse → fine
    for level in sorted(var_levels_patches.keys()):
        for patch_id in sorted(var_levels_patches[level].keys()):
            arr, patch_x, patch_y = var_levels_patches[level][patch_id]

            # Find overlap on canvas
            j0 = np.searchsorted(g_y, max(patch_y.min(), g_y.min()), side="left")
            j1 = np.searchsorted(g_y, min(patch_y.max(), g_y.max()), side="right")
            i0 = np.searchsorted(g_x, max(patch_x.min(), g_x.min()), side="left")
            i1 = np.searchsorted(g_x, min(patch_x.max(), g_x.max()), side="right")

            if j1 <= j0 or i1 <= i0:
                continue

            sub_y = g_y[j0:j1]
            sub_x = g_x[i0:i1]

            # Nearest-neighbor lookup (simpler than 3D interpolation)
            jj = np.clip(np.searchsorted(patch_y, sub_y), 0, len(patch_y) - 1)
            ii = np.clip(np.searchsorted(patch_x, sub_x), 0, len(patch_x) - 1)
            sub = arr[np.ix_(jj, ii)]

            # Erode fine patches to hide boundaries
            valid = np.isfinite(sub)
            core = _erode(valid, n=edge_fill_pix)
            write_mask = core

            if np.any(write_mask):
                block = canvas[j0:j1, i0:i1]
                block[write_mask] = sub[write_mask]
                canvas[j0:j1, i0:i1] = block

    return canvas, (g_y, g_x)


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
    it, structured, time_cu = read_plane_file(filepath)
    if it is None or not structured:
        print(f"  SKIP (no data): {filepath}")
        return None

    print(f"  Iteration {it}, t={time_cu:.6g} CU" if time_cu else f"  Iteration {it}")

    # Filter to requested variable if specified
    if args.variable:
        matching = {k: v for k, v in structured.items() if args.variable.lower() in k.lower()}
        if not matching:
            print(f"    Variable '{args.variable}' not found; available: {list(structured.keys())}")
            return None
        structured = matching

    # Create figure with one panel per variable
    nvars = len(structured)
    fig, axes = plt.subplots(1, nvars, figsize=(5 * nvars, 4.5), constrained_layout=True)
    if nvars == 1:
        axes = [axes]

    for (var_label, levels_patches), ax in zip(sorted(structured.items()), axes):
        # Composite AMR levels if requested
        if args.no_composite or len(levels_patches) == 1:
            # Use only finest level
            finest_level = max(levels_patches.keys())
            finest_patches = levels_patches[finest_level]
            # Merge all patches from finest level
            all_arrays = [arr for arr, x, y in finest_patches.values()]
            all_x = [x for arr, x, y in finest_patches.values()]
            all_y = [y for arr, x, y in finest_patches.values()]

            if all_arrays:
                # Simple concatenation (assumes patches tile properly)
                arr = np.concatenate(all_arrays, axis=1) if len(all_arrays) > 1 else all_arrays[0]
                x = np.concatenate(all_x, axis=0) if len(all_x) > 1 else all_x[0]
                y = np.concatenate(all_y, axis=0) if len(all_y) > 1 else all_y[0]
            else:
                continue
        else:
            # Composite all levels with proper boundary handling
            arr, (y, x) = composite_amr_plane(levels_patches,
                                             edge_fill_pix=args.edge_fill_pix,
                                             nxny=args.nxny)
            if arr is None:
                continue

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
    parser.add_argument("--edge-fill-pix", type=int, default=3,
                       help="Pixels to keep from coarse level at boundaries (default: 3)")
    parser.add_argument("--no-composite", action="store_true",
                       help="Skip AMR compositing; use only finest level per region")
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
