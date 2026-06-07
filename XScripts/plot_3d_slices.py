#!/usr/bin/env python3
"""Visualize 3D CarpetX data by extracting and compositing 2D slices.

Reads full 3D openPMD output, handles AMR level composition, and creates
publication-quality 2D slices with optional movie assembly.

Usage:
    python plot_3d_slices.py <data_dir> [--axes XY,XZ,YZ] [--out-dir DIR]
                             [--nxny N] [--vmin V] [--vmax V] [--fps 12]
                             [--variable VAR] [--interpolate] [--no-movie]
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


class SliceExtractor:
    """Extract 2D slices from 3D AMR data with level compositing."""

    def __init__(self, nxny=1024, interpolate=False, edge_fill_pix=3):
        """Initialize with canvas resolution and interpolation preference.

        nxny: resolution per axis on uniform canvas
        interpolate: use scipy RegularGridInterpolator (requires scipy)
        edge_fill_pix: pixels to keep from coarser level at patch boundaries
        """
        self.nxny = nxny
        self.interpolate = interpolate
        self.edge_fill_pix = edge_fill_pix

    def list_level_patches(self, mesh_name, series, iteration):
        """List (level, patch_id, mesh_key) sorted coarse→fine."""
        itobj = series.iterations[iteration]
        out = []
        pattern = re.compile(r"_patch(\d+)_lev(\d+)")
        for mesh_key in itobj.meshes:
            m = pattern.search(mesh_key)
            if m:
                patch_id = int(m.group(1))
                level = int(m.group(2))
                out.append((level, patch_id, mesh_key))
        out.sort(key=lambda t: (t[0], t[1]))
        return out

    def extract_plane(self, series, iteration, mesh_key, axis_idx, slice_value=0.0):
        """Extract a single 2D plane from a 3D mesh along an axis.

        axis_idx: 0=x, 1=y, 2=z
        slice_value: coordinate value to slice at
        Returns: (2d_array, (yz_coords, xy_coords))
        """
        itobj = series.iterations[iteration]
        mesh = itobj.meshes[mesh_key]

        # For simplicity, assume the first component is the data
        comp_name = list(mesh)[0] if mesh else None
        if not comp_name:
            return None, None

        field = opc.OpenPMDField(mesh, comp_name)

        # For 3D, assemble chunks into full array. Per-chunk reading is less effective
        # here since we need the full 3D array for slicing, but still safer than
        # loading unwritten padding from sparse AMR domains.
        arr = field.read_full()  # Shape: (nz, ny, nx)

        # Find the slice index closest to slice_value along axis_idx
        coords = field.get_axis_coords(axis_idx)
        slice_idx = np.argmin(np.abs(coords - slice_value))

        if axis_idx == 2:  # x-axis slice; keep (y, z)
            slab = arr[:, :, slice_idx]
            y = field.get_axis_coords(1)
            z = field.get_axis_coords(0)
            return slab, (z, y)
        elif axis_idx == 1:  # y-axis slice; keep (x, z)
            slab = arr[:, slice_idx, :]
            x = field.get_axis_coords(2)
            z = field.get_axis_coords(0)
            return slab, (z, x)
        elif axis_idx == 0:  # z-axis slice; keep (x, y)
            slab = arr[slice_idx, :, :]
            x = field.get_axis_coords(2)
            y = field.get_axis_coords(1)
            return slab, (y, x)

        return None, None

    def composite_axis(self, series, iteration, axis_names="xy", extent_2d=None):
        """Composite all AMR levels for a given slicing axis.

        axis_names: 'xy', 'xz', or 'yz' (axes kept in slice)
        extent_2d: (a_min, a_max, b_min, b_max) for the canvas, or None to auto-detect
        Returns: (canvas_data, (a_coords, b_coords))
        """
        # Determine which mesh to use (heuristic: first 3D mesh found)
        itobj = series.iterations[iteration]
        mesh_key = None
        for name in itobj.meshes:
            # Skip if clearly 1D/2D (could refine this heuristic)
            mesh_key = name
            break

        if not mesh_key:
            return None, None

        # Map axis names to extraction parameters
        axis_map = {
            "xy": (2, 1, 0),  # slice along z; keep x (fast), y (slow)
            "xz": (1, 2, 0),  # slice along y; keep x (fast), z (slow)
            "yz": (0, 2, 1),  # slice along x; keep y (fast), z (slow)
        }
        if axis_names not in axis_map:
            raise ValueError(f"axis_names must be one of {list(axis_map.keys())}")

        slice_axis, slow_axis, fast_axis = axis_map[axis_names]

        # Initialize canvas with NaNs
        canvas = np.full((self.nxny, self.nxny), np.nan, dtype=np.float64)
        g_slow = np.linspace(0, 1, self.nxny)  # will be updated
        g_fast = np.linspace(0, 1, self.nxny)

        # Extract and composite each level
        for level, patch_id, mesh_key_lp in self.list_level_patches(mesh_key, series, iteration):
            try:
                slab, (coords_slow, coords_fast) = self.extract_plane(
                    series, iteration, mesh_key_lp, slice_axis, slice_value=0.0)
                if slab is None:
                    continue

                # Quick auto-extent on first level
                if np.all(np.isnan(canvas)):
                    g_slow = np.linspace(coords_slow.min(), coords_slow.max(), self.nxny)
                    g_fast = np.linspace(coords_fast.min(), coords_fast.max(), self.nxny)

                # Find overlap on canvas
                j0 = np.searchsorted(g_slow, max(coords_slow.min(), g_slow.min()), side="left")
                j1 = np.searchsorted(g_slow, min(coords_slow.max(), g_slow.max()), side="right")
                i0 = np.searchsorted(g_fast, max(coords_fast.min(), g_fast.min()), side="left")
                i1 = np.searchsorted(g_fast, min(coords_fast.max(), g_fast.max()), side="right")

                if j1 <= j0 or i1 <= i0:
                    continue

                sub_slow = g_slow[j0:j1]
                sub_fast = g_fast[i0:i1]

                # Interpolate or nearest-neighbor
                if self.interpolate:
                    try:
                        from scipy.interpolate import RegularGridInterpolator
                        valid = np.where(slab > 0, slab, np.nan)
                        f = RegularGridInterpolator((coords_slow, coords_fast), valid,
                                                    bounds_error=False, fill_value=np.nan)
                        FF, SS = np.meshgrid(sub_fast, sub_slow, indexing="xy")
                        sub = f(np.stack([SS, FF], axis=-1))
                    except ImportError:
                        # Fall back to nearest
                        jj = np.clip(np.searchsorted(coords_slow, sub_slow), 0, len(coords_slow) - 1)
                        ii = np.clip(np.searchsorted(coords_fast, sub_fast), 0, len(coords_fast) - 1)
                        sub = slab[np.ix_(jj, ii)]
                else:
                    jj = np.clip(np.searchsorted(coords_slow, sub_slow), 0, len(coords_slow) - 1)
                    ii = np.clip(np.searchsorted(coords_fast, sub_fast), 0, len(coords_fast) - 1)
                    sub = slab[np.ix_(jj, ii)]

                # Erode fine patches to hide boundaries
                valid = np.isfinite(sub)
                core = _erode(valid, n=self.edge_fill_pix)
                write_mask = core

                if np.any(write_mask):
                    block = canvas[j0:j1, i0:i1]
                    block[write_mask] = sub[write_mask]
                    canvas[j0:j1, i0:i1] = block

            except Exception as e:
                print(f"  WARNING: failed to read level {level} patch {patch_id}: {e}")
                continue

        return canvas, (g_slow, g_fast)


def read_and_plot_3d(filepath, args, out_dir):
    """Read 3D file or directory and create slice plots; return list of frame paths.

    Handles both single files and ADIOS2 parallel directories.
    """
    try:
        import openpmd_api as io
    except ImportError:
        print("ERROR: openpmd_api not available")
        return []

    try:
        series = io.Series(filepath, io.Access.read_only)
    except Exception as e:
        print(f"ERROR opening {filepath}: {e}")
        return []

    iterations = list(series.iterations)
    if not iterations:
        series.close()
        return []

    it = iterations[0]
    time_cu = opc.get_openpmd_time(series, it)
    extractor = SliceExtractor(nxny=args.nxny, interpolate=args.interpolate)

    frames = []
    for axes in args.axes:
        try:
            canvas, (g_slow, g_fast) = extractor.composite_axis(series, it, axes)
            if canvas is None:
                print(f"  SKIP axis {axes}: no data")
                continue

            # Create plot
            fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)

            extent = (g_fast.min(), g_fast.max(), g_slow.min(), g_slow.max())
            vmin = args.vmin if args.vmin else np.nanpercentile(canvas, 1)
            vmax = args.vmax if args.vmax else np.nanpercentile(canvas, 99)

            norm = LogNorm(vmin=vmin, vmax=vmax) if vmin and vmax else None
            im = ax.imshow(canvas, origin="lower", extent=extent,
                          norm=norm, cmap=opc.setup_colormap(args.cmap),
                          interpolation="none")
            ax.set_xlabel(axes[0].upper())
            ax.set_ylabel(axes[1].upper())
            ax.set_title(f"{axes.upper()} slice, it={it}, t={time_cu:.6g} CU" if time_cu
                        else f"{axes.upper()} slice, it={it}")
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label("Field value")

            frame_path = os.path.join(out_dir, f"slice_{axes}_it{it:08d}.png")
            plt.savefig(frame_path, bbox_inches="tight", dpi=150)
            plt.close(fig)
            frames.append(frame_path)
            print(f"  ✓ {axes} slice → {frame_path}")

        except Exception as e:
            print(f"  ERROR plotting {axes}: {e}")

    series.close()
    return frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", help="Directory containing 3D .bp*/.h5 files")
    parser.add_argument("--axes", default="xy,xz,yz",
                       help="Comma-separated list of slice planes (xy/xz/yz)")
    parser.add_argument("--out-dir", default="slices_frames",
                       help="Output directory for frames")
    parser.add_argument("--nxny", type=int, default=1024,
                       help="Canvas resolution per axis")
    parser.add_argument("--vmin", type=float, default=None,
                       help="Color scale minimum")
    parser.add_argument("--vmax", type=float, default=None,
                       help="Color scale maximum")
    parser.add_argument("--cmap", default="plasma",
                       help="Colormap name")
    parser.add_argument("--fps", type=int, default=12,
                       help="Movie frame rate")
    parser.add_argument("--variable", default=None,
                       help="Filter to variable (for future use)")
    parser.add_argument("--interpolate", action="store_true",
                       help="Use scipy interpolation (requires scipy)")
    parser.add_argument("--no-movie", action="store_true",
                       help="Skip movie assembly")
    args = parser.parse_args()

    # Parse axes
    args.axes = [a.lower() for a in args.axes.split(",")]
    for ax in args.axes:
        if ax not in ("xy", "xz", "yz"):
            print(f"ERROR: invalid axis '{ax}'")
            sys.exit(1)

    opc.setup_matplotlib_style()

    data_dir = os.path.abspath(os.path.expanduser(args.data_dir))
    if not os.path.isdir(data_dir):
        print(f"ERROR: {data_dir} not found")
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)

    files = opc.gather_openpmd_series(data_dir)
    if not files:
        print(f"No openPMD files found in {data_dir}")
        sys.exit(1)

    print(f"Found {len(files)} file(s); processing...")
    all_frames = []
    for f in files:
        frames = read_and_plot_3d(f, args, args.out_dir)
        all_frames.extend(frames)

    # Optionally create movies per axis
    if all_frames and not args.no_movie:
        print(f"\nAssembling frames into movies...")
        for ax in args.axes:
            matching = [f for f in all_frames if f"slice_{ax}_" in f]
            if matching:
                movie_file = os.path.join(args.out_dir, f"slices_{ax}.mp4")
                opc.movie_from_frames(sorted(matching), movie_file, fps=args.fps)

    print(f"Done. Output in {args.out_dir}/")


if __name__ == "__main__":
    main()
