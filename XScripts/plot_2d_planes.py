#!/usr/bin/env python3
"""Visualize PlanesX/openPMD 2D plane output with chunk-safe AMR compositing.

The important rule is to read only written chunks. CarpetX/PlanesX meshes can
declare the full refined domain while storing only the AMR boxes that intersect
the plane. Loading the declared extent directly can read unwritten padding.
"""

import argparse
import os
import re
import sys

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

import openpmd_common as opc


def _parse_mesh_name(mesh_name):
    """Return (variable_label, level, patch_id) when a mesh name is parseable."""
    patterns = (
        r"(.+?)_patch0*(\d+)_lev0*(\d+)(?:$|_)",
        r"(.+?)_lev0*(\d+)_patch0*(\d+)(?:$|_)",
    )
    for pattern in patterns:
        match = re.match(pattern, mesh_name)
        if not match:
            continue
        label = match.group(1)
        if "patch" in pattern.split("_")[1]:
            patch_id = int(match.group(2))
            level = int(match.group(3))
        else:
            level = int(match.group(2))
            patch_id = int(match.group(3))
        return label, level, patch_id
    return None, None, None


def _safe_name(path):
    name = os.path.basename(path.rstrip(os.sep))
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def _iter_chunk_extents(field):
    """Yield clipped chunk offsets/extents without loading chunk data."""
    for off, ext in field.iter_chunk_extents():
        if len(ext) == 2 and all(e > 0 for e in ext):
            yield off, ext


def _field_bounds_2d(field):
    """Return (xmin, xmax, ymin, ymax) for actual written 2D chunks."""
    x_min, x_max = np.inf, -np.inf
    y_min, y_max = np.inf, -np.inf

    for off, ext in _iter_chunk_extents(field):
        y0 = field.offset[0] + (off[0] + field.position[0]) * field.spacing[0]
        x0 = field.offset[1] + (off[1] + field.position[1]) * field.spacing[1]
        y1 = y0 + ext[0] * field.spacing[0]
        x1 = x0 + ext[1] * field.spacing[1]

        x_min = min(x_min, x0, x1)
        x_max = max(x_max, x0, x1)
        y_min = min(y_min, y0, y1)
        y_max = max(y_max, y0, y1)

    if x_min == np.inf:
        return None
    return x_min, x_max, y_min, y_max


def _find_variable_meshes(iteration, variable_filter=None):
    """Group parseable meshes as {label: {level: [(patch, mesh_name, mesh, comp)]}}."""
    variable_filter = variable_filter.lower() if variable_filter else None
    grouped = {}
    skipped = []

    for mesh_name in sorted(iteration.meshes):
        label, level, patch_id = _parse_mesh_name(mesh_name)
        if label is None:
            skipped.append(mesh_name)
            continue

        mesh = iteration.meshes[mesh_name]
        components = list(mesh)
        if not components:
            continue

        for comp in components:
            comp_label = label if len(components) == 1 else f"{label}_{comp}"
            if variable_filter:
                haystack = f"{mesh_name} {comp_label} {comp}".lower()
                if variable_filter not in haystack:
                    continue

            grouped.setdefault(comp_label, {}).setdefault(level, []).append(
                (patch_id, mesh_name, mesh, comp)
            )

    return grouped, skipped


def _nearest_indices(coords, values):
    idx = np.searchsorted(coords, values)
    idx = np.clip(idx, 0, len(coords) - 1)
    left = np.clip(idx - 1, 0, len(coords) - 1)
    choose_left = np.abs(values - coords[left]) < np.abs(values - coords[idx])
    return np.where(choose_left, left, idx)


def _prepare_chunk_axes(data, y_coords, x_coords):
    """RegularGridInterpolator requires ascending axes."""
    if len(y_coords) > 1 and y_coords[0] > y_coords[-1]:
        y_coords = y_coords[::-1]
        data = data[::-1, :]
    if len(x_coords) > 1 and x_coords[0] > x_coords[-1]:
        x_coords = x_coords[::-1]
        data = data[:, ::-1]
    return data, y_coords, x_coords


def _write_chunk_to_canvas(canvas, grid_y, grid_x, data, y_coords, x_coords, method):
    data, y_coords, x_coords = _prepare_chunk_axes(data, y_coords, x_coords)

    j0 = np.searchsorted(grid_y, max(y_coords.min(), grid_y.min()), side="left")
    j1 = np.searchsorted(grid_y, min(y_coords.max(), grid_y.max()), side="right")
    i0 = np.searchsorted(grid_x, max(x_coords.min(), grid_x.min()), side="left")
    i1 = np.searchsorted(grid_x, min(x_coords.max(), grid_x.max()), side="right")

    if j1 <= j0 or i1 <= i0:
        return

    sub_y = grid_y[j0:j1]
    sub_x = grid_x[i0:i1]

    if method == "linear" and len(y_coords) > 1 and len(x_coords) > 1:
        try:
            from scipy.interpolate import RegularGridInterpolator

            valid = np.where(np.isfinite(data), data, np.nan)
            interp = RegularGridInterpolator(
                (y_coords, x_coords), valid, bounds_error=False, fill_value=np.nan
            )
            yy, xx = np.meshgrid(sub_y, sub_x, indexing="ij")
            values = interp(np.stack([yy, xx], axis=-1))
        except ImportError:
            values = None
    else:
        values = None

    if values is None:
        jj = _nearest_indices(y_coords, sub_y)
        ii = _nearest_indices(x_coords, sub_x)
        values = data[np.ix_(jj, ii)]

    valid_mask = np.isfinite(values)
    if not np.any(valid_mask):
        return

    block = canvas[j0:j1, i0:i1]
    block[valid_mask] = values[valid_mask]
    canvas[j0:j1, i0:i1] = block


def composite_variable(series, levels, nxny=1024, method="linear"):
    """Composite one variable's AMR levels onto a uniform 2D canvas."""
    if not levels:
        raise ValueError("no levels to composite")

    extent = None
    for level in sorted(levels):
        bounds = []
        for _, _, mesh, comp in sorted(levels[level], key=lambda item: item[0]):
            field = opc.OpenPMDField(mesh, comp)
            field_bounds = _field_bounds_2d(field)
            if field_bounds is not None:
                bounds.append(field_bounds)

        if bounds:
            extent = (
                min(b[0] for b in bounds),
                max(b[1] for b in bounds),
                min(b[2] for b in bounds),
                max(b[3] for b in bounds),
            )
            break

    if extent is None:
        raise ValueError("no written 2D chunks found")

    x_min, x_max, y_min, y_max = extent
    grid_x = np.linspace(x_min, x_max, nxny)
    grid_y = np.linspace(y_min, y_max, nxny)
    canvas = np.full((nxny, nxny), np.nan, dtype=np.float64)

    for level in sorted(levels):
        entries = sorted(levels[level], key=lambda item: item[0])
        for _, _, mesh, comp in entries:
            field = opc.OpenPMDField(mesh, comp)
            for data, off, ext in field.read_chunks(series):
                if data.ndim != 2:
                    continue

                y0 = field.offset[0] + (off[0] + field.position[0]) * field.spacing[0]
                x0 = field.offset[1] + (off[1] + field.position[1]) * field.spacing[1]
                y_coords = y0 + np.arange(ext[0]) * field.spacing[0]
                x_coords = x0 + np.arange(ext[1]) * field.spacing[1]

                _write_chunk_to_canvas(
                    canvas, grid_y, grid_x, data, y_coords, x_coords, method
                )

    return canvas, grid_x, grid_y


def _color_limits(data, args):
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return None, None, None

    if args.scale == "log":
        finite = finite[finite > 0]
        if finite.size == 0:
            return None, None, None

    vmin = args.vmin if args.vmin is not None else np.nanpercentile(finite, 1)
    vmax = args.vmax if args.vmax is not None else np.nanpercentile(finite, 99)

    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return None, None, None
    if vmin == vmax:
        vmax = vmin * 1.01 if vmin else 1.0

    norm = LogNorm(vmin=vmin, vmax=vmax) if args.scale == "log" and vmin > 0 else None
    return vmin, vmax, norm


def process_plane_file(filepath, args, out_dir):
    """Read and plot one plane series. Return the written frame path or None."""
    try:
        import openpmd_api as io
    except ImportError:
        print("ERROR: openpmd_api not available; install openPMD-api")
        return None

    try:
        series = io.Series(filepath, io.Access.read_only)
    except Exception as exc:
        print(f"ERROR opening {filepath}: {exc}")
        return None

    try:
        iterations = list(series.iterations)
        if not iterations:
            print(f"SKIP: no iterations in {filepath}")
            return None

        it = iterations[0]
        itobj = series.iterations[it]
        time_cu = opc.get_openpmd_time(series, it)
        grouped, skipped = _find_variable_meshes(itobj, args.variable)

        if skipped and args.verbose:
            print(f"  skipped {len(skipped)} unparseable mesh name(s)")

        if not grouped:
            print(f"SKIP: no matching 2D meshes in {filepath}")
            return None

        labels = sorted(grouped)
        fig, axes = plt.subplots(
            1, len(labels), figsize=(5.2 * len(labels), 4.8), constrained_layout=True
        )
        if len(labels) == 1:
            axes = [axes]

        plotted = 0
        for ax, label in zip(axes, labels):
            try:
                canvas, grid_x, grid_y = composite_variable(
                    series, grouped[label], nxny=args.nxny, method=args.method
                )
            except Exception as exc:
                print(f"  SKIP {label}: {exc}")
                ax.set_visible(False)
                continue

            _, _, norm = _color_limits(canvas, args)
            extent = (grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max())
            im = ax.imshow(
                canvas,
                origin="lower",
                extent=extent,
                norm=norm,
                cmap=opc.setup_colormap(args.cmap),
                interpolation="none",
            )
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            title = label
            if time_cu is not None:
                title += f", t={time_cu:.3e}"
            ax.set_title(title)
            plt.colorbar(im, ax=ax, label=label)
            plotted += 1

        if plotted == 0:
            plt.close(fig)
            return None

        frame_name = f"{_safe_name(filepath)}_it{it:08d}.png"
        frame_path = os.path.join(out_dir, frame_name)
        plt.savefig(frame_path, bbox_inches="tight", dpi=args.dpi)
        plt.close(fig)
        print(f"wrote {frame_path}")
        return frame_path
    finally:
        series.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", help="Directory containing .bp*/.h5 plane series")
    parser.add_argument("--out-dir", default="planes_frames", help="Output directory")
    parser.add_argument("--variable", default=None, help="Variable substring to plot")
    parser.add_argument("--nxny", type=int, default=1024, help="Uniform canvas size")
    parser.add_argument("--method", choices=("linear", "nearest"), default="linear")
    parser.add_argument("--scale", choices=("log", "linear"), default="log")
    parser.add_argument("--vmin", type=float, default=None)
    parser.add_argument("--vmax", type=float, default=None)
    parser.add_argument("--cmap", default="plasma")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--no-movie", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    opc.setup_matplotlib_style()

    data_dir = os.path.abspath(os.path.expanduser(args.data_dir))
    if not os.path.isdir(data_dir):
        print(f"ERROR: not a directory: {data_dir}")
        return 1

    files = opc.gather_openpmd_series(data_dir)
    if not files:
        print(f"ERROR: no openPMD series found in {data_dir}")
        return 1

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"found {len(files)} series")

    frames = []
    for filepath in files:
        frame = process_plane_file(filepath, args, args.out_dir)
        if frame:
            frames.append(frame)

    if frames and not args.no_movie:
        opc.movie_from_frames(frames, os.path.join(args.out_dir, "planes.mp4"), fps=args.fps)

    return 0


if __name__ == "__main__":
    sys.exit(main())
