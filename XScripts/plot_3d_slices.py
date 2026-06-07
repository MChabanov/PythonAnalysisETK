#!/usr/bin/env python3
"""Visualize 3D CarpetX/openPMD data by extracting chunk-safe 2D slices."""

import argparse
import os
import re
import sys

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

import openpmd_common as opc


AXES = {
    # Record order is assumed to be (z, y, x), matching CarpetX conventions.
    "xy": {"slice": 0, "slow": 1, "fast": 2, "xlabel": "x", "ylabel": "y"},
    "xz": {"slice": 1, "slow": 0, "fast": 2, "xlabel": "x", "ylabel": "z"},
    "yz": {"slice": 2, "slow": 0, "fast": 1, "xlabel": "y", "ylabel": "z"},
}


def _safe_name(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def _find_field_groups(iteration, variable_filter=None):
    """Return {label: {level: [(patch, mesh_name, mesh, comp)]}}."""
    groups = {}

    for mesh_name in sorted(iteration.meshes):
        mesh_info = opc.parse_amr_mesh_name(mesh_name)
        if mesh_info is None:
            continue

        mesh = iteration.meshes[mesh_name]
        components = list(mesh)
        if not components:
            continue

        for comp in components:
            comp_label = opc.component_label(mesh_info, comp, len(components))
            if not opc.mesh_matches_variable(mesh_name, comp, variable_filter, mesh_info):
                continue
            groups.setdefault(comp_label, {}).setdefault(mesh_info["level"], []).append(
                (mesh_info["patch"], mesh_name, mesh, comp)
            )

    return groups


def _axis_coords(field, off, ext, axis):
    idx = np.arange(off[axis], off[axis] + ext[axis])
    return field.offset[axis] + (idx + field.position[axis]) * field.spacing[axis]


def _chunk_slice_index(field, off, ext, axis, slice_value):
    coords = _axis_coords(field, off, ext, axis)
    pad = 0.5 * abs(field.spacing[axis])
    if slice_value < coords.min() - pad or slice_value > coords.max() + pad:
        return None
    return int(np.argmin(np.abs(coords - slice_value)))


def _prepare_patch(data, slow_coords, fast_coords):
    if len(slow_coords) > 1 and slow_coords[0] > slow_coords[-1]:
        slow_coords = slow_coords[::-1]
        data = data[::-1, :]
    if len(fast_coords) > 1 and fast_coords[0] > fast_coords[-1]:
        fast_coords = fast_coords[::-1]
        data = data[:, ::-1]
    return data, slow_coords, fast_coords


def _load_chunk_slab(series, field, off, ext, axes, slice_value):
    slice_axis = axes["slice"]
    local_idx = _chunk_slice_index(field, off, ext, slice_axis, slice_value)
    if local_idx is None:
        return None

    read_off = list(off)
    read_ext = list(ext)
    read_off[slice_axis] = off[slice_axis] + local_idx
    read_ext[slice_axis] = 1

    raw = field.record.load_chunk(read_off, read_ext)
    series.flush()
    slab = np.asarray(raw).reshape(tuple(read_ext))
    slab = np.squeeze(slab, axis=slice_axis)

    remaining = [axis for axis in range(len(ext)) if axis != slice_axis]
    desired = [axes["slow"], axes["fast"]]
    if remaining != desired:
        slab = np.transpose(slab, [remaining.index(axis) for axis in desired])

    slow_coords = _axis_coords(field, off, ext, axes["slow"])
    fast_coords = _axis_coords(field, off, ext, axes["fast"])
    return _prepare_patch(slab, slow_coords, fast_coords)


def _extent_for_slice(levels, axes, slice_value):
    """Use the coarsest level with intersecting chunks to define the canvas."""
    for level in sorted(levels):
        bounds = []
        for _, _, mesh, comp in sorted(levels[level], key=lambda item: item[0]):
            field = opc.OpenPMDField(mesh, comp)
            if len(field.shape) != 3:
                continue

            for off, ext in field.iter_chunk_extents():
                if len(ext) != 3:
                    continue
                if _chunk_slice_index(field, off, ext, axes["slice"], slice_value) is None:
                    continue
                slow = _axis_coords(field, off, ext, axes["slow"])
                fast = _axis_coords(field, off, ext, axes["fast"])
                bounds.append((fast.min(), fast.max(), slow.min(), slow.max()))

        if bounds:
            return (
                min(b[0] for b in bounds),
                max(b[1] for b in bounds),
                min(b[2] for b in bounds),
                max(b[3] for b in bounds),
            )
    return None


def composite_slice(series, levels, axis_name, nxny, method, slice_value):
    axes = AXES[axis_name]
    extent = _extent_for_slice(levels, axes, slice_value)
    if extent is None:
        raise ValueError(f"no chunks intersect {axis_name} slice at {slice_value}")

    canvas = opc.Canvas2D(extent, nxny)

    for level in sorted(levels):
        for _, _, mesh, comp in sorted(levels[level], key=lambda item: item[0]):
            field = opc.OpenPMDField(mesh, comp)
            if len(field.shape) != 3:
                continue

            for off, ext in field.iter_chunk_extents():
                if len(ext) != 3:
                    continue
                loaded = _load_chunk_slab(series, field, off, ext, axes, slice_value)
                if loaded is None:
                    continue
                slab, slow_coords, fast_coords = loaded
                canvas.add_patch(slab, fast_coords, slow_coords, method=method)

    return canvas.data, canvas.x, canvas.y


def _color_norm(data, args):
    finite = data[np.isfinite(data)]
    if args.scale == "log":
        finite = finite[finite > 0]
    if finite.size == 0:
        return None

    vmin = args.vmin if args.vmin is not None else np.nanpercentile(finite, 1)
    vmax = args.vmax if args.vmax is not None else np.nanpercentile(finite, 99)
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return None
    if vmin == vmax:
        vmax = vmin * 1.01 if vmin else 1.0
    if args.scale == "log" and vmin > 0:
        return LogNorm(vmin=vmin, vmax=vmax)
    return None


def process_file(filepath, args, out_dir):
    try:
        import openpmd_api as io
    except ImportError:
        print("ERROR: openpmd_api not available; install openPMD-api")
        return []

    try:
        series = io.Series(filepath, io.Access.read_only)
    except Exception as exc:
        print(f"ERROR opening {filepath}: {exc}")
        return []

    try:
        iterations = list(series.iterations)
        if not iterations:
            print(f"SKIP: no iterations in {filepath}")
            return []

        it = iterations[0]
        itobj = series.iterations[it]
        time_cu = opc.get_openpmd_time(series, it)
        groups = _find_field_groups(itobj, args.variable)
        if not groups:
            print(f"SKIP: no matching 3D meshes in {filepath}")
            return []

        labels = sorted(groups)
        if len(labels) > 1 and not args.all_variables:
            print(f"using {labels[0]} from {filepath}; pass --all-variables to plot all")
            labels = labels[:1]

        frames = []
        file_name = _safe_name(os.path.basename(filepath.rstrip(os.sep)))

        for label in labels:
            for axis_name in args.axes:
                try:
                    data, fast, slow = composite_slice(
                        series,
                        groups[label],
                        axis_name,
                        nxny=args.nxny,
                        method=args.method,
                        slice_value=args.slice_value,
                    )
                except Exception as exc:
                    print(f"  SKIP {label} {axis_name}: {exc}")
                    continue

                fig, ax = plt.subplots(figsize=(7.5, 6.5), constrained_layout=True)
                extent = (fast.min(), fast.max(), slow.min(), slow.max())
                im = ax.imshow(
                    data,
                    origin="lower",
                    extent=extent,
                    norm=_color_norm(data, args),
                    cmap=opc.setup_colormap(args.cmap),
                    interpolation="none",
                )
                ax.set_xlabel(AXES[axis_name]["xlabel"])
                ax.set_ylabel(AXES[axis_name]["ylabel"])
                title = f"{label} {axis_name}, it={it}"
                if time_cu is not None:
                    title += f", t={time_cu:.3e}"
                ax.set_title(title)
                plt.colorbar(im, ax=ax, label=label)

                frame_name = f"slice_{axis_name}_{_safe_name(label)}_{file_name}_it{it:08d}.png"
                frame_path = os.path.join(out_dir, frame_name)
                plt.savefig(frame_path, bbox_inches="tight", dpi=args.dpi)
                plt.close(fig)
                frames.append(frame_path)
                print(f"wrote {frame_path}")

        return frames
    finally:
        series.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_path", help="Directory or one 3D .bp*/.h5 series")
    parser.add_argument("--axes", default="xy,xz,yz", help="Comma-separated xy,xz,yz")
    parser.add_argument("--variable", default=None, help="Variable substring to plot")
    parser.add_argument("--all-variables", action="store_true")
    parser.add_argument("--slice-value", type=float, default=0.0)
    parser.add_argument("--out-dir", default="slices_frames")
    parser.add_argument("--nxny", type=int, default=1024)
    parser.add_argument("--method", choices=("linear", "nearest"), default="linear")
    parser.add_argument("--scale", choices=("log", "linear"), default="log")
    parser.add_argument("--vmin", type=float, default=None)
    parser.add_argument("--vmax", type=float, default=None)
    parser.add_argument("--cmap", default="plasma")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--no-movie", action="store_true")
    args = parser.parse_args()

    args.axes = [axis.strip().lower() for axis in args.axes.split(",") if axis.strip()]
    bad_axes = [axis for axis in args.axes if axis not in AXES]
    if bad_axes:
        print(f"ERROR: invalid axes: {', '.join(bad_axes)}")
        return 1

    opc.setup_matplotlib_style()

    data_path = os.path.abspath(os.path.expanduser(args.data_path))
    files = opc.gather_openpmd_series(data_path)
    if not files:
        print(f"ERROR: no openPMD series found in {data_path}")
        return 1

    os.makedirs(args.out_dir, exist_ok=True)
    frames = []
    for filepath in files:
        frames.extend(process_file(filepath, args, args.out_dir))

    if frames and not args.no_movie:
        for axis_name in args.axes:
            matching = [frame for frame in frames if f"slice_{axis_name}_" in os.path.basename(frame)]
            if matching:
                movie_path = os.path.join(args.out_dir, f"slices_{axis_name}.mp4")
                opc.movie_from_frames(sorted(matching), movie_path, fps=args.fps)

    return 0


if __name__ == "__main__":
    sys.exit(main())
