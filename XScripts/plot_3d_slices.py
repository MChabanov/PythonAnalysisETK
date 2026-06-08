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


# Manual plot recipes. Leave this list empty to use --variable normally.
# If this list is non-empty, --variable is ignored and only these recipes plot.
#
# Examples:
# MANUAL_PLOTS = [
#     {"label": "rho", "variables": {"rho": "hydrobasex_rho"}},
#     {
#         "label": "rho_over_press",
#         "variables": {"rho": "hydrobasex_rho", "press": "hydrobasex_press"},
#         "combine": lambda v: safe_divide(v["rho"], v["press"]),
#     },
# ]
MANUAL_PLOTS = []


def _safe_name(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def safe_divide(numerator, denominator):
    """Divide arrays while returning NaN where the denominator is zero/bad."""
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    valid = np.isfinite(numerator) & np.isfinite(denominator) & (denominator != 0.0)
    out = np.full_like(numerator, np.nan, dtype=np.float64)
    return np.divide(numerator, denominator, out=out, where=valid)


def announce_plot_selection(args):
    """Print whether manual recipes or --variable controls this run."""
    if MANUAL_PLOTS:
        labels = [recipe.get("label", "manual") for recipe in MANUAL_PLOTS]
        print(f"manual plot recipes active: {', '.join(labels)}")
        if args.variable:
            print(f"ignoring --variable {args.variable!r}")
    elif args.variable:
        print(f"manual plot recipes inactive; using --variable {args.variable!r}")
    else:
        print("manual plot recipes inactive; plotting all matching variables")


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


def _select_group(groups, variable_filter):
    """Choose one matched group for a manual source variable."""
    labels = sorted(groups)
    if not labels:
        raise ValueError(f"no match for {variable_filter!r}")
    if len(labels) == 1:
        return labels[0], groups[labels[0]]

    needle = str(variable_filter).lower()
    exact = [label for label in labels if label.lower() == needle]
    if len(exact) == 1:
        return exact[0], groups[exact[0]]

    contains = [label for label in labels if needle in label.lower()]
    if len(contains) == 1:
        return contains[0], groups[contains[0]]

    preview = ", ".join(labels[:8])
    if len(labels) > 8:
        preview += ", ..."
    raise ValueError(f"{variable_filter!r} matched multiple variables: {preview}")


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


def composite_slice(series, levels, axis_name, nxny, method, slice_value, extent=None):
    axes = AXES[axis_name]
    if extent is None:
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


def _source_slice_data(series, iteration, variable_filter, axis_name, args, target_extent=None):
    groups = _find_field_groups(iteration, variable_filter)
    label, levels = _select_group(groups, variable_filter)
    data, fast, slow = composite_slice(
        series,
        levels,
        axis_name,
        nxny=args.nxny,
        method=args.method,
        slice_value=args.slice_value,
        extent=target_extent,
    )
    extent = (fast.min(), fast.max(), slow.min(), slow.max())
    return label, data, fast, slow, extent


def _manual_slice_data(series, iteration, recipe, axis_name, args):
    variables = recipe.get("variables", {})
    if not variables:
        raise ValueError("manual recipe needs a non-empty 'variables' mapping")

    target_extent = None
    source_arrays = {}
    fast = slow = None
    source_labels = {}

    for alias, variable_filter in variables.items():
        label, data, fast, slow, source_extent = _source_slice_data(
            series, iteration, variable_filter, axis_name, args, target_extent
        )
        if target_extent is None:
            target_extent = source_extent
        source_arrays[alias] = data
        source_labels[alias] = label

    combine = recipe.get("combine")
    if combine is None:
        if len(source_arrays) != 1:
            raise ValueError("manual recipe with multiple variables needs 'combine'")
        data = next(iter(source_arrays.values()))
    else:
        data = combine(source_arrays)

    data = np.asarray(data, dtype=np.float64)
    if data.shape != next(iter(source_arrays.values())).shape:
        raise ValueError("manual recipe returned an array with the wrong shape")

    label = recipe.get("label") or next(iter(source_labels.values()))
    return label, data, fast, slow


def _color_limits(data, args):
    finite = data[np.isfinite(data)]
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


def create_figure():
    """Create a figure for one extracted 2D slice."""
    return plt.subplots(figsize=(7.5, 6.5), constrained_layout=True)


def panel_title(label, axis_name, iteration, time_cu):
    """Build a consistent 3D-slice panel title."""
    title = f"{label} {axis_name}, it={iteration}"
    if time_cu is not None:
        title += f", t={time_cu:.3e}"
    return title


def font_kwargs(args):
    """Return matplotlib keyword args for explicit text font size."""
    return {"fontsize": args.fontsize} if args.fontsize is not None else {}


def tick_kwargs(args, **base):
    """Return matplotlib tick keyword args with optional label font size."""
    if args.fontsize is not None:
        base["labelsize"] = args.fontsize
    return base


def set_offset_fontsize(axis, args):
    """Apply font size to matplotlib scientific-notation offset text."""
    if args.fontsize is None:
        return
    axis.xaxis.get_offset_text().set_fontsize(args.fontsize)
    axis.yaxis.get_offset_text().set_fontsize(args.fontsize)


def plot_panel(ax, data, fast, slow, label, axis_name, iteration, time_cu, args):
    """Draw and style one extracted 2D slice panel."""
    extent = (fast.min(), fast.max(), slow.min(), slow.max())
    vmin, vmax, norm = _color_limits(data, args)
    imshow_kwargs = {"norm": norm} if norm is not None else {"vmin": vmin, "vmax": vmax}

    im = ax.imshow(
        data,
        origin="lower",
        extent=extent,
        cmap=opc.setup_colormap(args.cmap),
        interpolation="none",
        **imshow_kwargs,
    )
    ax.set_xlabel(AXES[axis_name]["xlabel"], **font_kwargs(args))
    ax.set_ylabel(AXES[axis_name]["ylabel"], **font_kwargs(args))
    ax.set_title(panel_title(label, axis_name, iteration, time_cu), **font_kwargs(args))
    ax.set_aspect("equal")
    ax.tick_params(**tick_kwargs(args, direction="in", top=True, right=True))
    set_offset_fontsize(ax, args)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(label, **font_kwargs(args))
    cbar.ax.tick_params(**tick_kwargs(args, direction="in"))
    set_offset_fontsize(cbar.ax, args)
    return im


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
        frames = []
        file_name = _safe_name(os.path.basename(filepath.rstrip(os.sep)))

        if MANUAL_PLOTS:
            for recipe in MANUAL_PLOTS:
                for axis_name in args.axes:
                    label = recipe.get("label", "manual")
                    try:
                        label, data, fast, slow = _manual_slice_data(
                            series, itobj, recipe, axis_name, args
                        )
                    except Exception as exc:
                        print(f"  SKIP {label} {axis_name}: {exc}")
                        continue

                    fig, ax = create_figure()
                    plot_panel(ax, data, fast, slow, label, axis_name, it, time_cu, args)

                    frame_name = (
                        f"slice_{axis_name}_{_safe_name(label)}_{file_name}_it{it:08d}.png"
                    )
                    frame_path = os.path.join(out_dir, frame_name)
                    plt.savefig(frame_path, bbox_inches="tight", dpi=args.dpi)
                    plt.close(fig)
                    frames.append(frame_path)
                    print(f"wrote {frame_path}")
        else:
            groups = _find_field_groups(itobj, args.variable)
            if not groups:
                print(f"SKIP: no matching 3D meshes in {filepath}")
                return []

            labels = sorted(groups)
            if len(labels) > 1 and not args.all_variables:
                print(f"using {labels[0]} from {filepath}; pass --all-variables to plot all")
                labels = labels[:1]

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

                    fig, ax = create_figure()
                    plot_panel(ax, data, fast, slow, label, axis_name, it, time_cu, args)

                    frame_name = (
                        f"slice_{axis_name}_{_safe_name(label)}_{file_name}_it{it:08d}.png"
                    )
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
    parser.add_argument("--fontsize", type=float, default=None,
                        help="Use one font size for plot labels, titles, ticks, and colorbars")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--no-movie", action="store_true")
    args = parser.parse_args()

    args.axes = [axis.strip().lower() for axis in args.axes.split(",") if axis.strip()]
    bad_axes = [axis for axis in args.axes if axis not in AXES]
    if bad_axes:
        print(f"ERROR: invalid axes: {', '.join(bad_axes)}")
        return 1
    if args.fontsize is not None and args.fontsize <= 0:
        print("ERROR: --fontsize must be positive")
        return 1

    opc.setup_matplotlib_style()
    opc.apply_matplotlib_fontsize(args.fontsize)
    announce_plot_selection(args)

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
