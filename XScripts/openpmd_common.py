"""Shared utilities for openPMD 2D/3D visualization."""

import glob
import os
import re
import shutil
import numpy as np

OPENPMD_EXTENSIONS = ("bp5", "bp", "bp4", "h5")
PLANE_TAG_RE = re.compile(
    r"(?P<tag>(?P<plane>xy|xz|yz)_(?P<axis>[xyz])_"
    r"(?P<sign>pos|neg)(?P<int_digits>\d{1,8})(?:p(?P<frac_digits>\d{0,8}))?)"
)


def setup_matplotlib_style(use_tex=None):
    """Configure matplotlib for publication-quality plots."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    if use_tex is None:
        use_tex = shutil.which("latex") is not None

    plt.rcParams.update({
        "text.usetex": bool(use_tex),
        "font.family": "serif" if use_tex else "STIXGeneral",
        "mathtext.fontset": "stix",
        "font.size": 10,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.labelpad": 0.5,
        "ytick.major.pad": 0.5,
        "ytick.minor.pad": 0.5,
    })
    return plt


def is_openpmd_series_path(path):
    """Return True when path itself looks like an openPMD series file/directory."""
    path = os.path.abspath(os.path.expanduser(path)).rstrip(os.sep)
    basename = os.path.basename(path)
    if ".md." in basename or basename.endswith(".dir"):
        return False
    if not (os.path.isfile(path) or os.path.isdir(path)):
        return False
    return any(basename.endswith(f".{ext}") for ext in OPENPMD_EXTENSIONS)


def gather_openpmd_series(data_dir, pattern="*.bp*"):
    """Find openPMD series files/directories.

    Handles both:
    - Single files: simulation.it00000000.bp5
    - ADIOS2 parallel directories: simulation.it00000000.bp5/ (with data.0, data.1, ...)
    - Direct input paths to one openPMD series
    """
    data_dir = os.path.abspath(os.path.expanduser(data_dir))
    if is_openpmd_series_path(data_dir):
        return [data_dir.rstrip(os.sep)]

    if not os.path.isdir(data_dir):
        return []

    files = []
    for ext in OPENPMD_EXTENSIONS:
        pattern_ext = os.path.join(data_dir, f"*.it*.{ext}")
        for f in glob.glob(pattern_ext):
            # Skip metadata and lock files
            if ".md." in os.path.basename(f) or f.endswith(".dir"):
                continue
            # Accept both files (single-file ADIOS2) and directories (parallel ADIOS2)
            if os.path.isfile(f) or os.path.isdir(f):
                files.append(f)
    return sorted(files)


def parse_plane_tag(text):
    """Parse a PlanesX tag like xy_z_pos0012p500 from text.

    The integer and fractional digit counts are intentionally variable because
    they are controlled by planes_int_precision and planes_frac_precision.
    """
    match = PLANE_TAG_RE.search(os.path.basename(str(text)))
    if not match:
        match = PLANE_TAG_RE.search(str(text))
    if not match:
        return None

    frac_digits = match.group("frac_digits") or ""
    magnitude = float(int(match.group("int_digits")))
    if frac_digits:
        magnitude += int(frac_digits) / (10 ** len(frac_digits))
    if match.group("sign") == "neg":
        magnitude *= -1.0

    return {
        "tag": match.group("tag"),
        "plane": match.group("plane"),
        "normal_axis": match.group("axis"),
        "sign": match.group("sign"),
        "int_digits": match.group("int_digits"),
        "frac_digits": frac_digits,
        "elevation": magnitude,
        "start": match.start("tag"),
        "end": match.end("tag"),
    }


def parse_amr_mesh_name(mesh_name):
    """Parse AMR mesh names with optional PlanesX prefix and centering suffix.

    Examples:
    - xy_z_pos0000p000_hydrobasex_rho_patch00_lev09
    - xz_y_neg0003p000_hydrobasex_bvec_cv_patch00_lev02
    - hydrobasex_rho_patch00_lev09
    """
    patterns = (
        r"^(?P<prefix>.+)_patch0*(?P<patch>\d+)_lev0*(?P<level>\d+)$",
        r"^(?P<prefix>.+)_lev0*(?P<level>\d+)_patch0*(?P<patch>\d+)$",
    )
    match = None
    for pattern in patterns:
        match = re.match(pattern, mesh_name)
        if match:
            break
    if not match:
        return None

    prefix = match.group("prefix")
    plane = parse_plane_tag(prefix)
    group = prefix
    if plane and plane["start"] == 0:
        group = prefix[plane["end"]:]
        if group.startswith("_"):
            group = group[1:]

    centering = None
    for suffix in ("_cv", "_vc"):
        if group.endswith(suffix):
            centering = suffix[1:]
            group = group[:-len(suffix)]
            break

    return {
        "mesh_name": mesh_name,
        "group": group,
        "centering": centering,
        "level": int(match.group("level")),
        "patch": int(match.group("patch")),
        "plane": plane,
    }


def component_label(mesh_info, component_name, component_count=1):
    """Return the user-facing variable label for a mesh component."""
    group = mesh_info["group"] if mesh_info else ""
    comp = str(component_name)
    if comp and comp.lower() not in {"scalar", "value", "0"}:
        return comp
    return group or comp


def mesh_matches_variable(mesh_name, component_name, variable_pattern, mesh_info=None):
    """Match a user variable substring against mesh, group, tag, and component."""
    if not variable_pattern:
        return True

    needle = variable_pattern.lower()
    fields = [mesh_name, component_name]
    if mesh_info:
        fields.append(mesh_info.get("group") or "")
        plane = mesh_info.get("plane")
        if plane:
            fields.extend([plane["tag"], plane["plane"], plane["normal_axis"]])

    return any(needle in str(field).lower() for field in fields)


def filter_series_by_plane(files, tag=None, plane=None, normal_axis=None, elevation=None):
    """Filter openPMD series paths by parsed PlanesX plane tag metadata."""
    if not any(value is not None for value in (tag, plane, normal_axis, elevation)):
        return files

    out = []
    for path in files:
        info = parse_plane_tag(os.path.basename(path.rstrip(os.sep)))
        if not info:
            continue
        if tag is not None and info["tag"] != tag:
            continue
        if plane is not None and info["plane"] != plane:
            continue
        if normal_axis is not None and info["normal_axis"] != normal_axis:
            continue
        if elevation is not None and not np.isclose(info["elevation"], elevation):
            continue
        out.append(path)
    return out


def parse_iteration_number(filepath):
    """Extract iteration number from openPMD filename like 'name.it00012345.bp5'."""
    basename = os.path.basename(filepath)
    # Match patterns like .it12345.bp5 or .it12345.bp
    match = re.search(r"\.it(\d+)\.", basename)
    return int(match.group(1)) if match else 0


def get_openpmd_time(series, iteration):
    """Extract simulation time from an openPMD iteration (code units)."""
    try:
        itobj = series.iterations[iteration]
        # Try standard attributes in order
        for attr_getter in (lambda: itobj.time,
                           lambda: itobj.get_attribute("time")):
            try:
                return float(attr_getter())
            except (AttributeError, TypeError, KeyError):
                pass

        # Fallback: search in first mesh
        for mesh_name in itobj.meshes:
            mesh = itobj.meshes[mesh_name]
            for getter in (lambda: mesh.get_attribute("time"),
                          lambda: next(iter(mesh.values())).get_attribute("time")):
                try:
                    return float(getter())
                except (AttributeError, TypeError, KeyError, StopIteration):
                    pass
    except (AttributeError, TypeError, KeyError):
        pass
    return None


def setup_colormap(cmap_name="plasma", vmin=None, vmax=None):
    """Create a colormap with proper handling of bad/underflow values."""
    from matplotlib import cm
    cmap = cm.get_cmap(cmap_name).copy()
    cmap.set_bad(color="#0b0e2c", alpha=1.0)     # NaNs → dark
    cmap.set_under(color="#0b0e2c", alpha=1.0)   # values < vmin → dark
    return cmap


class OpenPMDField:
    """Helper to access openPMD field data with coordinate metadata."""

    def __init__(self, mesh, component_name):
        """Initialize with a mesh and component name."""
        self.mesh = mesh
        self.comp_name = component_name
        self.record = mesh[component_name]

        # Get grid metadata
        self.spacing = np.array(mesh.get_attribute("gridSpacing"))
        self.offset = np.array(mesh.get_attribute("gridGlobalOffset"))
        self.position = np.array(self.record.position)
        self.shape = np.array([int(s) for s in self.record.shape])

    def get_axis_coords(self, axis):
        """Get 1D coordinate array for a record axis."""
        n = self.shape[axis]
        return self.offset[axis] + (np.arange(n) + self.position[axis]) * self.spacing[axis]

    def iter_chunk_extents(self):
        """Yield clipped written chunk (offset, extent) pairs without loading data."""
        for ch in self.record.available_chunks():
            off = [int(v) for v in ch.offset]
            ext = [int(v) for v in ch.extent]

            # Clip cell-centred fill row/column: declared extent is vertex count N,
            # but only N-1 cells were written. No-op for ADIOS2 box chunks,
            # essential for HDF5 which reports whole extent as one chunk.
            for d in range(len(ext)):
                if self.position[d] != 0.0:  # cell-centred on this axis
                    valid = self.shape[d] - 1
                    ext[d] = min(ext[d], max(0, valid - off[d]))

            if any(e <= 0 for e in ext):
                continue

            yield off, ext

    def read_chunks(self, series=None):
        """Yield (data, offset, extent) for each written chunk (ADIOS2/HDF5 safe).

        series: openpmd_api.Series instance (needed to flush reads).
                Pass None if not available (data may be delayed).
        """
        for off, ext in self.iter_chunk_extents():
            data = self.record.load_chunk(off, ext)
            if series is not None:
                series.flush()
            data = np.asarray(data).reshape(tuple(ext))

            yield data, off, ext

    def read_full(self, series=None):
        """Assemble the written chunk bounding box into one array.

        Warning: this is still unsafe for large sparse AMR data if the written
        bounding box is huge. Prefer iterating with read_chunks() whenever a
        plot or diagnostic can operate patch-by-patch.
        """
        chunks = list(self.read_chunks(series))
        if not chunks:
            return np.empty(tuple(0 for _ in self.shape), dtype=np.float64)

        # Find bounding box of all chunks
        all_offsets = np.array([off for _, off, _ in chunks])
        all_extents = np.array([ext for _, _, ext in chunks])
        min_off = all_offsets.min(axis=0)
        max_pos = (all_offsets + all_extents).max(axis=0)
        bounds = max_pos - min_off

        result = np.full(tuple(bounds), np.nan, dtype=np.float64)
        for data, off, ext in chunks:
            slices = tuple(
                slice(off[d] - min_off[d], off[d] - min_off[d] + ext[d])
                for d in range(len(ext))
            )
            result[slices] = data

        return result


class Canvas2D:
    """Uniform 2D canvas for compositing AMR data."""

    def __init__(self, extent_xy, nxny):
        """extent_xy = (xmin, xmax, ymin, ymax), nxny = resolution per axis."""
        xmin, xmax, ymin, ymax = extent_xy
        self.x = np.linspace(xmin, xmax, nxny)
        self.y = np.linspace(ymin, ymax, nxny)
        self.data = np.full((nxny, nxny), np.nan, dtype=np.float64)

    def add_patch(self, data_2d, x_coords, y_coords, method="nearest", fill_edges=True):
        """Composite a 2D patch onto the canvas.

        data_2d: 2D array with shape (len(y_coords), len(x_coords))
        x_coords, y_coords: 1D coordinate arrays
        method: 'nearest' or 'linear' (requires scipy)
        fill_edges: if True, erode the patch before writing to avoid seams
        """
        # Find canvas region overlapping this patch
        j0 = np.searchsorted(self.y, max(y_coords.min(), self.y.min()), side="left")
        j1 = np.searchsorted(self.y, min(y_coords.max(), self.y.max()), side="right")
        i0 = np.searchsorted(self.x, max(x_coords.min(), self.x.min()), side="left")
        i1 = np.searchsorted(self.x, min(x_coords.max(), self.x.max()), side="right")

        if j1 <= j0 or i1 <= i0:
            return  # Patch outside canvas

        sub_y = self.y[j0:j1]
        sub_x = self.x[i0:i1]

        if method == "linear":
            try:
                from scipy.interpolate import RegularGridInterpolator
                valid = np.where(np.isfinite(data_2d), data_2d, np.nan)
                f = RegularGridInterpolator((y_coords, x_coords), valid,
                                           bounds_error=False, fill_value=np.nan)
                YY, XX = np.meshgrid(sub_y, sub_x, indexing="ij")
                interp_data = f(np.stack([YY, XX], axis=-1))
            except ImportError:
                method = "nearest"

        if method == "nearest":
            jj = np.clip(np.searchsorted(y_coords, sub_y), 0, len(y_coords) - 1)
            ii = np.clip(np.searchsorted(x_coords, sub_x), 0, len(x_coords) - 1)
            interp_data = data_2d[np.ix_(jj, ii)]

        # Write to canvas (skip NaN values)
        valid_mask = np.isfinite(interp_data)
        block = self.data[j0:j1, i0:i1]
        block[valid_mask] = interp_data[valid_mask]
        self.data[j0:j1, i0:i1] = block


def movie_from_frames(frame_list, output_path, fps=12):
    """Assemble frames into MP4 or GIF."""
    try:
        import imageio.v2 as imageio
    except ImportError:
        print("ERROR: imageio not available; install with: pip install imageio imageio-ffmpeg")
        return False

    if not frame_list:
        print("No frames to assemble")
        return False

    try:
        imgs = [imageio.imread(p) for p in frame_list]
    except Exception as e:
        print(f"ERROR reading frames: {e}")
        return False

    try:
        imageio.mimsave(output_path, imgs, fps=fps)
        print(f"✓ Saved: {output_path}")
        return True
    except Exception as e:
        print(f"⚠ Failed to write {output_path} ({e}). Trying GIF fallback...")
        try:
            gif_path = output_path.rsplit(".", 1)[0] + ".gif"
            imageio.mimsave(gif_path, imgs, fps=fps)
            print(f"✓ Saved GIF: {gif_path}")
            return True
        except Exception as e2:
            print(f"ERROR: Both MP4 and GIF failed: {e2}")
            return False
