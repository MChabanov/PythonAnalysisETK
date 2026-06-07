"""Shared utilities for openPMD 2D/3D visualization."""

import glob
import os
import re
import shutil
import numpy as np

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


def gather_openpmd_series(data_dir, pattern="*.bp*"):
    """Find all openPMD series files/directories in a directory.

    Handles both:
    - Single files: simulation.it00000000.bp5
    - ADIOS2 parallel directories: simulation.it00000000.bp5/ (with data.0, data.1, ...)
    """
    data_dir = os.path.abspath(os.path.expanduser(data_dir))
    files = []
    for ext in ("bp5", "bp", "bp4", "h5"):
        pattern_ext = os.path.join(data_dir, f"*.it*.{ext}")
        for f in glob.glob(pattern_ext):
            # Skip metadata and lock files
            if ".md." in os.path.basename(f) or f.endswith(".dir"):
                continue
            # Accept both files (single-file ADIOS2) and directories (parallel ADIOS2)
            if os.path.isfile(f) or os.path.isdir(f):
                files.append(f)
    return sorted(files)


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
