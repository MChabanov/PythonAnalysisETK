#!/usr/bin/env python3
"""Render movie frames from resampled 2D HDF5 data (see ../Resample).

This replaces the old ``movie_maker.py`` / ``movie_maker_parallel.py``. It
reads the per-variable HDF5 files written by the Resample pipeline, so there
is no SimDir scanning, no monolithic pickles, and no temp-file staging:

* **MPI-parallel over frames, any rank count.** Frames are distributed
  round-robin (``frames[rank::size]``); each rank reads only the 2D slices it
  needs straight from the (lazily-sliced) HDF5 files and writes its own PNGs.
  No gather, no temp files, bounded memory.
* **Serial fallback.** If mpi4py is not installed the script runs on one
  process - convenient for laptops/workstations.
* **Optional movie assembly** with ffmpeg at the end.

The file is organised in three sections:

    1. SETTINGS  - input files, units, frame selection, output, figure layout
    2. PLOTTING  - all matplotlib code for one frame (edit for your figure)
    3. MACHINERY - HDF5 access, MPI loop, ffmpeg (no need to edit)

Usage
-----
    mpirun -n <N> python make_frames.py
    python make_frames.py              # serial
"""

import os
import re
import subprocess
import sys

import numpy as np

import matplotlib

matplotlib.use("Agg")  # headless rendering; must be set before pyplot import

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

import h5py

# ===========================================================================
# 1. SETTINGS
# ===========================================================================

# --- Input: short name -> resampled HDF5 file (output of ../Resample) ------
# Example dataset produced by resample_rho_b.yaml in this folder. For a
# multi-field figure (e.g. Pi/p) add entries like "pi": .../pi__<label>.h5
# and reference them in the panel's data lambda; for a multi-simulation
# comparison add per-simulation names ("rho_tnt", "rho_low", ...).

DATA_DIR = "./data"

FIELDS = {
    "rho": DATA_DIR + "/rho_b__example.h5",
}

# --- Unit conversions -------------------------------------------------------

MSUN_TO_KM = 1.4766696910334391   # coordinate scale: code units -> km
MILLIS = 203.01930744592713       # 1 ms in code time units
T_MERGE = 0.0                     # subtracted from times in panel titles

# --- Frame selection (indices into the stored iterations, aligned by index) -

FRAME_START = 0
FRAME_STOP = None                 # None = all available frames
FRAME_STRIDE = 1

# --- Output -----------------------------------------------------------------

OUT_DIR = "./frames"
OUT_PREFIX = "rho_b"
ZFILL = 4                         # frame number padding in file names
MAKE_MOVIE = True                 # assemble OUT_DIR/MOVIE_NAME with ffmpeg
FRAMERATE = 25
MOVIE_NAME = "rho_b.mp4"

# --- Figure layout / global matplotlib properties ---------------------------

FIG_ROWS, FIG_COLS = 1, 1         # grid of panels (see PANELS below)
FIGSIZE = (7.5, 6.0)
DPI = 200
FONT_SIZE = 15
SUPTITLE = None
XLABEL, YLABEL = r"$x~[\mathrm{km}]$", r"$y~[\mathrm{km}]$"
XLIM = (-73.8, 73.8)              # in km (after MSUN_TO_KM scaling)
YLIM = (-73.8, 73.8)
SUBPLOTS_ADJUST = dict(left=0.12, right=0.95, bottom=0.085, top=0.95,
                       wspace=0.25, hspace=0.20)
RCPARAMS = {                      # any extra rcParams you want applied
    # "font.family": "serif",
}

# --- Contour levels shared by panels ----------------------------------------

RHO_LEVELS = [1.828929564024961e-06, 0.00073157182561, 0.0013]
RHO_COLORS = ["black", "black", "black"]
RHO_STYLES = ["dashed", "solid", "dotted"]
RHO_LW = 1.3

# --- Panels ------------------------------------------------------------------
# One entry per subplot, filled row-major into the FIG_ROWS x FIG_COLS grid.
#   data(f, i)  : 2D array to show; f.data(name, i) reads field `name`
#                 at frame i, so any numpy expression works.
#   ref         : field whose coordinates/times label this panel.

PANELS = [
    {
        "title": r"$\rho;~t=$ {t:.3f} ms",
        "ref": "rho",
        "data": lambda f, i: f.data("rho", i),
        "cmap": "magma",
        "norm": colors.LogNorm(vmin=1e-10, vmax=2e-3),
        "time_offset": T_MERGE,
        "colorbar": True,
        "contours": [
            {"data": lambda f, i: f.data("rho", i), "ref": "rho",
             "levels": RHO_LEVELS, "colors": RHO_COLORS,
             "linestyles": RHO_STYLES, "linewidths": RHO_LW},
        ],
    },
]

# ===========================================================================
# 2. PLOTTING - all matplotlib code for ONE frame. Edit for your figure.
# ===========================================================================

# Substitute only the known {t}/{t_raw}/{it}/{idx} placeholders (with optional
# format specs like {t:.3f}) so LaTeX braces in titles survive untouched.
_TITLE_TOKEN = re.compile(r"\{(t|t_raw|it|idx)(:[^{}]*)?\}")


def format_title(template, **values):
    def _sub(match):
        key, spec = match.group(1), match.group(2)
        return format(values[key], spec[1:]) if spec else str(values[key])
    return _TITLE_TOKEN.sub(_sub, template)


def render_frame(f, idx, seq):
    """Render frame `idx` of the data and save it as output number `seq`."""
    fig, axes = plt.subplots(FIG_ROWS, FIG_COLS, figsize=FIGSIZE,
                             squeeze=False)
    fig.subplots_adjust(**SUBPLOTS_ADJUST)
    flat = axes.ravel()

    for panel, ax in zip(PANELS, flat):
        ref = panel["ref"]
        x, y = f.coords(ref)
        X, Y = np.meshgrid(x * MSUN_TO_KM, y * MSUN_TO_KM, indexing="ij")

        mesh = ax.pcolormesh(
            X, Y, panel["data"](f, idx),
            norm=panel.get("norm"), cmap=panel.get("cmap", "viridis"),
            shading="nearest", rasterized=True,
        )

        for cont in panel.get("contours", []):
            cx, cy = f.coords(cont["ref"])
            CX, CY = np.meshgrid(cx * MSUN_TO_KM, cy * MSUN_TO_KM,
                                 indexing="ij")
            ax.contour(CX, CY, cont["data"](f, idx), sorted(cont["levels"]),
                       colors=cont.get("colors", "black"),
                       linestyles=cont.get("linestyles", "solid"),
                       linewidths=cont.get("linewidths", 1.0))

        # Title placeholders: {t} = (time - time_offset)/MILLIS, {t_raw},
        # {it} = iteration number, {idx} = frame index. Format specs work
        # ({t:.3f}); all other braces (LaTeX!) are left untouched.
        t_raw = f.time(ref, idx)
        ax.set_title(
            format_title(panel["title"],
                         t=(t_raw - panel.get("time_offset", 0.0)) / MILLIS,
                         t_raw=t_raw, it=f.iteration(ref, idx), idx=idx),
            fontsize=FONT_SIZE,
        )

        ax.set_xlim(*XLIM)
        ax.set_ylim(*YLIM)
        ax.tick_params(labelsize=FONT_SIZE)

        if panel.get("colorbar", True):
            cax = make_axes_locatable(ax).append_axes("right", size="5%",
                                                      pad=0.05)
            fig.colorbar(mesh, cax=cax).ax.tick_params(labelsize=FONT_SIZE)

    # Hide grid slots without a panel; label only the outer edges.
    for ax in flat[len(PANELS):]:
        ax.set_visible(False)
    for ax in axes[-1, :]:
        ax.set_xlabel(XLABEL, fontsize=FONT_SIZE)
    for ax in axes[:, 0]:
        ax.set_ylabel(YLABEL, fontsize=FONT_SIZE)

    if SUPTITLE:
        fig.suptitle(SUPTITLE, fontsize=FONT_SIZE, y=0.99)

    out_path = os.path.join(OUT_DIR,
                            OUT_PREFIX + "_" + str(seq).zfill(ZFILL) + ".png")
    fig.savefig(out_path, bbox_inches="tight", dpi=DPI)
    plt.close(fig)
    return out_path


# ===========================================================================
# 3. MACHINERY - HDF5 access, MPI frame loop, ffmpeg. No need to edit.
# ===========================================================================

# Optional MPI: fall back to a single process when mpi4py is unavailable.
try:
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
except ImportError:
    comm, rank, size = None, 0, 1


def log(message):
    if rank == 0:
        print(message, flush=True)


def abort(message):
    if rank == 0:
        print("ERROR: " + message, file=sys.stderr, flush=True)
    if comm is not None:
        comm.Abort(1)
    sys.exit(1)


class Fields:
    """Read-only access to the resampled per-variable HDF5 files.

    Slices are read lazily, so a rank only ever holds the 2D arrays of the
    frame it is currently rendering.
    """

    def __init__(self, mapping):
        missing = [p for p in mapping.values() if not os.path.exists(p)]
        if missing:
            abort("field file(s) not found:\n  " + "\n  ".join(missing))
        self.h5 = {name: h5py.File(path, "r") for name, path in mapping.items()}

    def n_frames(self):
        """Frames are aligned by index; the shortest field sets the count."""
        return min(f["iterations"].shape[0] for f in self.h5.values())

    def data(self, name, idx):
        return self.h5[name]["data"][idx]

    def coords(self, name):
        return self.h5[name]["x"][:], self.h5[name]["y"][:]

    def time(self, name, idx):
        return float(self.h5[name]["times"][idx])

    def iteration(self, name, idx):
        return int(self.h5[name]["iterations"][idx])


def assemble_movie():
    """Run ffmpeg over the rendered frames (rank 0, after the barrier)."""
    pattern = os.path.join(OUT_DIR, OUT_PREFIX + "_%0" + str(ZFILL) + "d.png")
    movie_path = os.path.join(OUT_DIR, MOVIE_NAME)
    cmd = ["ffmpeg", "-y", "-framerate", str(FRAMERATE), "-i", pattern,
           "-c:v", "libx264", "-pix_fmt", "yuv420p",
           # bbox_inches="tight" yields odd PNG sizes, but libx264/yuv420p
           # needs even dimensions: rescale down by at most 1 px.
           "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
           # explicit output rate (guards against variable-frame-rate output)
           "-r", str(FRAMERATE),
           movie_path]
    log("Assembling movie: " + " ".join(cmd))
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        log("Movie written to %s" % movie_path)
    except FileNotFoundError:
        log("ffmpeg not found - assemble manually with:\n  " + " ".join(cmd))
    except subprocess.CalledProcessError as exc:
        log("ffmpeg failed:\n" + exc.stderr.decode(errors="replace")[-2000:])


def main():
    plt.rcParams.update(RCPARAMS)

    if rank == 0:
        os.makedirs(OUT_DIR, exist_ok=True)
    if comm is not None:
        comm.Barrier()  # output dir must exist before anyone renders

    fields = Fields(FIELDS)

    stop = FRAME_STOP if FRAME_STOP is not None else fields.n_frames()
    frame_indices = list(range(FRAME_START, stop, FRAME_STRIDE))
    if not frame_indices:
        abort("frame selection is empty (check FRAME_START/STOP/STRIDE)")

    # `seq` numbers the output files contiguously (ffmpeg-friendly) even when
    # striding; `idx` is the index into the stored data.
    todo = list(enumerate(frame_indices))[rank::size]

    log("Rendering %d frame(s) on %d rank(s) -> %s"
        % (len(frame_indices), size, OUT_DIR))

    for seq, idx in todo:
        out_path = render_frame(fields, idx, seq)
        print("[rank %d] %s" % (rank, out_path), flush=True)

    if comm is not None:
        comm.Barrier()

    if rank == 0 and MAKE_MOVIE:
        assemble_movie()

    log("Done: %d frame(s) rendered." % len(frame_indices))


if __name__ == "__main__":
    main()
