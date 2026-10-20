#!/usr/bin/env python3
"""Flux integrals over a spherical shell, split by where each point's field line ends.

    mpirun -n <N> python shell_integrals.py config_shell.yaml

For every 3D output iteration: take the grid points of the shell
``r_in <= |x - centre| < r_out`` straight from the AMR output (no resampling;
``shell_cells.py`` for the volumes and integrands), trace the field line
through each of them both ways (``../FieldLines``), and sum the three flux
integrals of ``~/in_plane_flux_tube_diagnostics.md`` - Phi_phi, Phi_r, Phi_z -
and the volume separately for each class of line: ``BH1-exit`` (open field
threading BH1's horizon), ``BH1-BH1`` (closed loops on it), ``BH1-BH2``,
``exit-exit`` (field reaching no horizon inside the tracing box), ``cap-*``
(undetermined at the length cap) and so on. Summing the classes gives the
unmasked integral; leaving out those with a horizon end removes the holes'
magnetospheres. ``integrals_tools.py`` reads the output and forms such groups.

Work is split by iteration over the MPI ranks; each rank holds one
iteration's field in the tracing box (a few GB) and traces with numba threads
(``tracer.threads`` or NUMBA_NUM_THREADS). See ``README.md`` and
``config_example_shell.yaml``.

Output, in ``output_dir``, one file per quantity like the resampling pipeline
(``Phi_phi``, ``Phi_r``, ``Phi_z``, ``volume`` ``__<label>.h5``):

    data        (n_iter, n_hscale, n_class)          the integral per class
    horizon_scales (n_hscale,)                       horizon sizes, see below
    iterations, times
    shell_centre, shell_pole, bh1_position, bh2_position   (n_iter, 3)
    profile     (n_iter, n_hscale, n_class, n_bin)   Phi_z only: flux through
    z_edges     (n_bin + 1,)                         the planes zeta = const

with the class names (attribute ``classes``), definitions and every setting as
attributes; ``tracing__<label>.h5`` with per-iteration diagnostics (points and
line lengths per class, volume check, timings, horizons); and per iteration
``points__<label>__it<it>.h5``: every point's position, volume, level and line
ends, a few MB (plus lengths, end points and integrands for the iterations in
``save_points``).

The classes are counted for each of ``horizon_scales`` = ``[horizon_scale] +
extra_horizon_scales``: the lines are traced once, with the first, and each
records where it first enters the enlarged horizon spheroids - which is where
it would have ended with them (``tracer.trace``, exact).

Restarting: an iteration whose ``points__`` file holds ends traced with the
same settings from the same points is not traced again - its integrals are
recomputed from the stored ends (about a minute instead of ten). So a run that
stopped is simply submitted again; so is a run with new integrands or masks.
"""

import argparse
import json
import os
import sys
import time

import h5py
import numpy as np
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "FieldLines"))
sys.path.insert(0, os.path.join(_HERE, "..", "Resample"))

from resample_common import (abort, comm, log, rank, size,  # noqa: E402
                             _acquire_cold_simdir, _query_serial)
from resample_3d_data_postcactus import Postcactus3DBackend   # noqa: E402
from resample_sphere_data_postcactus import frame_spec         # noqa: E402
import plane_motion                                            # noqa: E402
from amr_blocks import Snapshot                                # noqa: E402
from amr_field import AMRVectorField                           # noqa: E402
from horizons import Horizons                                  # noqa: E402
import tracer                                                  # noqa: E402
import shell_cells                                             # noqa: E402

QUANTITIES = ("Phi_phi", "Phi_r", "Phi_z", "volume")

DEFINITIONS = {
    "Phi_phi": "1/(2 pi) int |calB . phi_hat| / R d^3x: mean unsigned flux "
               "through the meridional half-planes phi = const",
    "Phi_r": "1/(r_out - r_in) int |calB . r_hat| d^3x: mean unsigned flux "
             "through the spheres r = const",
    "Phi_z": "1/(2 h) int |calB . pole| d^3x, h = r_out: mean unsigned flux "
             "through the planes zeta = const that cross the shell",
    "volume": "int d^3x: coordinate volume",
}

_SHELL_KEYS = {"centre", "pole", "phi_zero", "r_in", "r_out", "z_bin",
               "axis_cells", "subsamples", "file", "time_offset"}
_TRACER_KEYS = {"box_half_width", "step", "max_length", "lookup_cell",
                "horizon_scale", "extra_horizon_scales", "batches", "threads",
                "margin"}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_config(path):
    """Read and check the YAML config on rank 0; fill in defaults."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    missing = [k for k in ("simdir", "label", "shell") if cfg.get(k) is None]
    if missing:
        abort("config is missing required key(s): " + ", ".join(missing))
    cfg.setdefault("output_dir", "./integrals")
    for key, val in (("iteration_stride", 1), ("iteration_min", None),
                     ("iteration_max", None), ("parallel_scan", True),
                     ("scan_max_depth", 8), ("share_tocs", True),
                     ("toc_cache", None), ("save_points", [])):
        cfg.setdefault(key, val)
    sp = cfg.get("simdir_pickle") or {}
    sp.setdefault("pickled", False)
    sp.setdefault("path", None)
    cfg["simdir_pickle"] = sp
    if sp["pickled"] and not (sp["path"] and os.path.isfile(sp["path"])):
        abort("simdir_pickle.pickled is yes but %r does not exist" % sp["path"])
    se = cfg.get("simdir_exclude") or {}
    for key in ("dirs", "files"):
        val = se.get(key) or []
        se[key] = [str(v) for v in ([val] if isinstance(val, str) else val)]
    cfg["simdir_exclude"] = se

    shell = dict(cfg["shell"])
    unknown = sorted(set(shell) - _SHELL_KEYS)
    if unknown:
        abort("unknown shell key(s): %s" % ", ".join(unknown))
    for key in ("r_in", "r_out"):
        if shell.get(key) is None:
            abort("shell.%s is required" % key)
    shell["r_in"], shell["r_out"] = float(shell["r_in"]), float(shell["r_out"])
    if not 0.0 <= shell["r_in"] < shell["r_out"]:
        abort("need 0 <= shell.r_in < shell.r_out")
    shell.setdefault("centre", "bh1")
    shell.setdefault("pole", "z")
    shell.setdefault("phi_zero", None)
    shell.setdefault("z_bin", 0.25)
    shell.setdefault("axis_cells", 4.0)
    shell.setdefault("subsamples", 8)
    shell.setdefault("file", "auto")
    shell.setdefault("time_offset", "auto")
    n_bin = int(round(2.0 * shell["r_out"] / float(shell["z_bin"])))
    shell["n_bin"] = max(1, n_bin)
    cfg["shell"] = shell

    fields = cfg.get("fields") or {}
    fields.setdefault("B", ["Bx", "By", "Bz"])
    fields.setdefault("metric", ["gxx", "gxy", "gxz", "gyy", "gyz", "gzz"])
    if len(fields["B"]) != 3 or len(fields["metric"]) != 6:
        abort("fields.B needs 3 names and fields.metric 6 (xx xy xz yy yz zz)")
    cfg["fields"] = fields
    # What the iteration query of resample_common looks for.
    cfg["variables"] = list(fields["B"]) + list(fields["metric"])

    tr = dict(cfg.get("tracer") or {})
    unknown = sorted(set(tr) - _TRACER_KEYS)
    if unknown:
        abort("unknown tracer key(s): %s" % ", ".join(unknown))
    tr.setdefault("box_half_width", 12.0)
    tr.setdefault("step", 0.5)
    tr.setdefault("max_length", 200.0)
    tr.setdefault("lookup_cell", 0.5)
    tr.setdefault("horizon_scale", 1.0)
    tr.setdefault("batches", 20)
    tr.setdefault("threads", None)
    tr.setdefault("margin", 3)
    tr.setdefault("extra_horizon_scales", [])
    tr["extra_horizon_scales"] = [float(x) for x in
                                  (tr["extra_horizon_scales"] or [])]
    if any(x <= float(tr["horizon_scale"]) for x in tr["extra_horizon_scales"]):
        abort("tracer.extra_horizon_scales must all exceed horizon_scale")
    if float(tr["box_half_width"]) <= shell["r_out"]:
        abort("tracer.box_half_width must exceed shell.r_out")
    cfg["tracer"] = tr
    cfg["horizon_scales"] = ([float(tr["horizon_scale"])]
                             + tr["extra_horizon_scales"])
    cfg["save_points"] = [int(i) for i in (cfg["save_points"] or [])]
    return cfg


# ---------------------------------------------------------------------------
# Per-iteration tables (rank 0)
# ---------------------------------------------------------------------------

def build_tables(cfg, iterations, times, sim):
    """Shell frame and horizons at every iteration, from the trajectory."""
    shell = cfg["shell"]
    sphere = {"centre": shell["centre"], "pole": shell["pole"],
              "file": shell["file"], "time_offset": shell["time_offset"]}
    if shell["phi_zero"] is not None:
        sphere["phi_zero"] = shell["phi_zero"]
    plane = frame_spec(sphere)
    log("Shell frame:")
    try:
        table = plane_motion.build_frame_table(plane, iterations, times, sim=sim,
                                               log=log, in_plane=False)
        traj = plane_motion.Trajectory(plane_motion.resolve_trajectory_path(
            shell["file"], sim, log))
        offset = plane_motion.resolve_time_offset(shell["time_offset"], sim,
                                                  log)
    except ValueError as exc:
        abort("shell: %s" % exc)
    state = traj.at(np.asarray(times) + offset)
    if state.spin1 is None or state.spin2 is None:
        abort("the trajectory %s has no spins (a1x ..): the horizons need them"
              % traj.path)
    hz = []
    for i in range(len(iterations)):
        h = Horizons.kerr_schild(
            [state.pos1[i], state.pos2[i]], [state.spin1[i], state.spin2[i]],
            [state.mass1[i], state.mass2[i]],
            scale=float(cfg["tracer"]["horizon_scale"]))
        hz.append((h.centres, h.axes, h.a_eq, h.a_pol))
    log("Horizons (analytic Kerr-Schild, x %g), first output:"
        % cfg["tracer"]["horizon_scale"])
    for row in Horizons(*hz[0], names=("BH1", "BH2")).describe():
        log("  " + row)
    return {
        "iterations": np.asarray(iterations, dtype=np.int64),
        "times": np.asarray(times, dtype=float),
        "centre": table.origin, "pole": table.n_hat, "phi_zero": table.u_hat,
        "bh1_position": state.pos1, "bh2_position": state.pos2,
        "horizons": hz,
    }


# ---------------------------------------------------------------------------
# One iteration
# ---------------------------------------------------------------------------

def horizons_at(tables, row, scale=1.0):
    """The :class:`Horizons` of one iteration, semi-axes times ``scale``."""
    centres, axes, a_eq, a_pol = tables["horizons"][row]
    return Horizons(centres, axes, scale * a_eq, scale * a_pol,
                    names=("BH1", "BH2"))


class ShellPoints(object):
    """The shell's points of one iteration with their volumes and integrands."""

    def __init__(self, cfg, snap, centre, pole):
        shell, fields = cfg["shell"], cfg["fields"]
        sel = shell_cells.select_shell(snap.components, centre, shell["r_in"],
                                       shell["r_out"], int(shell["subsamples"]))
        (self.points, self.weight, self.level,
         self.spacing) = shell_cells.selection_points(sel)
        vals = shell_cells.read_selected(snap.read, sel,
                                         fields["B"] + fields["metric"])
        sqrtg = shell_cells.sqrt_gamma(*[vals[n] for n in fields["metric"]])
        calB = sqrtg[:, None] * np.stack([vals[n] for n in fields["B"]], axis=1)
        (self.f_phi, self.f_r, self.f_z, self.zeta,
         self.n_axis) = shell_cells.integrands(
            self.points, calB, centre, pole, self.spacing, shell["r_in"],
            shell["r_out"], shell["axis_cells"], int(shell["subsamples"]))

    def __len__(self):
        return len(self.points)


def tracing_field(cfg, snap, centre, half=None):
    """B^i in the tracing box about ``centre`` (half-width from the config)."""
    tr = cfg["tracer"]
    half = float(tr["box_half_width"] if half is None else half)
    return AMRVectorField.from_snapshot(
        snap, cfg["fields"]["B"], centre - half, centre + half,
        margin=int(tr["margin"]), lookup_cell=float(tr["lookup_cell"]))


def settings_key(cfg):
    """Everything the per-point ends depend on, as a canonical string."""
    tr, shell = cfg["tracer"], cfg["shell"]
    return json.dumps({
        "shell": {k: shell[k] for k in ("centre", "pole", "r_in", "r_out",
                                         "subsamples", "file", "time_offset")},
        "tracer": {k: tr[k] for k in ("box_half_width", "step", "max_length",
                                       "lookup_cell", "horizon_scale",
                                       "extra_horizon_scales", "margin")},
        "B": cfg["fields"]["B"], "simdir": cfg["simdir"],
    }, sort_keys=True, default=str)


def points_path(cfg, it):
    return os.path.join(cfg["output_dir"],
                        "points__%s__it%d.h5" % (cfg["label"], it))


def load_ends(cfg, it, points):
    """The ends stored for this iteration, if they were traced with these
    settings from exactly these points; else None (and why, for the log)."""
    path = points_path(cfg, it)
    if not os.path.isfile(path):
        return None, None
    try:
        with h5py.File(path, "r") as h5:
            if h5.attrs.get("settings") != settings_key(cfg):
                return None, "other settings"
            stored = h5["points"]
            if stored.shape != points.shape or not np.array_equal(stored[()],
                                                                   points):
                return None, "other points"
            return (h5["ends"][()], h5["xends"][()]), None
    except (OSError, KeyError) as exc:
        return None, "unreadable (%s)" % exc


def write_points(cfg, it, t, sp, res, heavy):
    """The per-point classification of one iteration (``points__...h5``).

    Always: positions, volumes, levels and the end codes (also for the enlarged
    horizons) - a few MB, enough to recompute any integral over the shell
    without tracing again, and what a rerun reads instead of tracing. With
    ``heavy`` (``save_points``) also line lengths, end points and integrands.
    Written under a temporary name and renamed, so a file that exists is
    complete.
    """
    path = points_path(cfg, it)
    tmp = "%s.tmp.%d" % (path, os.getpid())
    kw = {"compression": "gzip", "compression_opts": 4}
    with h5py.File(tmp, "w") as h5:
        h5.create_dataset("points", data=sp.points, **kw)
        h5.create_dataset("weight", data=sp.weight, **kw)
        h5.create_dataset("level", data=sp.level, **kw)
        h5.create_dataset("ends", data=res.ends, **kw)
        h5.create_dataset("xends", data=res.xends, **kw)
        if heavy:
            h5.create_dataset("lengths", data=res.lengths.astype(np.float32),
                              **kw)
            h5.create_dataset("endpoints",
                              data=res.endpoints.astype(np.float32), **kw)
            for name in ("f_phi", "f_r", "f_z"):
                h5.create_dataset(name, data=getattr(sp, name), **kw)
        h5.attrs["iteration"] = it
        h5.attrs["time"] = t
        h5.attrs["ends_legend"] = res.legend
        h5.attrs["horizon_scales"] = cfg["horizon_scales"]
        h5.attrs["ends_order"] = ("ends[:, 0] along B, ends[:, 1] against B; "
                                  "xends[:, :, k] the same with the horizons "
                                  "at horizon_scales[k + 1]")
        h5.attrs["settings"] = settings_key(cfg)
    os.replace(tmp, path)
    print("[rank %d] wrote %s" % (rank, path), flush=True)


def process(cfg, xyz, tocs, tables, row):
    """All integrals of one iteration; returns a dict of small arrays."""
    it = int(tables["iterations"][row])
    shell, tr, fields = cfg["shell"], cfg["tracer"], cfg["fields"]
    centre, pole = tables["centre"][row], tables["pole"][row]
    horizons = horizons_at(tables, row)
    tm = {"field": 0.0, "trace": 0.0}
    t0 = time.perf_counter()
    snap = Snapshot(xyz, it, fields["B"][0], tocs=tocs)
    tm["hierarchy"] = time.perf_counter() - t0

    # 1. The shell's points, their volumes and the integrands.
    t0 = time.perf_counter()
    sp = ShellPoints(cfg, snap, centre, pole)
    tm["shell"] = time.perf_counter() - t0

    # 2. The line through every point: from a previous run if it traced these
    #    points with these settings, else traced now in the tracing box.
    stored, why = load_ends(cfg, it, sp.points)
    legend = tracer.end_names(horizons)
    extra = [x / float(tr["horizon_scale"]) for x in tr["extra_horizon_scales"]]
    field_gb = 0.0
    if stored is not None:
        snap.close()
        res = tracer.TraceResult(stored[0], None, None, legend, 0.0, stored[1],
                                 extra)
        print("[rank %d] it=%d: ends read from %s, not traced"
              % (rank, it, points_path(cfg, it)), flush=True)
    else:
        if why:
            print("[rank %d] it=%d: tracing again, the stored ends are for %s"
                  % (rank, it, why), flush=True)
        t0 = time.perf_counter()
        field = tracing_field(cfg, snap, centre)
        snap.close()
        tm["field"] = time.perf_counter() - t0
        field_gb = field.nbytes() / 1e9
        res = tracer.trace(field, horizons, sp.points, step=float(tr["step"]),
                           max_length=float(tr["max_length"]),
                           batches=int(tr["batches"]), extra_scales=extra,
                           log=lambda m: print("[rank %d] it=%d %s"
                                               % (rank, it, m), flush=True))
        del field
        tm["trace"] = res.seconds
        write_points(cfg, it, float(tables["times"][row]), sp, res,
                     it in cfg["save_points"])

    # 3. Per-class sums, for each horizon size.
    n_class = len(res.class_names())
    per_scale = []
    for j in range(len(cfg["horizon_scales"])):
        cls = res.classes(None if j == 0 else j - 1)
        sums = shell_cells.class_sums(cls, n_class, sp.weight, sp.f_phi,
                                      sp.f_r, sp.f_z, sp.zeta, shell["r_in"],
                                      shell["r_out"], shell["n_bin"])
        sums["n_points"] = np.bincount(cls, minlength=n_class)
        per_scale.append(sums)
    out = {key: np.array([p[key] for p in per_scale])
           for key in QUANTITIES + ("profile", "n_points")}
    out["z_edges"] = per_scale[0]["z_edges"]
    cls = res.classes()
    if res.lengths is not None:
        out["mean_length"] = (np.bincount(cls, res.lengths.sum(axis=1), n_class)
                              / np.maximum(out["n_points"][0], 1))
    else:
        out["mean_length"] = np.full(n_class, np.nan)
    out["traced"] = int(stored is None)
    out["volume_exact"] = 4.0 * np.pi / 3.0 * (shell["r_out"] ** 3
                                               - shell["r_in"] ** 3)
    out["levels"] = np.bincount(sp.level.astype(np.int64), minlength=16)[:16]
    out["n_axis"] = sp.n_axis
    out["field_gb"] = field_gb
    out["timings"] = np.array([tm[k] for k in ("hierarchy", "shell", "field",
                                               "trace")])
    out["legend"] = res.legend
    out["classes"] = res.class_names()

    vol = sp.weight.sum()
    print("[rank %d] it=%d t=%g: %d points (levels %s), volume %.6g vs %.6g "
          "(%+.1e), %d near-axis; field %.2f GB; times hierarchy %.0f s, "
          "shell %.0f s, field %.0f s, trace %.0f s"
          % (rank, it, tables["times"][row], len(sp),
             ",".join(str(lv) for lv in np.flatnonzero(out["levels"])), vol,
             out["volume_exact"], vol / out["volume_exact"] - 1.0, sp.n_axis,
             field_gb, tm["hierarchy"], tm["shell"], tm["field"], tm["trace"]),
          flush=True)
    return out


# ---------------------------------------------------------------------------
# Output (rank 0)
# ---------------------------------------------------------------------------

def common_attrs(cfg, first):
    shell, tr = cfg["shell"], cfg["tracer"]
    return {
        "label": cfg["label"], "simdir": cfg["simdir"],
        "source": "raw AMR grid points of the 3D output, finest level owning "
                  "each region, dual-cell volumes; no interpolation",
        "shell_centre_spec": str(shell["centre"]),
        "shell_pole_spec": str(shell["pole"]),
        "frame": "translated to the shell centre, z along the pole; not boosted",
        "r_in": shell["r_in"], "r_out": shell["r_out"], "h": shell["r_out"],
        "axis_cells": float(shell["axis_cells"]),
        "subsamples": int(shell["subsamples"]),
        "B_variables": cfg["fields"]["B"],
        "metric_variables": cfg["fields"]["metric"],
        "calB": "sqrt(det gamma_ij) B^i, projected on flat unit vectors",
        "classes": first["classes"], "ends": first["legend"],
        "classes_meaning": "unordered pair of the ends of the field line "
                           "through each point: a horizon (BH1, BH2), exit "
                           "(left the tracing box), cap (length cap reached), "
                           "null (B = 0)",
        "tracer_box_half_width": float(tr["box_half_width"]),
        "tracer_step": float(tr["step"]),
        "tracer_step_units": "local grid spacing (finest level at the point)",
        "tracer_max_length": float(tr["max_length"]),
        "tracer_interpolation": "trilinear in the finest owning component",
        "tracer_integrator": "RK4, fixed step",
        "horizons": "analytic Kerr-Schild spheroids from the trajectory "
                    "(m, a per hole), semi-axes x horizon_scales[j]; the "
                    "first is traced with, the others recorded from the same "
                    "lines (first entry into the enlarged spheroid)",
        "horizon_scales": np.asarray(cfg["horizon_scales"], dtype=float),
    }


def write_outputs(cfg, tables, results):
    rows = sorted(results)
    first = results[rows[0]]
    iters = tables["iterations"][rows]
    per_iter = {
        "iterations": iters, "times": tables["times"][rows],
        "horizon_scales": np.asarray(cfg["horizon_scales"], dtype=float),
        "shell_centre": tables["centre"][rows],
        "shell_pole": tables["pole"][rows],
        "bh1_position": tables["bh1_position"][rows],
        "bh2_position": tables["bh2_position"][rows],
    }
    attrs = common_attrs(cfg, first)
    os.makedirs(cfg["output_dir"], exist_ok=True)
    for q in QUANTITIES:
        path = os.path.join(cfg["output_dir"], "%s__%s.h5" % (q, cfg["label"]))
        with h5py.File(path, "w") as h5:
            h5.create_dataset("data", data=np.array([results[r][q]
                                                     for r in rows]))
            for name, val in per_iter.items():
                h5.create_dataset(name, data=val)
            if q == "Phi_z":
                h5.create_dataset("profile", data=np.array(
                    [results[r]["profile"] for r in rows]))
                h5.create_dataset("z_edges", data=first["z_edges"])
                h5.attrs["profile_definition"] = (
                    "profile[iteration, horizon_scale, class, bin]: flux "
                    "through the planes zeta = const per bin, sum w "
                    "|calB . pole| / bin width; its mean over the bins is data")
            h5.attrs.update(attrs)
            h5.attrs["quantity"] = q
            h5.attrs["definition"] = DEFINITIONS[q]
            h5.attrs["layout"] = "data[iteration, horizon_scale, class]"
        log("wrote %s" % path)

    path = os.path.join(cfg["output_dir"], "tracing__%s.h5" % cfg["label"])
    with h5py.File(path, "w") as h5:
        for name, val in per_iter.items():
            h5.create_dataset(name, data=val)
        for name in ("n_points", "mean_length", "levels", "timings"):
            h5.create_dataset(name, data=np.array([results[r][name]
                                                   for r in rows]))
        for name in ("volume_exact", "n_axis", "field_gb", "traced"):
            h5.create_dataset(name, data=np.array([results[r][name]
                                                   for r in rows]))
        h5.create_dataset("volume_sum", data=np.array(
            [results[r]["volume"][0].sum() for r in rows]))
        hz = [tables["horizons"][r] for r in rows]
        h5.create_dataset("horizon_centres", data=np.array([h[0] for h in hz]))
        h5.create_dataset("horizon_axes", data=np.array([h[1] for h in hz]))
        h5.create_dataset("horizon_a_eq", data=np.array([h[2] for h in hz]))
        h5.create_dataset("horizon_a_pol", data=np.array([h[3] for h in hz]))
        h5.attrs.update(attrs)
        h5.attrs["timings_definition"] = ("hierarchy, shell points + fields, "
                                          "tracing field, trace [s]")
        h5.attrs["mean_length_definition"] = (
            "mean over the class (first horizon scale) of the summed length of "
            "both half-lines; NaN where the ends were read, not traced")
        h5.attrs["n_points_definition"] = "points per [iteration, horizon_scale, class]"
        h5.attrs["traced_definition"] = "1 traced in this run, 0 ends read from points__ files"
        h5.attrs["levels_definition"] = "points per refinement level"
    log("wrote %s" % path)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def set_threads(cfg):
    """Set the numba thread count; warn if the process has fewer cores.

    OpenMPI binds each rank to cores - an MPI singleton (plain ``python``,
    as mpi4py initialises MPI) to a single one - and numba's threads then
    share those. Give mpirun ``--map-by ...:PE=<threads>`` (or set
    ``OMPI_MCA_hwloc_base_binding_policy=none`` for one rank).
    """
    import numba
    if cfg["tracer"]["threads"]:
        numba.set_num_threads(int(cfg["tracer"]["threads"]))
    threads = numba.get_num_threads()
    cores = len(os.sched_getaffinity(0))
    if cores < threads:
        print("[rank %d] WARNING: %d numba thread(s) but only %d core(s) this "
              "process may run on (MPI binding?); tracing will be slow"
              % (rank, threads, cores), flush=True)
    return threads, cores


def prepare(cfg):
    """Collective: SimDir, iterations and per-iteration tables on every rank.

    Returns ``(backend, xyz reader, tables)``; the iterations are those every
    variable has.
    """
    backend = Postcactus3DBackend()
    sim_cold = _acquire_cold_simdir(backend, cfg)
    sim, _, var_iters = _query_serial(backend, cfg, sim_cold)
    missing = [v for v in cfg["variables"] if v not in var_iters]
    if missing:
        abort("no 3D data for %s" % ", ".join(missing))
    common = None
    for iters, _ in var_iters.values():
        common = set(iters) if common is None else common & set(iters)
    b_iters, b_times = var_iters[cfg["fields"]["B"][0]]
    keep = np.isin(b_iters, sorted(common))
    iterations, times = b_iters[keep], b_times[keep]
    log("%d iteration(s) with every variable, t = %g .. %g"
        % (len(iterations), times[0], times[-1]))
    tables = build_tables(cfg, iterations, times, sim) if rank == 0 else None
    tables = comm.bcast(tables, root=0)
    return backend, backend.plane_index(sim, cfg), tables


def main():
    cfg = None
    if rank == 0:
        parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
        parser.add_argument("config")
        cfg = load_config(parser.parse_args().config)
        os.makedirs(cfg["output_dir"], exist_ok=True)
    cfg = comm.bcast(cfg, root=0)
    threads, cores = set_threads(cfg)
    log("numba: %d thread(s) per rank, %d core(s) on rank 0" % (threads, cores))
    t_start = time.perf_counter()
    log("Shell integrals %r: r = %g .. %g about %s, %d rank(s)"
        % (cfg["label"], cfg["shell"]["r_in"], cfg["shell"]["r_out"],
           cfg["shell"]["centre"], size))

    backend, xyz, tables = prepare(cfg)
    tocs = backend._tocs
    iterations = tables["iterations"]
    mine = list(range(rank, len(iterations), size))
    results = {}
    for row in mine:
        results[row] = process(cfg, xyz, tocs, tables, row)
    gathered = comm.gather(results, root=0)
    backend.finish(cfg)
    if rank == 0:
        merged = {}
        for part in gathered:
            merged.update(part)
        write_outputs(cfg, tables, merged)
    comm.Barrier()
    log("Done: %d iteration(s), total wall %.0f s"
        % (len(iterations), time.perf_counter() - t_start))


if __name__ == "__main__":
    main()
