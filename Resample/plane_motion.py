#!/usr/bin/env python3
"""Time-dependent orientation of the resampling plane.

The plane the 3D backend cuts need not be fixed: it can rotate with time, for
instance following a binary so that both black holes always lie in it. This
module turns the config's ``plane`` block into a :class:`FrameTable` - one
orthonormal frame ``(origin, u, v, n)`` per output iteration - which the
backend then simply looks up.

Three kinds of plane, all producing a frame table:

``static``
    The frame is constant. This is the original behaviour, and is bit-for-bit
    unchanged: the frame comes from :meth:`plane_geom.PlaneSpec.from_config`.

``analytic``
    You supply a small Python module with ``normal(t, it)`` (and optionally
    ``origin(t, it)`` / ``u_axis(t, it)``); see ``plane_module_example.py``.

``trajectory``
    The frame is derived from a black-hole trajectory file. Normally you name a
    second in-plane vector and the plane spans it together with the separation
    vector, so both holes lie in it at every time and it tilts whenever they
    leave it. Prefer the presets defined by the orbit itself rather than by the
    coordinate axes:

    ``orbital``
        the relative velocity. Separation and relative velocity span the
        **instantaneous orbital plane** by definition, so this is exact for an
        inclined, precessing or eccentric orbit - whereas ``equatorial``
        (from the z axis) quietly assumes the orbit lies in z = 0.
    ``orbital_meridional``
        the orbital angular momentum axis. A meridional cut through both holes
        that is vertical with respect to the **orbit** rather than the grid,
        and perpendicular to the orbital motion. Reduces exactly to
        ``meridional`` for an orbit in the z = 0 plane.
    ``spin1`` / ``spin2``
        that hole's spin axis, giving its meridional plane facing the
        companion - natural together with ``origin: bh1`` / ``bh2``.

    A plane that need *not* contain the separation vector is given by ``normal``
    instead; ``normal: spin1`` with ``origin: bh1`` is one hole's **equatorial
    plane**, the one its accretion flow wants to align with.

Why the whole table is precomputed
----------------------------------
Two reasons, both about correctness rather than speed.

*Ranks resample iterations out of order* (round-robin, and split into chunks),
so a frame can never be defined relative to "the previous frame" - by the time
a rank reaches iteration k it has usually not seen k-1. The frame must be a
pure function of the iteration.

But *the naive pure function is discontinuous*: the static convention (drop the
coordinate axis most aligned with the normal) makes u and v swap as the normal
rotates past a boundary, which would flip the movie mid-sequence. Computing all
frames together on one rank solves both: continuity can be imposed and, more
importantly, **verified** before any data is read, since every output time is
already known from the iteration query.

Building the table on rank 0 also means the trajectory file is read once rather
than once per rank, and lets the geometry be written into the output as
per-iteration datasets.
"""

import contextlib
import hashlib
import importlib.util
import io
import os

import numpy as np
import h5py

from postcactus import cactus_parfile

from plane_geom import PlaneSpec

# Named choices for the second in-plane vector of a trajectory-driven plane.
# The plane then holds the separation vector and this one, so both bodies always
# lie in it. The first two are defined by the orbit itself and are what you
# normally want; the last two are their coordinate-frame approximations, correct
# only while the orbit stays in the z = 0 plane.
SECOND_VECTOR_PRESETS = ("orbital", "orbital_meridional", "spin1", "spin2",
                         "equatorial", "meridional")

# Named choices for a normal given directly, when the plane is *not* required to
# contain the separation vector (a single body's equatorial plane, say).
NORMAL_PRESETS = ("orbital", "spin1", "spin2")

# Named choices for the in-plane u axis (which direction is "right" in a frame).
U_AXIS_PRESETS = ("separation", "second")

_TINY = 1e-12

# Below this the separation direction is meaningless (the holes have merged).
_MIN_SEPARATION = 1e-8


# ---------------------------------------------------------------------------
# Small vector helpers (all operate on (n, 3) stacks)
# ---------------------------------------------------------------------------

def _normalise(vecs, what, times=None):
    """Normalise a stack of vectors, reporting where it is degenerate."""
    norms = np.linalg.norm(vecs, axis=1)
    bad = norms <= _TINY
    if bad.any():
        where = ""
        if times is not None:
            where = " (first at t = %g)" % times[bad][0]
        raise ValueError("%s is degenerate at %d of %d time(s)%s"
                         % (what, int(bad.sum()), len(norms), where))
    return vecs / norms[:, None]


def _broadcast(vec, n):
    """Turn a single 3-vector into an ``(n, 3)`` stack."""
    return np.repeat(np.asarray(vec, dtype=float).reshape(1, 3), n, axis=0)


def _orthogonalise(u, n_hat, what, times=None):
    """Project ``u`` into the plane with normal ``n_hat`` and normalise."""
    return _normalise(u - (np.sum(u * n_hat, axis=1))[:, None] * n_hat,
                      what, times)


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------

class FrameTable(object):
    """The plane's frame at every output iteration of one variable.

    ``origin``/``u_hat``/``v_hat``/``n_hat`` are ``(n_iter, 3)``; ``extras``
    holds any further per-iteration arrays to store in the output (black-hole
    positions and such). Picklable, so it survives the broadcast to all ranks.
    """

    def __init__(self, iterations, times, origin, u_hat, v_hat, n_hat,
                 label="static", moving=False, extras=None):
        self.iterations = np.asarray(iterations, dtype=np.int64)
        self.times = np.asarray(times, dtype=np.float64)
        self.origin = np.asarray(origin, dtype=float)
        self.u_hat = np.asarray(u_hat, dtype=float)
        self.v_hat = np.asarray(v_hat, dtype=float)
        self.n_hat = np.asarray(n_hat, dtype=float)
        self.label = label
        self.moving = bool(moving)
        self.extras = dict(extras or {})
        self._row = {int(it): i for i, it in enumerate(self.iterations)}

    def spec_at(self, iteration):
        """The :class:`~plane_geom.PlaneSpec` for one iteration."""
        try:
            i = self._row[int(iteration)]
        except KeyError:
            raise KeyError(
                "no plane frame was precomputed for iteration %d; the frame "
                "table was built from a different iteration list"
                % int(iteration))
        return PlaneSpec.from_frame(self.origin[i], self.u_hat[i],
                                    self.v_hat[i], self.n_hat[i],
                                    name=self.label)

    def per_iteration_data(self, iterations):
        """The per-iteration datasets for the given iterations, in that order.

        Empty for a static plane: its geometry is already in the file's
        attributes, and constant columns would only bloat every output.
        """
        if not self.moving:
            return {}
        rows = [self._row[int(it)] for it in iterations]
        out = {
            "plane_origin": self.origin[rows],
            "plane_u_axis": self.u_hat[rows],
            "plane_v_axis": self.v_hat[rows],
            "plane_normal": self.n_hat[rows],
        }
        for name, arr in self.extras.items():
            out[name] = np.asarray(arr)[rows]
        return out

    # -- diagnostics -------------------------------------------------------

    def continuity_report(self):
        """Per-frame rotation of the plane, for the startup log.

        Returns ``(total_degrees, max_step_degrees, flips)``: how far the frame
        turns in total, the largest turn between consecutive output frames (a
        large value means the rotation is undersampled - the movie will jump),
        and the number of places the frame reverses, which should be zero.
        """
        if len(self.iterations) < 2:
            return 0.0, 0.0, 0
        dots = np.clip(np.sum(self.u_hat[1:] * self.u_hat[:-1], axis=1)
                       + np.sum(self.n_hat[1:] * self.n_hat[:-1], axis=1), -2.0, 2.0)
        steps = np.degrees(np.arccos(np.clip(
            np.sum(self.u_hat[1:] * self.u_hat[:-1], axis=1), -1.0, 1.0)))
        return float(steps.sum()), float(steps.max()), int((dots < 0).sum())


# ---------------------------------------------------------------------------
# Continuity
# ---------------------------------------------------------------------------

def _make_normal_continuous(n_hat):
    """Flip normals so consecutive frames agree on which side is "up".

    A plane does not care about the sign of its normal, but the frame does:
    ``v = n x u`` would flip with it, mirroring the image. Only the *relative*
    signs matter, so this walks the sequence once. Returns the fixed normals
    and how many flips were applied.
    """
    n_hat = np.array(n_hat, dtype=float, copy=True)
    flips = 0
    for i in range(1, len(n_hat)):
        if float(np.dot(n_hat[i], n_hat[i - 1])) < 0.0:
            n_hat[i] = -n_hat[i]
            flips += 1
    return n_hat, flips


def _stable_reference_axis(n_hat):
    """Pick the coordinate axis least aligned with *any* of the normals.

    Used as the u-axis reference when none was given: projecting one fixed
    direction into every plane is continuous, whereas choosing per frame (the
    static convention) is not. Having the whole table in hand is what makes
    the globally best choice possible.
    """
    worst = [float(np.abs(n_hat[:, d]).max()) for d in range(3)]
    return int(np.argmin(worst)), worst


# ---------------------------------------------------------------------------
# Trajectory files
# ---------------------------------------------------------------------------

class TrajectoryState(object):
    """The binary's state interpolated to a set of times (each ``(n, 3)``).

    ``vel1``/``vel2`` and ``spin1``/``spin2`` are None when the trajectory file
    carries no velocities or spins; the presets that need them say so.
    """

    def __init__(self, pos1, pos2, vel1, vel2, spin1, spin2, mass1, mass2):
        self.pos1, self.pos2 = pos1, pos2
        self.vel1, self.vel2 = vel1, vel2
        self.spin1, self.spin2 = spin1, spin2
        self.mass1, self.mass2 = mass1, mass2

    def require(self, attr, what, preset):
        value = getattr(self, attr)
        if value is None:
            raise ValueError("%r needs %s in the trajectory file, which does "
                             "not have it" % (preset, what))
        return value

    def relative_velocity(self, preset):
        v1 = self.require("vel1", "velocities (vx1/vy1/vz1)", preset)
        v2 = self.require("vel2", "velocities (vx2/vy2/vz2)", preset)
        return v2 - v1

    def spin(self, body, preset):
        return self.require("spin%d" % body,
                            "spins (a%dx/a%dy/a%dz)" % (body, body, body),
                            preset)


class Trajectory(object):
    """Body positions, velocities, spins and masses from a trajectory HDF5 file.

    Expects flat datasets ``t``, ``x1``/``y1``/``z1``, ``x2``/``y2``/``z2``, and
    optionally velocities ``vx1``.., spins ``a1x``.. and masses ``m1``/``m2``
    (scalars) or ``m1_full``/``m2_full`` (time series) - the layout written by
    AnalyticalSpacetime's trajectory tables.
    """

    def __init__(self, path):
        self.path = path
        with h5py.File(path, "r") as h5:
            missing = [k for k in ("t", "x1", "y1", "z1", "x2", "y2", "z2")
                       if k not in h5]
            if missing:
                raise ValueError("%s is missing dataset(s): %s"
                                 % (path, ", ".join(missing)))
            self.t = np.asarray(h5["t"][:], dtype=float)
            self.pos = [self._stack(h5, ("x1", "y1", "z1")),
                        self._stack(h5, ("x2", "y2", "z2"))]
            self.vel = [self._stack(h5, ("vx1", "vy1", "vz1")),
                        self._stack(h5, ("vx2", "vy2", "vz2"))]
            self.spin = [self._stack(h5, ("a1x", "a1y", "a1z")),
                         self._stack(h5, ("a2x", "a2y", "a2z"))]
            self.mass = [self._mass(h5, "m1"), self._mass(h5, "m2")]

        if self.t.ndim != 1 or self.t.size < 2:
            raise ValueError("%s: `t` must be a 1D series of at least 2 samples"
                             % path)
        if np.any(np.diff(self.t) <= 0.0):
            raise ValueError("%s: `t` is not strictly increasing, so it cannot "
                             "be interpolated" % path)

    @staticmethod
    def _stack(h5, keys):
        """An ``(n, 3)`` stack of three datasets, or None if any is absent."""
        if not all(k in h5 for k in keys):
            return None
        return np.column_stack([np.asarray(h5[k][:], dtype=float) for k in keys])

    @staticmethod
    def _mass(h5, key):
        """Mass as a time series when available, else the constant."""
        if key + "_full" in h5:
            return np.asarray(h5[key + "_full"][:], dtype=float)
        if key in h5:
            return float(np.asarray(h5[key]).ravel()[0])
        return 1.0

    def span(self):
        return float(self.t[0]), float(self.t[-1])

    def available(self):
        """Which optional quantities this file carries, for the log."""
        have = ["positions"]
        if all(v is not None for v in self.vel):
            have.append("velocities")
        if all(s is not None for s in self.spin):
            have.append("spins")
        return have

    def at(self, times):
        """Interpolate everything to ``times``; returns a :class:`TrajectoryState`."""
        times = np.asarray(times, dtype=float)

        def resample(stack):
            if stack is None:
                return None
            return np.column_stack([np.interp(times, self.t, stack[:, d])
                                    for d in range(3)])

        def mass(m):
            return (np.interp(times, self.t, m) if np.ndim(m)
                    else np.full(times.shape, float(m)))

        return TrajectoryState(
            resample(self.pos[0]), resample(self.pos[1]),
            resample(self.vel[0]), resample(self.vel[1]),
            resample(self.spin[0]), resample(self.spin[1]),
            mass(self.mass[0]), mass(self.mass[1]))


def resolve_trajectory_path(spec, sim, log):
    """Locate the trajectory file, resolving ``auto`` from the parfile.

    ``auto`` reads ``AnalyticalSpacetime::traj_table_name``, whose value is
    relative to the parfile's own directory, and is the safest option: it
    cannot disagree with what the simulation actually used.
    """
    if spec != "auto":
        path = os.path.abspath(os.path.expanduser(str(spec)))
        if not os.path.isfile(path):
            raise ValueError("trajectory file not found: %s" % path)
        return path

    name = _parfile_param(sim, "traj_table_name", log)
    if name is None:
        raise ValueError(
            "plane.file: auto needs AnalyticalSpacetime::traj_table_name in "
            "the simulation's parfile, which was not found - give the path "
            "explicitly instead")
    name = str(name).strip().strip('"').strip("'")

    candidates = [os.path.join(os.path.dirname(p), name)
                  for p in (getattr(sim, "parfiles", None) or [])]
    candidates.append(os.path.join(sim.path, os.path.basename(name)))
    for path in candidates:
        path = os.path.abspath(path)
        if os.path.isfile(path):
            log("  trajectory file (from parfile): %s" % path)
            return path
    raise ValueError(
        "AnalyticalSpacetime::traj_table_name is %r but no such file was found "
        "near the parfile(s); give plane.file explicitly" % name)


def resolve_time_offset(spec, sim, log):
    """The offset from simulation time to trajectory time.

    ``auto`` reads ``AnalyticalSpacetime::AST_t0``, the trajectory time the run
    started from. Getting this wrong rotates the plane by a constant angle
    without any other symptom, so deriving it from the parfile rather than
    copying it by hand is worth the special case.
    """
    if spec != "auto":
        return float(spec)
    value = _parfile_param(sim, "ast_t0", log)
    if value is None:
        raise ValueError(
            "plane.time_offset: auto needs AnalyticalSpacetime::AST_t0 in the "
            "simulation's parfile, which was not found - give the offset "
            "explicitly (0 if simulation and trajectory times agree)")
    log("  time offset (from parfile AST_t0): %.12g" % float(value))
    return float(value)


def _parfile_param(sim, key, log=None):
    """One AnalyticalSpacetime parameter, searched across the SimDir's parfiles.

    ``SimDir.initial_params`` only parses ``parfiles[0]``, which for a
    multi-segment run (or one whose tree holds output from several machines) is
    often not the parfile that configured the evolution. So all of them are
    searched, and a disagreement is reported rather than silently resolved -
    for a value like ``AST_t0`` picking the wrong one rotates the plane.
    """
    values, sources = [], []
    for parfile in _candidate_parfiles(sim):
        value = _thorn_param(parfile, key)
        if value is None:
            continue
        if not any(_same_value(value, v) for v in values):
            values.append(value)
            sources.append(parfile)
    if not values:
        return None
    if len(values) > 1 and log is not None:
        log("  WARNING: the parfiles disagree on AnalyticalSpacetime::%s "
            "(%s); using %r from %s. Set the value explicitly in the config if "
            "that is not the right one."
            % (key, ", ".join(repr(v) for v in values), values[0],
               os.path.basename(str(sources[0]))))
    return values[0]


def _candidate_parfiles(sim):
    """The already-parsed parfile first, then each distinct parfile on disk.

    Restarts copy the parfile verbatim, so identical ones are collapsed by
    content: it keeps the disagreement check meaningful and avoids parsing the
    same text dozens of times. postcactus reports unparsed parfile fragments on
    stdout, which would bury the caller's own output, so that is swallowed here.
    """
    parsed = [getattr(sim, "initial_params", None)]
    seen = set()
    for path in (getattr(sim, "parfiles", None) or []):
        try:
            with open(path, "rb") as f:
                digest = hashlib.sha1(f.read()).hexdigest()
        except OSError:
            continue
        if digest in seen:
            continue
        seen.add(digest)
        try:
            with contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()):
                parsed.append(cactus_parfile.load_parfile(path))
        except Exception:      # noqa: BLE001 - an unparseable parfile is skipped
            continue
    return [p for p in parsed if p is not None]


def _thorn_param(parfile, key):
    try:
        return getattr(parfile.analyticalspacetime, key)
    except Exception:          # noqa: BLE001 - no thorn, no key
        return None


def _same_value(a, b):
    try:
        return bool(np.isclose(float(a), float(b), rtol=1e-12, atol=0.0))
    except (TypeError, ValueError):
        return str(a) == str(b)


# ---------------------------------------------------------------------------
# The user-supplied module (analytic planes)
# ---------------------------------------------------------------------------

def load_plane_module(path):
    """Import a plane module by file path (no package/sys.path involvement).

    Only rank 0 ever loads it, since that is where the table is built.
    """
    path = os.path.abspath(os.path.expanduser(str(path)))
    if not os.path.isfile(path):
        raise ValueError("plane.module not found: %s" % path)
    spec = importlib.util.spec_from_file_location("_etk_plane_module", path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:   # noqa: BLE001 - report the user's error clearly
        raise ValueError("plane.module %s failed to import: %s: %s"
                         % (path, type(exc).__name__, exc))
    return module


def _evaluate(module, name, times, iterations, what):
    """Call ``module.name(t, it)`` for every frame; returns an ``(n, 3)`` stack.

    Vectorised first (one call with the whole time array, if the function
    happens to be numpy-friendly), else per frame.
    """
    func = getattr(module, name, None)
    if func is None:
        return None
    try:
        result = np.asarray(func(times, iterations), dtype=float)
        if result.shape == (len(times), 3):
            return result
        if result.shape == (3, len(times)):
            return result.T
        if result.shape == (3,) and len(times) == 1:
            return result.reshape(1, 3)
    except Exception:          # noqa: BLE001 - fall back to per-frame calls
        pass

    rows = []
    for t, it in zip(times, iterations):
        try:
            value = np.asarray(func(float(t), int(it)), dtype=float).ravel()
        except Exception as exc:   # noqa: BLE001 - name the offending frame
            raise ValueError("%s: %s(t=%g, it=%d) raised %s: %s"
                             % (what, name, t, it, type(exc).__name__, exc))
        if value.size != 3:
            raise ValueError("%s: %s(t=%g, it=%d) returned %d value(s), need 3"
                             % (what, name, t, it, value.size))
        rows.append(value)
    return np.array(rows, dtype=float)


# ---------------------------------------------------------------------------
# Building the table
# ---------------------------------------------------------------------------

def plane_kind(plane):
    """The ``kind`` of a config ``plane`` value; ``static`` unless it says so."""
    if isinstance(plane, dict):
        return str(plane.get("kind") or "static").strip().lower()
    return "static"


def build_frame_table(plane, iterations, times, sim=None, log=print,
                      trajectory=None):
    """Build the :class:`FrameTable` for one variable's output iterations.

    ``sim`` is the SimDir (only needed to resolve ``auto`` values from the
    parfile) and ``trajectory`` an already-loaded :class:`Trajectory` to reuse
    across variables. Raises ``ValueError`` on any bad specification; the
    caller turns that into an abort.
    """
    iterations = np.asarray(iterations, dtype=np.int64)
    times = np.asarray(times, dtype=np.float64)
    kind = plane_kind(plane)

    if kind == "static":
        return _static_table(plane, iterations, times)
    if kind == "analytic":
        return _analytic_table(plane, iterations, times, log)
    if kind == "trajectory":
        return _trajectory_table(plane, iterations, times, sim, log, trajectory)
    raise ValueError("unknown plane kind %r (use static, analytic or trajectory)"
                     % kind)


def _static_table(plane, iterations, times):
    """A constant frame, from the original static convention."""
    if isinstance(plane, dict):
        plane = {k: v for k, v in plane.items() if k != "kind"}
    spec = PlaneSpec.from_config(plane)
    n = len(iterations)
    return FrameTable(iterations, times,
                      _broadcast(spec.origin, n), _broadcast(spec.u_hat, n),
                      _broadcast(spec.v_hat, n), _broadcast(spec.n_hat, n),
                      label=spec.label, moving=False)


def _finish(iterations, times, origin, n_raw, u_raw, label, log, extras=None):
    """Turn raw normals / u directions into a continuous orthonormal table."""
    n_hat = _normalise(n_raw, "plane normal", times)
    n_hat, flips = _make_normal_continuous(n_hat)
    if flips:
        log("  fixed %d normal-sign flip(s) for a continuous frame" % flips)

    if u_raw is None:
        axis, alignment = _stable_reference_axis(n_hat)
        log("  in-plane u axis: projection of the %s axis (chosen as the least "
            "aligned with the plane normal, |n.e| <= %.3f)"
            % ("xyz"[axis], alignment[axis]))
        if alignment[axis] > 0.99:
            log("  WARNING: every coordinate axis comes within %.2f degrees of "
                "the normal somewhere; set plane.u_axis explicitly or the "
                "in-plane orientation will swing"
                % np.degrees(np.arccos(min(1.0, alignment[axis]))))
        u_raw = _broadcast(np.eye(3)[axis], len(iterations))

    u_hat = _orthogonalise(u_raw, n_hat, "in-plane u axis", times)
    v_hat = np.cross(n_hat, u_hat)

    table = FrameTable(iterations, times, origin, u_hat, v_hat, n_hat,
                       label=label, moving=True, extras=extras)

    total, largest, remaining = table.continuity_report()
    log("  frame turns %.1f degrees in total over %d output frame(s); "
        "largest step %.2f degrees" % (total, len(iterations), largest))
    if largest > 30.0:
        log("  WARNING: the plane turns more than 30 degrees between "
            "consecutive frames - the rotation is undersampled by the output "
            "cadence and the movie will jump")
    if remaining:
        log("  WARNING: the frame still reverses at %d place(s); the plane "
            "specification is not continuous in time" % remaining)
    return table


def _analytic_table(plane, iterations, times, log):
    """Frames from a user-supplied Python module."""
    if plane.get("module") is None:
        raise ValueError("an analytic plane needs `module`: the path to a "
                         "Python file defining normal(t, it)")
    module = load_plane_module(plane["module"])
    log("  analytic plane from %s" % os.path.abspath(str(plane["module"])))

    unknown = sorted(set(plane) - {"kind", "module", "origin", "u_axis"})
    if unknown:
        raise ValueError("unknown plane key(s) for kind analytic: %s"
                         % ", ".join(unknown))

    n_raw = _evaluate(module, "normal", times, iterations, "analytic plane")
    if n_raw is None:
        raise ValueError("plane.module %s defines no normal(t, it)"
                         % plane["module"])

    origin = _evaluate(module, "origin", times, iterations, "analytic plane")
    if origin is None:
        origin = _broadcast(_as_vector_or_zero(plane.get("origin")),
                            len(iterations))

    u_raw = _evaluate(module, "u_axis", times, iterations, "analytic plane")
    if u_raw is None and plane.get("u_axis") is not None:
        u_raw = _broadcast(_as_vector(plane["u_axis"]), len(iterations))

    return _finish(iterations, times, origin, n_raw, u_raw, "analytic", log)


def _trajectory_table(plane, iterations, times, sim, log, trajectory):
    """Frames tied to the binary: the separation vector lies in the plane.

    Either ``second_vector`` (the plane spans the separation and that vector, so
    both bodies are always in it) or ``normal`` (the orientation given directly,
    for a plane not required to contain the separation) - not both.
    """
    unknown = sorted(set(plane) - {"kind", "file", "time_offset", "origin",
                                   "second_vector", "normal", "u_axis",
                                   "module"})
    if unknown:
        raise ValueError("unknown plane key(s) for kind trajectory: %s"
                         % ", ".join(unknown))
    if plane.get("second_vector") is not None and plane.get("normal") is not None:
        raise ValueError(
            "give either `second_vector` (the plane contains the separation "
            "vector and that one) or `normal` (the orientation directly), not "
            "both")

    if trajectory is None:
        trajectory = Trajectory(
            resolve_trajectory_path(plane.get("file", "auto"), sim, log))
        log("  trajectory carries: %s" % ", ".join(trajectory.available()))
    offset = resolve_time_offset(plane.get("time_offset", "auto"), sim, log)

    t_traj = times + offset
    lo, hi = trajectory.span()
    slack = 1e-9 * max(1.0, abs(hi))
    outside = (t_traj < lo - slack) | (t_traj > hi + slack)
    if outside.any():
        raise ValueError(
            "the output spans simulation time %g..%g, i.e. trajectory time "
            "%g..%g with offset %g, but %s only covers %g..%g (%d of %d frames "
            "fall outside). Check plane.time_offset - for an "
            "AnalyticalSpacetime run it is AST_t0."
            % (times[0], times[-1], t_traj[0], t_traj[-1], offset,
               os.path.basename(trajectory.path), lo, hi,
               int(outside.sum()), len(t_traj)))

    state = trajectory.at(t_traj)
    p1, p2 = state.pos1, state.pos2
    sep_vec = p2 - p1
    separation = np.linalg.norm(sep_vec, axis=1)
    if (separation < _MIN_SEPARATION).any():
        raise ValueError(
            "the two bodies coincide at %d of %d output time(s) (first at "
            "simulation time %g): the separation direction is undefined there, "
            "so restrict the run with iteration_max"
            % (int((separation < _MIN_SEPARATION).sum()), len(separation),
               times[separation < _MIN_SEPARATION][0]))

    d_hat = sep_vec / separation[:, None]
    module = (load_plane_module(plane["module"])
              if plane.get("module") is not None else None)

    origin = _trajectory_origin(plane.get("origin", "com"), state,
                                times, iterations, module)

    second = None
    if plane.get("normal") is not None:
        n_raw = _trajectory_normal(plane["normal"], state, d_hat,
                                   times, iterations, module, log)
    else:
        second = _second_vector(plane.get("second_vector", "orbital"),
                                state, d_hat, times, iterations, module, log)
        second = _normalise(second, "second in-plane vector", times)
        n_raw = np.cross(d_hat, second)
        _check_span(n_raw, d_hat, second, times, log)

    u_raw = _trajectory_u_axis(plane.get("u_axis", "separation"),
                               d_hat, second, state, times, iterations, module)

    log("  separation %.4f .. %.4f over the output; masses %.4f / %.4f"
        % (separation.min(), separation.max(), state.mass1[0], state.mass2[0]))

    table = _finish(iterations, times, origin, n_raw, u_raw,
                    "trajectory", log)

    # In-plane coordinates of the two bodies. Both sit at v = 0 whenever the
    # plane was built to contain the separation and the origin lies on the line
    # joining them - the off-plane residual below is the self-check for that.
    table.extras.update(_body_positions(table, p1, p2, separation, log))
    return table


# How nearly parallel the separation and the second in-plane vector may get
# before the plane they span becomes ill-conditioned. |sin(angle)| thresholds.
_SPAN_ERROR = 1e-3     # 0.06 degrees - the frame would be round-off
_SPAN_WARN = 5e-2      # 2.9 degrees - usable but the frame starts to swing


def _check_span(n_raw, d_hat, second, times, log):
    """Refuse a second in-plane vector that runs (nearly) along the separation.

    Both are unit vectors, so ``|d x w| = |sin(angle)|`` between them; when that
    approaches zero the two no longer span a plane and the normal is noise. This
    catches the case a single-frame check would miss: a *constant* vector that
    the separation only sweeps past part way through the orbit.
    """
    sine = np.linalg.norm(n_raw, axis=1)
    worst = int(np.argmin(sine))
    angle = np.degrees(np.arcsin(np.clip(sine[worst], 0.0, 1.0)))
    if sine[worst] < _SPAN_ERROR:
        raise ValueError(
            "the second in-plane vector comes within %.4g degrees of the "
            "separation vector at simulation time %g, so the two do not span a "
            "plane there. A constant vector cannot work for a binary that turns "
            "through it - use one of the orbit-defined presets (orbital, "
            "orbital_meridional, spin1/spin2), which stay well conditioned."
            % (angle, times[worst]))
    if sine[worst] < _SPAN_WARN:
        log("  WARNING: the second in-plane vector comes within %.2f degrees of "
            "the separation vector (at t = %g); the frame will swing there"
            % (angle, times[worst]))
    else:
        log("  separation and second in-plane vector stay at least %.2f degrees "
            "apart" % angle)


def _trajectory_normal(spec, state, d_hat, times, iterations, module, log):
    """The plane normal given directly, rather than via a second in-plane vector.

    This is how you ask for a plane that need *not* contain the separation
    vector - most usefully one body's equatorial plane, ``normal: spin1``.
    """
    if isinstance(spec, str):
        key = spec.strip().lower()
        if key == "orbital":
            # The orbital angular momentum axis: same plane as
            # second_vector: orbital, stated the other way round.
            log("  normal: orbital angular momentum axis, d x v_rel")
            return np.cross(d_hat, state.relative_velocity(spec))
        if key in ("spin1", "spin2"):
            body = int(key[-1])
            log("  normal: spin axis of body %d (that body's equatorial plane)"
                % body)
            return _spin_direction(state, body, spec, times)
        if key == "module":
            value = _evaluate(module, "normal", times, iterations,
                              "trajectory plane")
            if value is None:
                raise ValueError("normal: module needs the plane module to "
                                 "define normal(t, it)")
            log("  normal: from the plane module")
            return value
        if key in ("x", "y", "z"):
            log("  normal: %s axis (a fixed plane)" % key)
            return _broadcast(_as_vector(key), len(times))
        raise ValueError("normal %r is not one of %s, an axis name, `module`, "
                         "or three numbers"
                         % (spec, ", ".join(NORMAL_PRESETS)))
    vec = _as_vector(spec)
    log("  normal: constant %s" % np.round(vec, 6).tolist())
    return _broadcast(vec, len(times))


def _spin_direction(state, body, preset, times):
    """Unit spin axis of one body, rejecting a non-spinning one."""
    spin = state.spin(body, preset)
    magnitude = np.linalg.norm(spin, axis=1)
    if (magnitude <= 1e-8).any():
        i = int(np.argmin(magnitude))
        raise ValueError(
            "body %d has zero spin at simulation time %g (|a| = %.3e), so it "
            "has no spin axis to orient the plane by"
            % (body, times[i], magnitude[i]))
    return spin / magnitude[:, None]


def _body_positions(table, p1, p2, separation, log):
    """Per-iteration body positions, in world and in-plane coordinates."""
    extras = {"bh1_position": p1, "bh2_position": p2,
              "separation": separation}
    for name, pos in (("bh1_uv", p1), ("bh2_uv", p2)):
        rel = pos - table.origin
        extras[name] = np.column_stack([
            np.sum(rel * table.u_hat, axis=1),
            np.sum(rel * table.v_hat, axis=1)])
        off = np.abs(np.sum(rel * table.n_hat, axis=1)).max()
        log("  %s: |u| up to %.4f, |v| up to %.3e, off-plane residual %.3e"
            % (name, np.abs(extras[name][:, 0]).max(),
               np.abs(extras[name][:, 1]).max(), off))
    return extras


def _second_vector(spec, state, d_hat, times, iterations, module, log):
    """The second in-plane vector, so the plane holds it and the separation.

    The two orbit-defined presets are the ones to prefer. ``orbital`` spans the
    plane with the separation and the *relative velocity*, which is the
    instantaneous orbital plane by definition - exact for an inclined,
    precessing or eccentric orbit, where the coordinate-frame version
    (``equatorial``, from the z axis) silently assumes the orbit lies in
    z = 0. ``orbital_meridional`` is the same idea for the perpendicular cut:
    vertical with respect to the *orbit* rather than to the grid.
    """
    n = len(times)
    if isinstance(spec, str):
        key = spec.strip().lower()

        if key == "orbital":
            # Separation + relative velocity span the instantaneous orbital
            # plane; the normal is then the orbital angular momentum axis.
            log("  second in-plane vector: relative velocity v2 - v1 "
                "(instantaneous orbital plane)")
            return state.relative_velocity(spec)

        if key == "orbital_meridional":
            # The orbital angular momentum axis, so the plane holds the
            # separation and the orbit's own "vertical": a meridional cut
            # through both bodies, perpendicular to the orbital motion.
            log("  second in-plane vector: orbital angular momentum axis "
                "d x v_rel (meridional cut through the orbit)")
            return _normalise(np.cross(d_hat, state.relative_velocity(spec)),
                              "orbital angular momentum axis", times)

        if key in ("spin1", "spin2"):
            # The plane holds the separation and that body's spin axis: its
            # meridional plane, oriented towards the companion.
            body = int(key[-1])
            log("  second in-plane vector: spin axis of body %d (meridional "
                "plane of that body, facing the companion)" % body)
            return _spin_direction(state, body, spec, times)

        if key == "meridional":
            # The z axis: vertical with respect to the grid.
            log("  second in-plane vector: z axis (meridional slice in the "
                "coordinate frame)")
            return _broadcast((0.0, 0.0, 1.0), n)

        if key == "equatorial":
            # In the xy plane and perpendicular to the separation, so the plane
            # is the z = 0 plane, tilting only when the bodies leave it.
            log("  second in-plane vector: z x separation (equatorial slice in "
                "the coordinate frame)")
            return np.cross(_broadcast((0.0, 0.0, 1.0), n), d_hat)

        if key == "module":
            value = _evaluate(module, "second_vector", times, iterations,
                              "trajectory plane")
            if value is None:
                raise ValueError("second_vector: module needs the plane module "
                                 "to define second_vector(t, it)")
            log("  second in-plane vector: from the plane module")
            return value

        raise ValueError("second_vector %r is not one of %s, `module`, or three "
                         "numbers" % (spec, ", ".join(SECOND_VECTOR_PRESETS)))

    vec = _as_vector(spec)
    log("  second in-plane vector: constant %s" % np.round(vec, 6).tolist())
    return _broadcast(vec, n)


def _trajectory_u_axis(spec, d_hat, second, state, times, iterations, module):
    """Which in-plane direction is +u (the output's `x` axis)."""
    if isinstance(spec, str):
        key = spec.strip().lower()
        if key == "separation":
            # Co-rotating: the bodies stay put at u = +-(their distance to the
            # origin), v = 0, and the gas moves around them.
            return d_hat
        if key == "second":
            if second is None:
                raise ValueError("u_axis: second needs `second_vector` to be "
                                 "the way the plane was specified")
            return second
        if key in ("spin1", "spin2"):
            return _spin_direction(state, int(key[-1]), spec, times)
        if key == "module":
            value = _evaluate(module, "u_axis", times, iterations,
                              "trajectory plane")
            if value is None:
                raise ValueError("u_axis: module needs the plane module to "
                                 "define u_axis(t, it)")
            return value
        # An axis name ('x'/'y'/'z') gives an inertial view: the frame stays as
        # close to that fixed direction as the plane allows, so the binary
        # visibly orbits inside it.
        return _broadcast(_as_vector(spec), len(times))
    return _broadcast(_as_vector(spec), len(times))


def _trajectory_origin(spec, state, times, iterations, module):
    """Where the in-plane coordinates are centred."""
    if isinstance(spec, str):
        key = spec.strip().lower()
        if key == "com":
            m1, m2 = state.mass1, state.mass2
            total = m1 + m2
            return ((m1[:, None] * state.pos1 + m2[:, None] * state.pos2)
                    / total[:, None])
        if key == "midpoint":
            return 0.5 * (state.pos1 + state.pos2)
        if key == "bh1":
            return state.pos1
        if key == "bh2":
            return state.pos2
        if key == "module":
            value = _evaluate(module, "origin", times, iterations,
                              "trajectory plane")
            if value is None:
                raise ValueError("origin: module needs the plane module to "
                                 "define origin(t, it)")
            return value
        raise ValueError("origin %r is not one of com, midpoint, bh1, bh2, "
                         "`module`, or three numbers" % spec)
    return _broadcast(_as_vector(spec), len(times))


def _as_vector(value):
    """Three numbers, or an axis name."""
    names = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}
    if isinstance(value, str):
        key = value.strip().lower()
        if key not in names:
            raise ValueError("%r is not an axis name or three numbers" % value)
        return np.asarray(names[key], dtype=float)
    vec = np.asarray(value, dtype=float).ravel()
    if vec.size != 3 or not np.all(np.isfinite(vec)):
        raise ValueError("expected three finite numbers, got %r" % (value,))
    return vec


def _as_vector_or_zero(value):
    return np.zeros(3) if value is None else _as_vector(value)
