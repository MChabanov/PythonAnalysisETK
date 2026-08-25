#!/usr/bin/env python3
"""Geometry of an arbitrary 2D plane cut out of a 3D domain.

Backend-agnostic (numpy only). Used by the 3D resampling backends to turn a
``(normal, origin)`` plane specification plus an in-plane ``(u, v)`` grid into
3D sample points, and - the part that makes 3D resampling affordable - to
decide *cheaply*, from grid metadata alone, whether an AMR component is worth
reading at all and which corner of the ``(u, v)`` grid it can contribute to.

The in-plane axes
-----------------
A plane fixes its normal but not the rotation of the coordinates within it, so
a convention is needed. The default: **drop the coordinate axis most aligned
with the normal and keep the other two, in x < y < z order** (projected onto
the plane and orthonormalised). For an axis-aligned normal this reproduces the
2D pipeline's conventions exactly - normal z gives (u, v) = (x, y), normal y
gives (x, z), normal x gives (y, z) - so an axis-aligned 3D cut is directly
comparable with the corresponding output of ``resample_2d.py``. It also makes
the frame independent of the *sign* of the normal, since flipping the normal
should not flip the picture. Pass ``u_axis`` to override it.

Note that (u, v, n) is therefore not always right-handed (it is not for the xz
plane, matching the 2D pipeline). Handedness has no effect on the output: the
result is a scalar field sampled on the (u, v) grid.
"""

import numpy as np

# Plane name -> (u axis index, v axis index); the normal is the missing axis.
SHORTHAND = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}

_AXIS_VECTORS = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}

# Directions this close to zero-length (or to lying in the plane) are rejected.
_TINY = 1e-12


def _as_vector(value, what):
    """Coerce a config value to a 3-vector, accepting 'x'/'y'/'z' as names."""
    if isinstance(value, str):
        key = value.strip().lower()
        if key not in _AXIS_VECTORS:
            raise ValueError(
                "%s: %r is not an axis name (use x, y, z or three numbers)"
                % (what, value))
        value = _AXIS_VECTORS[key]
    vec = np.asarray(value, dtype=float).ravel()
    if vec.size != 3 or not np.all(np.isfinite(vec)):
        raise ValueError("%s must be three finite numbers (got %r)" % (what, value))
    return vec


def _unit(vec, what):
    norm = float(np.linalg.norm(vec))
    if norm <= _TINY:
        raise ValueError("%s must not be (numerically) the zero vector" % what)
    return vec / norm


def plane_basis(normal, u_axis=None):
    """Return orthonormal ``(u_hat, v_hat, n_hat)`` for the plane.

    ``normal`` need not be normalised. If ``u_axis`` is given it is projected
    into the plane and used as the u direction (v completes a right-handed
    frame); otherwise the axis-dropping convention described in the module
    docstring is used.
    """
    n_hat = _unit(_as_vector(normal, "plane normal"), "plane normal")

    if u_axis is not None:
        raw = _as_vector(u_axis, "plane u_axis")
        u_hat = _unit(raw - np.dot(raw, n_hat) * n_hat,
                      "plane u_axis (after removing its normal component; it "
                      "must not be parallel to the normal)")
        return u_hat, np.cross(n_hat, u_hat), n_hat

    # Drop the coordinate axis most aligned with the normal, keep the other two
    # in index order, project into the plane and orthonormalise (Gram-Schmidt).
    dropped = int(np.argmax(np.abs(n_hat)))
    first, second = [k for k in range(3) if k != dropped]
    e_first, e_second = np.eye(3)[first], np.eye(3)[second]

    u_hat = _unit(e_first - np.dot(e_first, n_hat) * n_hat, "in-plane u axis")
    v = e_second - np.dot(e_second, n_hat) * n_hat
    v_hat = _unit(v - np.dot(v, u_hat) * u_hat, "in-plane v axis")
    return u_hat, v_hat, n_hat


def _clip_halfplane(poly, a, b, c):
    """Clip a convex polygon by the half plane ``a*u + b*v <= c``.

    ``poly`` is a list of ``(u, v)`` vertices in order; the result is the
    clipped polygon (possibly empty). Sutherland-Hodgman; only the resulting
    bounding box is used by callers, so duplicated vertices are harmless.
    """
    out = []
    n = len(poly)
    for k in range(n):
        p, q = poly[k], poly[(k + 1) % n]
        fp = a * p[0] + b * p[1] - c
        fq = a * q[0] + b * q[1] - c
        if fp <= 0.0:
            out.append(p)
        if (fp < 0.0) != (fq < 0.0):
            t = fp / (fp - fq)
            out.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
    return out


class PlaneSpec(object):
    """A plane in 3D plus the orthonormal in-plane frame used to sample it.

    A point of the in-plane grid at coordinates ``(u, v)`` sits at
    ``origin + u * u_hat + v * v_hat`` in the simulation's coordinates.
    """

    def __init__(self, normal, origin=(0.0, 0.0, 0.0), u_axis=None, name=None):
        self.u_hat, self.v_hat, self.n_hat = plane_basis(normal, u_axis)
        self.origin = _as_vector(origin, "plane origin")
        self._name = name

    # -- construction ------------------------------------------------------

    @classmethod
    def from_config(cls, plane):
        """Build from the config's ``plane`` value.

        Accepts the 2D pipeline's shorthand strings (``xy``/``xz``/``yz``,
        meaning the plane through the origin with that normal) or a mapping:

            plane:
              normal: [0.0, 1.0, 1.0]   # or an axis name: z
              origin: [0.0, 0.0, 0.0]   # optional, a point on the plane
              u_axis: [1.0, 0.0, 0.0]   # optional, fixes the in-plane rotation
        """
        if isinstance(plane, str):
            key = plane.strip().lower()
            if key not in SHORTHAND:
                raise ValueError(
                    "plane %r is not one of %s; use a mapping with a `normal` "
                    "for any other plane"
                    % (plane, ", ".join(sorted(SHORTHAND))))
            i_u, i_v = SHORTHAND[key]
            normal = np.eye(3)[3 - i_u - i_v]
            return cls(normal, name=key)

        if isinstance(plane, dict):
            unknown = sorted(set(plane) - {"normal", "origin", "u_axis"})
            if unknown:
                raise ValueError("unknown plane key(s): %s" % ", ".join(unknown))
            if plane.get("normal") is None:
                raise ValueError("a plane mapping needs a `normal`")
            return cls(plane["normal"], plane.get("origin") or (0.0, 0.0, 0.0),
                       plane.get("u_axis"))

        raise ValueError(
            "plane must be one of %s or a mapping with a `normal` (got %r)"
            % (", ".join(sorted(SHORTHAND)), plane))

    @classmethod
    def from_frame(cls, origin, u_hat, v_hat, n_hat, name=None):
        """Build directly from an already-orthonormal frame, skipping the
        default axis convention.

        Used for time-dependent planes, whose frame is precomputed per
        iteration (see ``plane_motion``): the convention that fixes the
        in-plane rotation there is continuity in time, not
        :func:`plane_basis`.
        """
        self = cls.__new__(cls)
        self.u_hat = np.asarray(u_hat, dtype=float)
        self.v_hat = np.asarray(v_hat, dtype=float)
        self.n_hat = np.asarray(n_hat, dtype=float)
        self.origin = np.asarray(origin, dtype=float)
        self._name = name
        return self

    # -- description -------------------------------------------------------

    @property
    def label(self):
        """Short name for the ``plane`` attribute of the output files.

        The 2D pipeline's ``xy``/``xz``/``yz`` when the plane really is that
        one (through the origin), those names with the offset appended when it
        is parallel to it, and ``oblique`` otherwise.
        """
        if self._name is not None:
            return self._name
        axis = np.argmax(np.abs(self.n_hat))
        if abs(abs(self.n_hat[axis]) - 1.0) > 1e-9:
            return "oblique"
        name = {2: "xy", 1: "xz", 0: "yz"}[int(axis)]
        offset = float(np.dot(self.origin, self.n_hat))
        if abs(offset) <= 1e-12:
            return name
        return "%s@%s=%g" % (name, "xyz"[int(axis)], self.origin[int(axis)])

    def describe(self):
        """One-line human-readable summary for the startup log."""
        fmt = lambda v: "[% .4f % .4f % .4f]" % tuple(v)
        return ("%s  normal %s  origin %s  u %s  v %s"
                % (self.label, fmt(self.n_hat), fmt(self.origin),
                   fmt(self.u_hat), fmt(self.v_hat)))

    # -- sampling ----------------------------------------------------------

    def points(self, u, v):
        """3D coordinates of the in-plane grid as three ``(nu, nv)`` arrays."""
        col = np.asarray(u, dtype=float)[:, None]
        row = np.asarray(v, dtype=float)[None, :]
        return tuple(self.origin[d] + col * self.u_hat[d] + row * self.v_hat[d]
                     for d in range(3))

    # -- cheap culling against AMR component boxes -------------------------

    def intersects_box(self, x0, x1):
        """Whether the (infinite) plane cuts the axis-aligned box [x0, x1].

        Box-plane test via the box's support along the normal, so it costs no
        array work: the plane hits the box iff the distance from its centre is
        within the box's half-extent projected on the normal.
        """
        x0, x1 = np.asarray(x0, float), np.asarray(x1, float)
        centre_dist = float(np.dot(0.5 * (x0 + x1) - self.origin, self.n_hat))
        reach = float(np.dot(np.abs(self.n_hat), 0.5 * (x1 - x0)))
        return abs(centre_dist) <= reach

    def uv_window(self, x0, x1, u, v):
        """Index bounds ``(i0, i1, j0, j1)`` into the ``(u, v)`` grid, or None.

        Only grid points inside ``u[i0:i1]`` x ``v[j0:j1]`` can lie inside the
        box ``[x0, x1]``; None means the plane's sampled patch misses the box
        entirely, so the component never has to be read.

        The exact feasible set is ``{(u, v) : x0_d <= origin_d + u*u_d + v*v_d
        <= x1_d for d in x, y, z}`` - six half planes, hence a convex polygon.
        It is obtained by clipping the ``(u, v)`` rectangle against them; the
        polygon's bounding box then gives the index window. ``u`` and ``v``
        must be ascending.
        """
        x0, x1 = np.asarray(x0, float), np.asarray(x1, float)
        poly = [(u[0], v[0]), (u[-1], v[0]), (u[-1], v[-1]), (u[0], v[-1])]

        for d in range(3):
            a, b = self.u_hat[d], self.v_hat[d]
            lo, hi = x0[d] - self.origin[d], x1[d] - self.origin[d]
            if abs(a) <= _TINY and abs(b) <= _TINY:
                # The sampled patch has no extent along d (an axis-aligned
                # plane, for its normal direction): pure feasibility test.
                if not (lo <= 0.0 <= hi):
                    return None
                continue
            poly = _clip_halfplane(poly, a, b, hi)      # +a*u +b*v <=  hi
            if not poly:
                return None
            poly = _clip_halfplane(poly, -a, -b, -lo)   # -a*u -b*v <= -lo
            if not poly:
                return None

        us = [p[0] for p in poly]
        vs = [p[1] for p in poly]
        # One cell of slack on each side, so a point sitting exactly on the
        # polygon boundary cannot be lost to rounding.
        i0 = max(0, int(np.searchsorted(u, min(us), side="left")) - 1)
        i1 = min(len(u), int(np.searchsorted(u, max(us), side="right")) + 1)
        j0 = max(0, int(np.searchsorted(v, min(vs), side="left")) - 1)
        j1 = min(len(v), int(np.searchsorted(v, max(vs), side="right")) + 1)
        if i1 <= i0 or j1 <= j0:
            return None
        return i0, i1, j0, j1
