#!/usr/bin/env python3
"""Geometry of spheres cut out of a 3D domain, sampled on a (theta, phi) grid.

Backend-agnostic (numpy only); the sphere counterpart of ``plane_geom.py``,
used by ``resample_sphere_data_postcactus.py``. It turns a centre, a frame and
one or more radii into 3D sample points and - the part that keeps 3D reads
small - decides from grid metadata alone whether an AMR component can reach a
sphere at all, and which part of the (theta, phi) grid it can supply.

The grid
--------
theta is cell-centred, ``theta_j = (j + 1/2) pi / n_theta``: no point sits on a
pole, where a whole row would be ``n_phi`` copies of one point and theta-hat,
phi-hat are undefined. phi is periodic, ``phi_k = 2 pi k / n_phi``, without
the duplicate endpoint. ``n_phi = 2 n_theta`` gives square cells at the
equator. For surface integrals :func:`area_weights` gives each point its
cell's exact area, so the weights sum to ``4 pi r^2``; for a smooth integrand
the sum is second order in theta and spectrally accurate in phi.

The frame
---------
theta is measured from the pole ``e3``, phi from ``e1`` towards
``e2 = e3 x e1``, so ``(e1, e2, e3)`` is right-handed and a grid point is

    centre + r (sin(theta) cos(phi) e1 + sin(theta) sin(phi) e2 + cos(theta) e3).

The backend takes the centre, ``e1`` and ``e3`` from the same frame table as a
moving plane (its origin, u axis and normal), so a sphere can follow a body
and its pole can turn with a spin or the orbit.

Culling
-------
:meth:`SphereSpec.windows` must never miss a point; it may return too many,
because the backend tests every candidate point against the component box
exactly. Per radius:

1. *Shell test* (exact): the sphere ``|x - c| = r`` meets the box iff
   ``d_min <= r <= d_max``, the distances from ``c`` to the box's nearest point
   and to its farthest corner.
2. *Cap*: the box lies inside its bounding ball (centre ``b``, radius ``rho``).
   The part of the sphere inside that ball is a cap about the direction of
   ``b - c`` with angular radius ``alpha``, where
   ``cos(alpha) = (r^2 + D^2 - rho^2) / (2 r D)`` and ``D = |b - c|``. The
   cap's theta range is ``theta_b -+ alpha``; unless it contains a pole, its
   phi range is ``phi_b -+ arcsin(sin(alpha) / sin(theta_b))``. Both are exact
   for a cap. A phi range across phi = 0 becomes two windows.

Each index window gets one grid cell of slack on either side, as in
``plane_geom.PlaneSpec.uv_window``, so a point on the boundary cannot be lost to
rounding.
"""

import numpy as np

_TINY = 1e-12

_AXIS_VECTORS = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}


# ---------------------------------------------------------------------------
# The grid
# ---------------------------------------------------------------------------

def theta_grid(n):
    """Cell-centred polar angles ``(j + 1/2) pi / n``, j = 0 .. n-1."""
    return (np.arange(int(n)) + 0.5) * (np.pi / int(n))


def phi_grid(n):
    """Periodic azimuths ``2 pi k / n``, k = 0 .. n-1 (no duplicate 2 pi)."""
    return np.arange(int(n)) * (2.0 * np.pi / int(n))


def area_weights(n_theta, n_phi, radius=1.0):
    """Exact area of each grid cell, shape ``(n_theta, n_phi)``.

    ``r^2 dphi (cos(theta_j - dtheta/2) - cos(theta_j + dtheta/2))``: the theta
    cells tile [0, pi] and the phi cells [0, 2 pi), so the weights sum to
    ``4 pi r^2`` exactly. ``sum(w * f)`` is the midpoint-value approximation
    of the surface integral of ``f``.
    """
    theta = theta_grid(n_theta)
    half = 0.5 * np.pi / int(n_theta)
    band = np.cos(theta - half) - np.cos(theta + half)
    dphi = 2.0 * np.pi / int(n_phi)
    return np.repeat((radius ** 2 * dphi * band)[:, None], int(n_phi), axis=1)


# ---------------------------------------------------------------------------
# The frame
# ---------------------------------------------------------------------------

def _as_vector(value, what):
    """Coerce a config value to a 3-vector, accepting 'x'/'y'/'z' as names."""
    if isinstance(value, str):
        key = value.strip().lower()
        if key not in _AXIS_VECTORS:
            raise ValueError("%s: %r is not an axis name (use x, y, z or three "
                             "numbers)" % (what, value))
        value = _AXIS_VECTORS[key]
    vec = np.asarray(value, dtype=float).ravel()
    if vec.size != 3 or not np.all(np.isfinite(vec)):
        raise ValueError("%s must be three finite numbers (got %r)"
                         % (what, value))
    return vec


def _unit(vec, what):
    norm = float(np.linalg.norm(vec))
    if norm <= _TINY:
        raise ValueError("%s must not be (numerically) the zero vector" % what)
    return vec / norm


def sphere_frame(pole, phi_zero):
    """Right-handed ``(e1, e2, e3)``: e3 along ``pole``, e1 along ``phi_zero``.

    ``phi_zero`` is projected perpendicular to the pole first, so it only has
    to point somewhere off the pole axis.
    """
    e3 = _unit(_as_vector(pole, "sphere pole"), "sphere pole")
    raw = _as_vector(phi_zero, "sphere phi_zero")
    e1 = _unit(raw - np.dot(raw, e3) * e3,
               "sphere phi_zero (after removing its component along the pole; "
               "it must not be parallel to the pole)")
    return e1, np.cross(e3, e1), e3


# ---------------------------------------------------------------------------
# The sphere(s)
# ---------------------------------------------------------------------------

class SphereSpec(object):
    """Concentric spheres about ``centre`` in the frame ``(e1, e2, e3)``.

    ``radii`` is one radius or several; the sample points of radius ``k`` are
    sheet ``k`` of the ``(n_radii, n_theta, n_phi)`` arrays :meth:`points`
    returns, and :meth:`windows` indexes into such arrays.
    """

    def __init__(self, centre, radii, e1, e3):
        self.centre = np.asarray(centre, dtype=float).reshape(3)
        self.radii = np.atleast_1d(np.asarray(radii, dtype=float))
        if self.radii.ndim != 1 or np.any(self.radii <= 0.0):
            raise ValueError("sphere radii must be positive numbers (got %r)"
                             % (radii,))
        self.e1, self.e2, self.e3 = sphere_frame(e3, e1)
        # Rows are the frame vectors: rot @ (x - centre) gives frame components.
        self._rot = np.vstack([self.e1, self.e2, self.e3])

    def describe(self):
        """One-line human-readable summary for the startup log."""
        fmt = lambda v: "[% .4f % .4f % .4f]" % tuple(v)
        return ("centre %s  pole %s  phi = 0 along %s  radii %s"
                % (fmt(self.centre), fmt(self.e3), fmt(self.e1),
                   ", ".join("%g" % r for r in self.radii)))

    # -- sampling ----------------------------------------------------------

    def directions(self, theta, phi):
        """Unit radial vectors as three ``(n_theta, n_phi)`` arrays."""
        st = np.sin(np.asarray(theta, dtype=float))[:, None]
        ct = np.cos(np.asarray(theta, dtype=float))[:, None]
        sp = np.sin(np.asarray(phi, dtype=float))[None, :]
        cp = np.cos(np.asarray(phi, dtype=float))[None, :]
        a, b, c = st * cp, st * sp, ct * np.ones_like(sp)
        return tuple(a * self.e1[d] + b * self.e2[d] + c * self.e3[d]
                     for d in range(3))

    def points(self, theta, phi):
        """3D coordinates of every grid point, three ``(n_radii, n_theta,
        n_phi)`` arrays."""
        r = self.radii[:, None, None]
        return tuple(self.centre[d] + r * n[None, :, :]
                     for d, n in enumerate(self.directions(theta, phi)))

    # -- cheap culling against AMR component boxes -------------------------

    def meets_shell(self, x0, x1, radius):
        """Whether the sphere of ``radius`` passes through the box [x0, x1]."""
        lo = np.asarray(x0, float) - self.centre
        hi = np.asarray(x1, float) - self.centre
        near = np.maximum(np.maximum(lo, -hi), 0.0)
        far = np.maximum(np.abs(lo), np.abs(hi))
        slack = _TINY * max(1.0, radius)
        return (float(np.linalg.norm(near)) <= radius + slack and
                radius <= float(np.linalg.norm(far)) + slack)

    def windows(self, x0, x1, theta, phi):
        """Index windows of the grid whose points can lie in the box [x0, x1].

        A list of ``(k, theta slice, phi slice)`` tuples into the
        ``(n_radii, n_theta, n_phi)`` output; empty when no sphere reaches the
        box. ``theta`` must be ascending; ``phi`` must be :func:`phi_grid`
        (uniform from 0), since the phi windows are found arithmetically so
        that they can wrap around phi = 0.
        """
        x0, x1 = np.asarray(x0, float), np.asarray(x1, float)
        rel = self._rot.dot(0.5 * (x0 + x1) - self.centre)
        dist = float(np.linalg.norm(rel))
        rho = 0.5 * float(np.linalg.norm(x1 - x0))
        theta_b = float(np.arccos(np.clip(rel[2] / dist, -1.0, 1.0))) \
            if dist > 0.0 else 0.0
        phi_b = float(np.arctan2(rel[1], rel[0]))

        out = []
        for k, radius in enumerate(self.radii):
            if not self.meets_shell(x0, x1, radius):
                continue
            if dist <= _TINY * max(1.0, radius):
                alpha = np.pi                  # the ball is centred on c
            else:
                cos_a = (radius * radius + dist * dist - rho * rho) \
                    / (2.0 * radius * dist)
                alpha = float(np.arccos(np.clip(cos_a, -1.0, 1.0)))

            t_lo, t_hi = theta_b - alpha, theta_b + alpha
            if t_lo <= 0.0 or t_hi >= np.pi:
                phi_windows = [slice(0, len(phi))]     # the cap holds a pole
            else:
                half = float(np.arcsin(min(1.0, np.sin(alpha) / np.sin(theta_b))))
                phi_windows = _phi_windows(len(phi), phi_b - half, phi_b + half)

            i0 = max(0, int(np.searchsorted(theta, t_lo, side="left")) - 1)
            i1 = min(len(theta),
                     int(np.searchsorted(theta, t_hi, side="right")) + 1)
            if i1 <= i0:
                continue
            out.extend((k, slice(i0, i1), w) for w in phi_windows)
        return out


def _phi_windows(n_phi, lo, hi):
    """Slices of a :func:`phi_grid` holding every point with phi in [lo, hi].

    ``lo``/``hi`` may lie outside [0, 2 pi): the points are ``k h`` for integer
    ``k`` and ``h = 2 pi / n_phi``, so the candidate indices are found on the
    unwrapped line (one index of slack each side) and then folded back, as one
    slice or - across phi = 0 - two.
    """
    h = 2.0 * np.pi / n_phi
    k_lo = int(np.floor(lo / h)) - 1
    k_hi = int(np.ceil(hi / h)) + 1
    count = k_hi - k_lo + 1
    if count >= n_phi:
        return [slice(0, n_phi)]
    start = k_lo % n_phi
    if start + count <= n_phi:
        return [slice(start, start + count)]
    return [slice(start, n_phi), slice(0, start + count - n_phi)]
