#!/usr/bin/env python3
"""Horizons as the tracer's absorbing surfaces: analytic Kerr-Schild spheroids.

In Kerr-Schild Cartesian coordinates centred on a hole of mass m and spin
parameter a = |S|/m along the unit vector s, the surfaces of constant
Boyer-Lindquist-like radius r are the oblate spheroids

    rho^2 / (r^2 + a^2) + zeta^2 / r^2 = 1,      zeta = d . s,  rho^2 = |d|^2 - zeta^2,

and the outer horizon is r = r+ = m + sqrt(m^2 - a^2): polar semi-axis r+,
equatorial semi-axis sqrt(r+^2 + a^2) = sqrt(2 m r+). For the superposed
Kerr-Schild metric of the AnalyticalSpacetime runs this is each hole's horizon
as placed by the trajectory table (the boost, v ~ 0.09 there, is ignored: a
0.4% effect on the shape). A field line ends on a horizon when it enters its
spheroid; ``scale`` > 1 enlarges both semi-axes, e.g. to check that the
classification depends on the exact surface only for grazing lines.
"""

import numpy as np


class Horizons(object):
    """Spheroids ``rho^2 / a_eq^2 + zeta^2 / a_pol^2 < 1`` about ``centres``.

    ``centres`` and ``axes`` (unit) are ``(n, 3)``, ``a_eq``/``a_pol`` the
    equatorial and polar semi-axes ``(n,)``. ``names`` label the ends of the
    lines that reach them (``BH1``, ``BH2``).
    """

    def __init__(self, centres, axes, a_eq, a_pol, names):
        self.centres = np.asarray(centres, dtype=float).reshape(-1, 3)
        axes = np.asarray(axes, dtype=float).reshape(-1, 3)
        self.axes = axes / np.linalg.norm(axes, axis=1)[:, None]
        self.a_eq = np.asarray(a_eq, dtype=float).ravel()
        self.a_pol = np.asarray(a_pol, dtype=float).ravel()
        self.names = [str(n) for n in names]
        n = len(self.centres)
        if not (len(self.axes) == len(self.a_eq) == len(self.a_pol)
                == len(self.names) == n):
            raise ValueError("centres, axes, semi-axes and names must agree "
                             "in length")
        if np.any(self.a_eq <= 0) or np.any(self.a_pol <= 0):
            raise ValueError("semi-axes must be positive")

    def __len__(self):
        return len(self.centres)

    @classmethod
    def kerr_schild(cls, positions, spins, masses, scale=1.0,
                    names=("BH1", "BH2")):
        """From positions ``(n, 3)``, spin vectors ``a = S/m`` ``(n, 3)`` and masses.

        The spin vector is the dimensional ``a`` (length ``a = chi m``), as the
        trajectory tables store it (``a1x`` ..). A non-spinning hole gets a
        sphere of radius ``2 m`` about the z axis.
        """
        positions = np.asarray(positions, dtype=float).reshape(-1, 3)
        spins = np.asarray(spins, dtype=float).reshape(-1, 3)
        masses = np.asarray(masses, dtype=float).ravel()
        a = np.linalg.norm(spins, axis=1)
        if np.any(a > masses):
            raise ValueError("spin a = %s exceeds the mass %s" % (a, masses))
        r_plus = masses + np.sqrt(masses ** 2 - a ** 2)
        axes = np.where(a[:, None] > 0, spins / np.maximum(a, 1e-300)[:, None],
                        np.array([0.0, 0.0, 1.0]))
        return cls(positions, axes, scale * np.sqrt(r_plus ** 2 + a ** 2),
                   scale * r_plus, names[:len(positions)])

    def kernel_args(self):
        """``(centres, axes, 1 / a_eq^2, 1 / a_pol^2)`` for the numba kernels."""
        return (np.ascontiguousarray(self.centres),
                np.ascontiguousarray(self.axes),
                1.0 / self.a_eq ** 2, 1.0 / self.a_pol ** 2)

    def inside(self, points):
        """Index of the horizon containing each point, -1 for none."""
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        out = np.full(len(points), -1, dtype=int)
        for h in range(len(self)):
            d = points - self.centres[h]
            zeta = d @ self.axes[h]
            rho2 = np.einsum("ij,ij->i", d, d) - zeta ** 2
            hit = rho2 / self.a_eq[h] ** 2 + zeta ** 2 / self.a_pol[h] ** 2 < 1.0
            out[(out < 0) & hit] = h
        return out

    def describe(self):
        return ["%s at %s, axis %s, semi-axes %.4f (equatorial) / %.4f (polar)"
                % (n, np.round(c, 4).tolist(), np.round(s, 4).tolist(), e, p)
                for n, c, s, e, p in zip(self.names, self.centres, self.axes,
                                         self.a_eq, self.a_pol)]
