#!/usr/bin/env python3
"""Tests of the shell volumes and integrands on synthetic AMR hierarchies.

    python test_shell_cells.py     (one line per check, then ALL OK)

A three-level hierarchy with Carpet's layout (vertex-centred, 2:1, components
split along x with three ghost zones each) is cut by a shell whose centre is
off the grid and whose pole is tilted, so the shell crosses both refinement
boundaries. Fields with closed-form integrals over a shell r1 < r < r2 about
the centre (sqrt(gamma) = s, V the shell volume, dr = r2 - r1):

    calB = s B0 pole       Phi_z = s B0 V / (2 r2),  Phi_r = s B0 2 pi (r2^3 - r1^3) / (3 dr),  Phi_phi = 0
    calB = s B0 e_perp     Phi_phi = s B0 (r2^2 - r1^2),  Phi_r = s B0 2 pi (r2^3 - r1^3) / (3 dr),  Phi_z = 0
    calB = s B0 (R/R0) phi_hat   Phi_phi = s B0 V / (2 pi R0),  Phi_r = Phi_z = 0
    calB = s B0 r_hat      Phi_r = s B0 V / dr,  Phi_z = s B0 2 pi (r2^3 - r1^3) / (6 r2),  Phi_phi = 0

(e_perp a unit vector normal to the pole). The sums over grid points approach
them as the spacing falls (the cells on the bounding spheres are counted whole
or not at all). The uniform field normal to the pole is also the test of the
pole axis: |calB . phi_hat| / R = B0 |sin phi| / R is singular there, and with
the centre on (or 1e-7 off) a grid point some points lie on (or next to) the
axis.
"""

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "Resample"))

from amr_blocks import Component     # noqa: E402
import shell_cells                   # noqa: E402

FAILURES = []
NAMES = ["Bx", "By", "Bz", "gxx", "gxy", "gxz", "gyy", "gyz", "gzz"]
R1, R2 = 4.0, 5.0
B0 = 1.7
R0 = 4.5                            # toroidal field B_phi = B0 R / R0
CONF = 1.3                          # gamma_ij = CONF delta_ij
S = CONF ** 1.5                     # sqrt(gamma)


def check(name, ok, detail=""):
    print("%s %s%s" % ("ok  " if ok else "FAIL", name,
                       ("  (%s)" % detail) if detail else ""), flush=True)
    if not ok:
        FAILURES.append(name)


def hierarchy(scale=1.0, nghost=3):
    """Levels dx = 0.5, 0.25, 0.125 (times scale) over +-8, +-4.5, +-3."""
    levels = [(0.5 * scale, 8.0, 2), (0.25 * scale, 4.5, 3),
              (0.125 * scale, 3.0, 2)]
    comps = []
    for level, (dx, half, n_split) in enumerate(levels):
        lo = np.full(3, -half)
        npts = np.rint(2 * half / dx).astype(int) + 1
        edges = np.linspace(0, npts, n_split + 1).astype(int)
        for s in range(n_split):
            own0 = np.array([edges[s], 0, 0])
            own1 = np.array([edges[s + 1] - 1, npts - 1, npts - 1])
            i0 = own0 - nghost
            n = own1 - own0 + 1 + 2 * nghost
            comps.append(Component(level, s, ("synthetic", level, s), n,
                                   lo + i0 * dx, np.full(3, dx),
                                   np.full(3, nghost)))
    return comps


def make_read(kind, centre, pole):
    pole = np.asarray(pole, float) / np.linalg.norm(pole)

    def field(X, Y, Z):
        d = np.stack([X - centre[0], Y - centre[1], Z - centre[2]], axis=-1)
        if kind == "uniform":
            return np.broadcast_to(B0 * pole, d.shape)
        if kind == "perpendicular":
            e = np.cross(pole, [1.0, 0.0, 0.0])
            return np.broadcast_to(B0 * e / np.linalg.norm(e), d.shape)
        if kind == "radial":
            r = np.maximum(np.linalg.norm(d, axis=-1), 1e-12)
            return B0 * d / r[..., None]
        if kind == "toroidal":
            # B_phi growing as R from the axis: regular there, as a real
            # field is (the singular 1/R is the perpendicular field's test).
            return B0 * np.cross(pole, d) / R0
        raise ValueError(kind)

    def read(var, comp, i0, i1):
        x, y, z = comp.coords(i0, i1)
        Z, Y, X = np.meshgrid(z, y, x, indexing="ij")
        if var.startswith("g"):
            diag = var in ("gxx", "gyy", "gzz")
            return np.full(X.shape, CONF if diag else 0.0)
        # B^i = B0 x unit vector, so calB = sqrt(gamma) B^i = S B0 x unit vector.
        return field(X, Y, Z)[..., "xyz".index(var[1])]

    return read


def integrate(comps, read, centre, pole, r1=R1, r2=R2, cls=None, n_class=1,
              n_bin=40):
    sel = shell_cells.select_shell(comps, centre, r1, r2)
    pts, w, lev, dx = shell_cells.selection_points(sel)
    vals = shell_cells.read_selected(read, sel, NAMES)
    sqrtg = shell_cells.sqrt_gamma(*[vals[n] for n in NAMES[3:]])
    calB = sqrtg[:, None] * np.stack([vals[n] for n in NAMES[:3]], axis=1)
    f_phi, f_r, f_z, zeta, n_axis = shell_cells.integrands(pts, calB, centre,
                                                           pole, dx, r1, r2)
    if cls is None:
        cls = np.zeros(len(pts), dtype=np.int64)
    sums = shell_cells.class_sums(cls, n_class, w, f_phi, f_r, f_z, zeta, r1,
                                  r2, n_bin)
    return sums, lev, n_axis, len(pts)


def exact(kind):
    V = 4 * np.pi / 3 * (R2 ** 3 - R1 ** 3)
    cos_int = 2 * np.pi * (R2 ** 3 - R1 ** 3) / 3        # int |cos theta| dV
    if kind == "uniform":
        return {"Phi_z": S * B0 * V / (2 * R2),
                "Phi_r": S * B0 * cos_int / (R2 - R1), "Phi_phi": 0.0}
    if kind == "perpendicular":
        return {"Phi_phi": S * B0 * (R2 ** 2 - R1 ** 2),
                "Phi_r": S * B0 * cos_int / (R2 - R1), "Phi_z": 0.0}
    if kind == "toroidal":
        return {"Phi_phi": S * B0 * V / (2 * np.pi * R0), "Phi_r": 0.0,
                "Phi_z": 0.0}
    return {"Phi_r": S * B0 * V / (R2 - R1),
            "Phi_z": S * B0 * cos_int / (2 * R2), "Phi_phi": 0.0}


def test_volumes():
    comps = hierarchy()
    sel = shell_cells.select_shell(comps, np.zeros(3), 0.0, 1e3)
    _, w, lev, _ = shell_cells.selection_points(sel)
    box = (16.0 + 0.5) ** 3          # coarsest owned box: +-(8 + dx/2)
    check("weights of all levels tile the coarsest owned box",
          abs(w.sum() / box - 1) < 1e-12, "relative error %.1e"
          % (w.sum() / box - 1))
    for level, dx, half in ((1, 0.25, 4.5), (2, 0.125, 3.0)):
        inner = (2 * half + dx) ** 3
        own = w[lev >= level].sum()
        check("levels >= %d own exactly their box" % level,
              abs(own / inner - 1) < 1e-12, "%.1e" % (own / inner - 1))
    partial = np.mean((w[lev == 0] > 0) & (w[lev == 0] < 0.5 ** 3 - 1e-12))
    check("coarse cells on the level edge are split, not dropped",
          partial > 0, "%.2f%% of level-0 points partial" % (100 * partial))


def test_integrals():
    centre = np.array([0.37, -0.21, 0.13])
    pole = np.array([0.3, -0.1, 1.0])
    V = 4 * np.pi / 3 * (R2 ** 3 - R1 ** 3)
    errors = {}
    kinds = ("uniform", "perpendicular", "toroidal", "radial")
    for scale in (1.0, 0.5):
        comps = hierarchy(scale)
        for kind in kinds:
            sums, lev, n_axis, n = integrate(comps, make_read(kind, centre, pole),
                                             centre, pole)
            ref = exact(kind)
            norm = max(abs(v) for v in ref.values())
            for q, val in ref.items():
                errors[(scale, kind, q)] = abs(sums[q][0] - val) / norm
            errors[(scale, kind, "volume")] = abs(sums["volume"][0] / V - 1)
        levels = sorted(set(lev.tolist()))
    for kind in kinds:
        for q in ("Phi_phi", "Phi_r", "Phi_z", "volume"):
            e1, e2 = errors[(1.0, kind, q)], errors[(0.5, kind, q)]
            # Below ~1e-4 the sub-sampled cut cells set the floor.
            ok = e2 < 5e-3 and (e2 < 1e-4 or e2 <= e1)
            check("%s field, %s" % (kind, q), ok,
                  "relative error %.1e -> %.1e at half the spacing" % (e1, e2))
    check("the shell crosses every level", levels == [0, 1, 2], str(levels))


def test_axis():
    """Centre on / next to a grid point: points on the axis, 1/R singular."""
    pole = np.array([0.0, 0.0, 1.0])
    ref = exact("perpendicular")["Phi_phi"]
    for label, centre in (("on", np.zeros(3)),
                          ("1e-7 off", np.array([1e-7, 2e-7, 3e-7]))):
        sums, _, n_axis, _ = integrate(hierarchy(0.5),
                                       make_read("perpendicular", centre, pole),
                                       centre, pole)
        err = abs(sums["Phi_phi"][0] / ref - 1)
        check("Phi_phi with the axis %s grid points: finite, accurate" % label,
              np.isfinite(sums["Phi_phi"][0]) and err < 2e-3,
              "%d near-axis cells, relative error %.1e" % (n_axis, err))


def test_classes():
    centre = np.array([0.37, -0.21, 0.13])
    pole = np.array([0.0, 0.0, 1.0])
    comps = hierarchy()
    read = make_read("radial", centre, pole)
    whole, _, _, n = integrate(comps, read, centre, pole)
    rng = np.random.default_rng(1)
    cls = rng.integers(0, 5, size=n)
    split, _, _, _ = integrate(comps, read, centre, pole, cls=cls, n_class=5)
    ok = all(np.isclose(split[q].sum(), whole[q][0], rtol=1e-12)
             for q in ("Phi_phi", "Phi_r", "Phi_z", "volume"))
    check("classes add up to the unmasked integrals", ok)
    prof = split["profile"].mean(axis=1)
    check("mean of the zeta profile = Phi_z, per class",
          np.allclose(prof, split["Phi_z"], rtol=1e-12),
          "%s vs %s" % (np.round(prof, 4), np.round(split["Phi_z"], 4)))


def main():
    test_volumes()
    test_integrals()
    test_axis()
    test_classes()
    if FAILURES:
        print("FAILED: %d check(s): %s" % (len(FAILURES), "; ".join(FAILURES)))
        sys.exit(1)
    print("ALL OK")


if __name__ == "__main__":
    main()
