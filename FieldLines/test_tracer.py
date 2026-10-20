#!/usr/bin/env python3
"""Tests of the field-line tracer on analytic fields.

    python test_tracer.py          (prints one line per check, then ALL OK)

The fields are sampled on synthetic AMR hierarchies (``AMRVectorField.
from_function``) of one and of two levels, with Carpet's ghost-zone layout, and
traced with a unit-sphere "horizon" at the origin. Each field has field lines
known in closed form, so the classification of every seed is known:

* uniform B = z: lines through R < 1 hit the sphere one way and leave the box
  the other; the rest leave both ways;
* dipole: lines r = L sin^2(theta) close on the sphere unless they leave the
  box about the equator (L times the larger of |cos phi|, |sin phi| above the
  half-width), in which case a seed reaches the sphere one way only;
  L = r^3 / R^2 is conserved along each line, which measures the integration
  error;
* split monopole B = sign(z) r_hat / r^2: every line is radial, one end on the
  sphere, the other leaving the box;
* toroidal B = phi_hat: circles, never ending - both halves reach the length
  cap, and R is conserved;
* B = 0: both ends null.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from amr_field import AMRVectorField          # noqa: E402
from horizons import Horizons                 # noqa: E402
import tracer                                 # noqa: E402

FAILURES = []
HALF = 5.0                                    # tracing box half-width


def check(name, ok, detail=""):
    print("%s %s%s" % ("ok  " if ok else "FAIL", name,
                       ("  (%s)" % detail) if detail else ""), flush=True)
    if not ok:
        FAILURES.append(name)


def hierarchies(func):
    """The same field on one level and on two nested levels."""
    box_lo, box_hi = np.full(3, -HALF), np.full(3, HALF)
    one = AMRVectorField.from_function(
        func, [(0.1, [-5.4] * 3, [5.4] * 3, 2)], box_lo, box_hi)
    two = AMRVectorField.from_function(
        func, [(0.2, [-5.6] * 3, [5.6] * 3, 3),
               (0.1, [-2.0] * 3, [2.0] * 3, 2)], box_lo, box_hi)
    return {"1 level": one, "2 levels": two}


def unit_sphere():
    return Horizons([[0.0, 0.0, 0.0]], [[0.0, 0.0, 1.0]], [1.0], [1.0], ["BH"])


def seeds_in_box(n, rng, r_min=1.2):
    pts = rng.uniform(-HALF + 0.05, HALF - 0.05, size=(4 * n, 3))
    pts = pts[np.linalg.norm(pts, axis=1) > r_min]
    return pts[:n]


def names(result):
    legend = result.legend
    return np.array(["-".join(sorted((legend[a], legend[b])))
                     for a, b in result.ends])


def test_interpolation(rng):
    def linear(x, y, z):
        return 1 + 2 * x - y + 0.5 * z, 3 * y - z, x + y + z + 4

    for label, field in hierarchies(linear).items():
        pts = rng.uniform(-HALF, HALF, size=(20000, 3))
        val, level = tracer.sample(field, pts)
        exact = np.stack(linear(pts[:, 0], pts[:, 1], pts[:, 2]), axis=1)
        err = np.abs(val - exact).max()
        check("trilinear exact for a linear field, %s" % label, err < 1e-11,
              "max error %.1e" % err)
        if label == "2 levels":
            # The fine level owns |x_i| <= 2 + half its cell.
            fine = np.all(np.abs(pts) <= 2.05, axis=1)
            want = np.where(fine, 1, 0)
            edge = np.any(np.abs(np.abs(pts) - 2.05) < 1e-9, axis=1)
            ok = np.array_equal(level[~edge], want[~edge])
            check("finest level supplies each point", ok,
                  "%d fine, %d coarse" % (fine.sum(), (~fine).sum()))


def test_uniform(rng):
    def uniform(x, y, z):
        return 0 * x, 0 * x, 1 + 0 * x

    seeds = seeds_in_box(4000, rng)
    R = np.hypot(seeds[:, 0], seeds[:, 1])
    keep = np.abs(R - 1.0) > 0.02
    seeds, R = seeds[keep], R[keep]
    want = np.where(R < 1.0, "BH-exit", "exit-exit")
    for label, field in hierarchies(uniform).items():
        res = tracer.trace(field, unit_sphere(), seeds, step=0.5,
                           max_length=100.0)
        got = names(res)
        check("uniform field: classes, %s" % label, np.array_equal(got, want),
              "%d of %d differ" % ((got != want).sum(), len(want)))
        # Along +B (end 0) a seed above the sphere leaves; against B it lands.
        hit = R < 1.0
        up = hit & (seeds[:, 2] > 0)
        ok = (np.all(res.ends[up, 0] == 1) and np.all(res.ends[up, 1] == 0))
        check("uniform field: direction of each end, %s" % label, ok)
        r_end = np.linalg.norm(res.endpoints[hit][res.ends[hit] == 0], axis=-1)
        check("uniform field: lines end on the sphere, %s" % label,
              np.all((r_end <= 1.0) & (r_end > 0.94)),
              "r at the end %.4f .. %.4f" % (r_end.min(), r_end.max()))


def dipole(x, y, z):
    r2 = x * x + y * y + z * z
    r5 = np.where(r2 > 0.25, r2 ** 2.5, 1.0)
    on = r2 > 0.25
    return (np.where(on, 3 * x * z / r5, 0.0), np.where(on, 3 * y * z / r5, 0.0),
            np.where(on, (3 * z * z - r2) / r5, 0.0))


def test_dipole(rng):
    seeds = seeds_in_box(4000, rng, r_min=1.3)
    R2 = seeds[:, 0] ** 2 + seeds[:, 1] ** 2
    keep = R2 > 0.05
    seeds, R2 = seeds[keep], R2[keep]
    L = np.linalg.norm(seeds, axis=1) ** 3 / R2
    phi = np.arctan2(seeds[:, 1], seeds[:, 0])
    reach = L * np.maximum(np.abs(np.cos(phi)), np.abs(np.sin(phi)))
    keep = np.abs(reach - HALF) > 0.1
    seeds, L, reach = seeds[keep], L[keep], reach[keep]
    # A line that leaves the box does so about the equator, and every seed in
    # the box lies on one of its two inner arcs: one end on the sphere.
    want = np.where(reach < HALF, "BH-BH", "BH-exit")
    for label, field in hierarchies(dipole).items():
        res = tracer.trace(field, unit_sphere(), seeds, step=0.5,
                           max_length=200.0)
        got = names(res)
        check("dipole: classes, %s" % label, np.array_equal(got, want),
              "%d of %d differ; %d closed" % ((got != want).sum(), len(want),
                                              (want == "BH-BH").sum()))
        closed = want == "BH-BH"
        ends = res.endpoints[closed].reshape(-1, 3)
        L_end = np.linalg.norm(ends, axis=1) ** 3 / (ends[:, 0] ** 2
                                                     + ends[:, 1] ** 2)
        drift = np.abs(L_end / np.repeat(L[closed], 2) - 1.0).max()
        check("dipole: L = r^3/R^2 conserved to the footpoints, %s" % label,
              drift < 1e-2, "max relative drift %.1e" % drift)


def test_convergence(rng):
    """The drift of L along dipole lines falls as dx^2 (trilinear field)."""
    seeds = seeds_in_box(1000, rng, r_min=1.3)
    R2 = seeds[:, 0] ** 2 + seeds[:, 1] ** 2
    seeds = seeds[R2 > 0.05]
    L = np.linalg.norm(seeds, axis=1) ** 3 / (seeds[:, 0] ** 2
                                              + seeds[:, 1] ** 2)
    drifts = []
    for dx in (0.2, 0.1, 0.05):
        field = AMRVectorField.from_function(
            dipole, [(dx, [-5.4] * 3, [5.4] * 3, 2)], np.full(3, -HALF),
            np.full(3, HALF))
        res = tracer.trace(field, unit_sphere(), seeds, step=0.5,
                           max_length=200.0)
        closed = np.all(res.ends == 0, axis=1)
        ends = res.endpoints[closed].reshape(-1, 3)
        L_end = np.linalg.norm(ends, axis=1) ** 3 / (ends[:, 0] ** 2
                                                     + ends[:, 1] ** 2)
        drifts.append(np.median(np.abs(L_end / np.repeat(L[closed], 2) - 1.0)))
    ratios = [drifts[i] / drifts[i + 1] for i in range(2)]
    check("dipole: drift of L second order in dx", min(ratios) > 3.0,
          "median drift %s at dx 0.2 / 0.1 / 0.05, ratios %s"
          % (", ".join("%.1e" % d for d in drifts),
             ", ".join("%.1f" % r for r in ratios)))


def test_monopole(rng):
    def split_monopole(x, y, z):
        r2 = np.maximum(x * x + y * y + z * z, 0.25)
        s = np.sign(z) / r2 ** 1.5
        return s * x, s * y, s * z

    # Radial lines keep |z| / r; with |z| / r > 0.25 they stay more than a
    # coarse cell (0.2) away from the current sheet at z = 0 down to r = 1.
    seeds = seeds_in_box(4000, rng)
    seeds = seeds[np.abs(seeds[:, 2]) / np.linalg.norm(seeds, axis=1) > 0.25]
    for label, field in hierarchies(split_monopole).items():
        res = tracer.trace(field, unit_sphere(), seeds, step=0.5,
                           max_length=100.0)
        got = names(res)
        check("split monopole: classes, %s" % label,
              np.all(got == "BH-exit"), "%d of %d differ"
              % ((got != "BH-exit").sum(), len(got)))
        # Outward along B above the equator, inward below it.
        above = seeds[:, 2] > 0
        ok = (np.all(res.ends[above, 0] == 1) and np.all(res.ends[~above, 0] == 0))
        check("split monopole: polarity of the ends, %s" % label, ok)


def test_toroidal(rng):
    def toroidal(x, y, z):
        R = np.maximum(np.hypot(x, y), 1e-3)
        return -y / R, x / R, 0 * x

    seeds = seeds_in_box(4000, rng)
    R = np.hypot(seeds[:, 0], seeds[:, 1])
    keep = (R > 1.2) & (R < 4.5) & (np.abs(seeds[:, 2]) < 4.0)
    seeds, R = seeds[keep], R[keep]
    for label, field in hierarchies(toroidal).items():
        res = tracer.trace(field, unit_sphere(), seeds, step=0.5,
                           max_length=60.0)
        got = names(res)
        check("toroidal circles: both halves capped, %s" % label,
              np.all(got == "cap-cap"), "%d of %d differ"
              % ((got != "cap-cap").sum(), len(got)))
        check("toroidal circles: half-line length = cap, %s" % label,
              np.allclose(res.lengths, 60.0, atol=0.11))
        R_end = np.hypot(res.endpoints[..., 0], res.endpoints[..., 1])
        drift = np.abs(R_end - R[:, None]).max()
        check("toroidal circles: R conserved over 60 M, %s" % label,
              drift < 1e-2, "max |dR| %.1e" % drift)


def test_null(rng):
    def zero(x, y, z):
        return 0 * x, 0 * x, 0 * x

    field = hierarchies(zero)["1 level"]
    res = tracer.trace(field, unit_sphere(), seeds_in_box(500, rng))
    check("zero field: both ends null", np.all(names(res) == "null-null"))


def test_horizons(rng):
    hz = Horizons.kerr_schild([[1.0, -2.0, 0.5]], [[0.0, 0.0, 0.45]], [0.5])
    a_eq, a_pol = hz.a_eq[0], hz.a_pol[0]
    check("Kerr-Schild r+ for m = 0.5, chi = 0.9", abs(a_pol - 0.71794) < 1e-4,
          "%.5f" % a_pol)
    check("equatorial semi-axis sqrt(2 m r+)",
          abs(a_eq - np.sqrt(2 * 0.5 * a_pol)) < 1e-12, "%.5f" % a_eq)
    c = hz.centres[0]
    probe = c + np.array([[0, 0, 0.99 * a_pol], [0, 0, 1.01 * a_pol],
                          [0.99 * a_eq, 0, 0], [0, 1.01 * a_eq, 0]])
    check("inside / outside on the axis and the equator",
          np.array_equal(hz.inside(probe), [0, -1, 0, -1]))
    tilted = Horizons.kerr_schild([[0.0, 0.0, 0.0]], [[0.3, 0.0, 0.3]], [0.5])
    pts = rng.uniform(-1.0, 1.0, size=(20000, 3))
    kern = np.array([tracer._in_horizon(p[0], p[1], p[2], *tilted.kernel_args(),
                                        1.0)
                     for p in pts])
    check("kernel horizon test = Horizons.inside (tilted spin)",
          np.array_equal(kern, tilted.inside(pts)),
          "%d inside" % (kern >= 0).sum())


def test_determinism(rng):
    field = hierarchies(dipole)["2 levels"]
    seeds = seeds_in_box(2000, rng, r_min=1.3)
    a = tracer.trace(field, unit_sphere(), seeds, shuffle_seed=None)
    b = tracer.trace(field, unit_sphere(), seeds, shuffle_seed=7, batches=3)
    same = (np.array_equal(a.ends, b.ends) and np.array_equal(a.lengths, b.lengths)
            and np.array_equal(a.endpoints, b.endpoints))
    check("result independent of seed order and batching", same)
    cls = a.classes()
    legend = a.class_names()
    via = np.array([legend[k] for k in cls])
    check("class index <-> names", np.array_equal(
        via, [("%s-%s" % tuple(sorted(
            (a.legend[e0], a.legend[e1]), key=a.legend.index)))
            for e0, e1 in a.ends]))


def test_extra_scales(rng):
    """Ends with enlarged horizons from one trace = tracing with them."""
    seeds = seeds_in_box(3000, rng, r_min=1.3)
    for label, func in (("dipole", dipole), ("uniform", lambda x, y, z: (
            0 * x, 0 * x, 1 + 0 * x))):
        field = hierarchies(func)["2 levels"]
        one = tracer.trace(field, unit_sphere(), seeds, extra_scales=(1.1, 1.25))
        same = True
        for k, s in enumerate((1.1, 1.25)):
            big = Horizons([[0.0, 0.0, 0.0]], [[0.0, 0.0, 1.0]], [s], [s], ["BH"])
            direct = tracer.trace(field, big, seeds)
            same &= np.array_equal(one.xends[:, :, k], direct.ends)
        moved = np.mean(one.classes(1) != one.classes())
        check("enlarged-horizon ends = tracing with them, %s" % label, same,
              "%.1f%% of the seeds change class at 1.25x" % (100 * moved))


def main():
    rng = np.random.default_rng(2026)
    test_interpolation(rng)
    test_uniform(rng)
    test_dipole(rng)
    test_convergence(rng)
    test_monopole(rng)
    test_toroidal(rng)
    test_null(rng)
    test_horizons(rng)
    test_determinism(rng)
    test_extra_scales(rng)
    if FAILURES:
        print("FAILED: %d check(s): %s" % (len(FAILURES), "; ".join(FAILURES)))
        sys.exit(1)
    print("ALL OK")


if __name__ == "__main__":
    main()
