#!/usr/bin/env python3
"""Unit tests for sphere_geom.py (numpy only, a few seconds).

    python test_sphere_geom.py        -> ALL OK

The culling must be conservative: every grid point inside a box has to lie in
one of the windows :meth:`SphereSpec.windows` returns for that box. Checked by
brute force over random spheres, frames and boxes - thin slabs, boxes holding
the centre, boxes across phi = 0 and over the poles included - together with
the grid, frame and area weights.
"""

import sys

import numpy as np

from sphere_geom import (SphereSpec, area_weights, phi_grid, sphere_frame,
                         theta_grid, _phi_windows)

FAILED = []


def check(name, ok, detail=""):
    print("%-58s %s %s" % (name, "ok" if ok else "FAIL", detail))
    if not ok:
        FAILED.append(name)


def random_unit(rng):
    v = rng.normal(size=3)
    return v / np.linalg.norm(v)


def grid_and_frame():
    theta, phi = theta_grid(7), phi_grid(14)
    check("theta cell-centred, off the poles",
          np.allclose(theta, (np.arange(7) + 0.5) * np.pi / 7)
          and theta[0] > 0 and theta[-1] < np.pi)
    check("phi periodic, no duplicate endpoint",
          np.isclose(phi[0], 0.0) and np.isclose(phi[-1] + 2 * np.pi / 14,
                                                 2 * np.pi))

    w = area_weights(31, 62, radius=2.5)
    check("area weights sum to 4 pi r^2",
          np.isclose(w.sum(), 4 * np.pi * 2.5 ** 2, rtol=1e-13, atol=0),
          "(%.3e)" % abs(w.sum() / (4 * np.pi * 2.5 ** 2) - 1))
    # z^2 over the unit sphere is 4 pi / 3; second order in theta.
    th = theta_grid(200)
    w = area_weights(200, 8)
    integral = float((w * np.cos(th)[:, None] ** 2).sum())
    check("area weights integrate cos^2(theta) to 4 pi / 3",
          abs(integral / (4 * np.pi / 3) - 1) < 1e-4,
          "(rel %.1e)" % abs(integral / (4 * np.pi / 3) - 1))

    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(50):
        e1, e2, e3 = sphere_frame(random_unit(rng), random_unit(rng))
        m = np.vstack([e1, e2, e3])
        worst = max(worst, np.abs(m.dot(m.T) - np.eye(3)).max(),
                    abs(np.linalg.det(m) - 1.0))
    check("sphere_frame orthonormal and right-handed", worst < 1e-14,
          "(%.1e)" % worst)
    e1, e2, e3 = sphere_frame("z", "x")
    check("pole z, phi_zero x gives the standard frame",
          np.allclose(np.vstack([e1, e2, e3]), np.eye(3)))

    spec = SphereSpec([1.0, -2.0, 0.5], [0.7, 3.0], "x", "z")
    x, y, z = spec.points(theta_grid(9), phi_grid(18))
    r = np.sqrt((x - 1.0) ** 2 + (y + 2.0) ** 2 + (z - 0.5) ** 2)
    check("points lie on their spheres",
          x.shape == (2, 9, 18) and np.allclose(r[0], 0.7) and np.allclose(r[1], 3.0))
    # theta from +z, phi from +x towards +y
    th, ph = theta_grid(9)[2], phi_grid(18)[5]
    want = np.array([1.0, -2.0, 0.5]) + 3.0 * np.array(
        [np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
    check("theta from the pole, phi from phi_zero towards e2",
          np.allclose([x[1, 2, 5], y[1, 2, 5], z[1, 2, 5]], want))


def phi_wrapping():
    n = 16
    h = 2 * np.pi / n
    cases = [(-0.3, 0.3), (2 * np.pi - 0.2, 2 * np.pi + 0.4), (1.0, 1.5),
             (-np.pi, np.pi), (5 * h, 5 * h), (-4 * np.pi + 0.1, -4 * np.pi + 0.2)]
    ok = True
    for lo, hi in cases:
        idx = np.concatenate([np.arange(n)[w] for w in _phi_windows(n, lo, hi)])
        k = np.arange(-3 * n, 4 * n)
        need = np.unique(k[(k * h >= lo - 1e-15) & (k * h <= hi + 1e-15)] % n)
        ok &= set(need) <= set(idx) and len(idx) == len(set(idx))
    check("phi windows wrap around phi = 0, no duplicates", ok)


def brute_force(seed, n_spheres, n_boxes):
    """Every grid point inside a box is in some window of that box."""
    rng = np.random.default_rng(seed)
    theta, phi = theta_grid(41), phi_grid(82)
    missed = checked = wasted = needed = empty_ok = 0
    for _ in range(n_spheres):
        centre = rng.uniform(-4, 4, 3)
        radii = np.sort(rng.uniform(0.3, 6.0, 3))
        spec = SphereSpec(centre, radii, random_unit(rng), random_unit(rng))
        xs, ys, zs = spec.points(theta, phi)
        for _ in range(n_boxes):
            kind = rng.integers(4)
            if kind == 0:      # anywhere, any shape
                x0 = rng.uniform(-10, 10, 3)
                size = rng.uniform(0.05, 8, 3)
            elif kind == 1:    # a thin slab
                x0 = rng.uniform(-8, 8, 3)
                size = rng.uniform(4, 12, 3)
                size[rng.integers(3)] = rng.uniform(0.01, 0.2)
            elif kind == 2:    # holds the centre
                size = rng.uniform(0.5, 15, 3)
                x0 = centre - rng.uniform(0, 1, 3) * size
            else:              # small, on a sphere near a pole or phi = 0
                r = radii[rng.integers(3)]
                where = [spec.e3, -spec.e3, spec.e1][rng.integers(3)]
                size = rng.uniform(0.05, 1.0, 3)
                x0 = centre + r * where - rng.uniform(0, 1, 3) * size
            x1 = x0 + size
            inside = ((xs >= x0[0]) & (xs <= x1[0]) & (ys >= x0[1]) &
                      (ys <= x1[1]) & (zs >= x0[2]) & (zs <= x1[2]))
            covered = np.zeros_like(inside)
            for w in spec.windows(x0, x1, theta, phi):
                covered[w] = True
            checked += 1
            missed += int((inside & ~covered).sum())
            wasted += int(covered.sum())
            needed += int(inside.sum())
            empty_ok += int(not inside.any() and not covered.any())
    return checked, missed, wasted, needed, empty_ok


def main():
    grid_and_frame()
    phi_wrapping()
    checked, missed, wasted, needed, empty_ok = brute_force(7, 60, 120)
    check("windows never miss a point inside the box", missed == 0,
          "(%d boxes, %d points inside, %d missed)" % (checked, needed, missed))
    print("    candidates tested per point inside: %.2f; boxes no window "
          "touched: %d" % (wasted / max(needed, 1), empty_ok))
    print("ALL OK" if not FAILED else "FAILED: %s" % ", ".join(FAILED))
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
