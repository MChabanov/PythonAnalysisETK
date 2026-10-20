#!/usr/bin/env python3
"""The grid points of a spherical shell in raw AMR data, their volumes and integrands.

Volume integrals are sums over grid points, each standing for its dual cell -
the cube of side dx centred on it (Carpet is vertex-centred) - over which the
data are taken as constant. Where refinement levels overlap, the finer one
owns the region (``Resample/amr_blocks.py``), so a point of level L counts with
the part of its dual cell that level L+1 does not own,

    w = dx^3 - sum over components c of level L+1 of |dual cell  &  owned box of c| ,

restricted to the shell ``r_in <= |x - centre| < r_out``. The owned boxes of
one level are disjoint, and with proper nesting (Carpet's
``ensure_proper_nesting``) level L+2 lies inside level L+1, so this counts every
point of space exactly once: the weights of all levels add up to the volume of
the coarsest owned region, to round-off; a coarse cell straddling the edge of a
finer level is split exactly, not assigned to one level. A cell entirely
inside the shell takes w as it is. For the cells a bounding sphere cuts, w is
the cell volume times the fraction of ``subsamples``^3 midpoint sub-points
that lie in the shell and outside the finer level's owned boxes. Those edges
fall on quarter-cells of the coarser level (2:1, vertex-centred, snapped to
the coarse grid), so a multiple of 4 (default 8) resolves them exactly.
(Counting the cut cells whole or not at all by their centre leaves an error
of first order in dx that does not average out where a sphere runs parallel
to grid planes, as at the poles of a shell about a grid axis or along a
refinement boundary close to a sphere.)

The integrands are those of ``~/in_plane_flux_tube_diagnostics.md`` for a
spherical shell about a hole, with the densitized field
``calB^i = sqrt(gamma) B^i`` projected on the flat unit vectors of the frame
(centre c, pole p; d = x - c, r = |d|, zeta = d.p, R = sqrt(r^2 - zeta^2)):

    Phi_phi = 1/(2 pi)      sum w |calB . phi_hat| / R     phi_hat = p x d / R
    Phi_r   = 1/(r_out-r_in) sum w |calB . r_hat|          r_hat = d / r
    Phi_z   = 1/(2 h)        sum w |calB . p|              h = r_out

- the mean unsigned flux through the meridional half-planes, the spheres and
the planes zeta = const that cross the shell. They are exact for coordinate
surfaces: calB^i is divergence-free with the coordinate divergence, so its
flux through a coordinate surface needs no metric factor besides sqrt(gamma).
``|calB . p|`` binned in zeta (``profile``) is the flux through each plane,
whose mean over all bins is Phi_z.

Near the pole axis 1/R varies within a cell; cells closer than ``axis_cells``
spacings to it use the average of ``|calB . phi_hat| / R`` over the same
sub-points (those inside the shell), with calB held at the grid value - the
exact cell integral of the piecewise-constant data, to the sub-sampling. That
keeps a point that happens to lie almost on the axis from contributing an
arbitrarily large value.
"""

import numpy as np


class Selection(object):
    """The shell points of one component: an index slab and the points in it."""

    __slots__ = ("comp", "i0", "i1", "mask", "weight")

    def __init__(self, comp, i0, i1, mask, weight):
        self.comp = comp
        self.i0 = i0
        self.i1 = i1
        self.mask = mask          # [k, j, i] bool over the slab
        self.weight = weight      # (n,) volumes of the selected points


def _dual_overlap(x, dx, lo, hi):
    """Length of [x - dx/2, x + dx/2] inside [lo, hi], per point of ``x``."""
    return np.clip(np.minimum(x + 0.5 * dx, hi) - np.maximum(x - 0.5 * dx, lo),
                   0.0, None)


def sub_offsets(m):
    """Midpoint offsets of an m^3 sub-grid of the unit cell, ``(m^3, 3)``."""
    off = (np.arange(int(m)) + 0.5) / int(m) - 0.5
    ox, oy, oz = np.meshgrid(off, off, off, indexing="ij")
    return np.stack([ox.ravel(), oy.ravel(), oz.ravel()], axis=1)


def inside_fraction(d, dx, r_in, r_out, m, centre=None, boxes=(),
                    chunk=20000):
    """Fraction of each cell inside the shell and outside every box.

    ``d`` ``(n, 3)`` are the cell centres relative to the shell centre; the
    ``boxes`` ``[(lo, hi)]`` are in absolute coordinates (``centre`` needed).
    """
    off = sub_offsets(m) * np.asarray(dx, dtype=float)
    frac = np.empty(len(d))
    for a in range(0, len(d), chunk):
        ds = d[a:a + chunk, None, :] + off[None, :, :]
        r2 = np.einsum("ijk,ijk->ij", ds, ds)
        keep = (r2 >= r_in ** 2) & (r2 < r_out ** 2)
        if boxes:
            p = ds + centre
            for lo, hi in boxes:
                keep &= ~np.all((p >= lo) & (p <= hi), axis=2)
        frac[a:a + chunk] = np.mean(keep, axis=1)
    return frac


def select_shell(components, centre, r_in, r_out, subsamples=8):
    """:class:`Selection` per component holding shell points of nonzero weight.

    ``components`` are ``amr_blocks.Component`` objects (all levels).
    """
    centre = np.asarray(centre, dtype=float)
    by_level = {}
    for c in components:
        by_level.setdefault(c.level, []).append(c)
    out = []
    for comp in components:
        dx = comp.dx
        # A cell can meet a sphere of radius rb only if its centre is within
        # half a diagonal of it.
        delta = 0.5 * float(np.linalg.norm(dx))
        rng = comp.index_range(centre - r_out - delta, centre + r_out + delta,
                               owned=True)
        if rng is None:
            continue
        i0, i1 = rng
        x, y, z = comp.coords(i0, i1)
        r = np.sqrt((z - centre[2])[:, None, None] ** 2
                    + (y - centre[1])[None, :, None] ** 2
                    + (x - centre[0])[None, None, :] ** 2)
        whole = (r >= r_in + delta) & (r < r_out - delta)
        edge = (r >= r_in - delta) & (r < r_out + delta) & ~whole
        mask = whole | edge
        if not mask.any():
            continue
        cell = float(np.prod(dx))
        covered = np.zeros(mask.shape)
        boxes = []
        for fine in by_level.get(comp.level + 1, []):
            flo, fhi = fine.owned_box()
            # Only the points whose dual cells can meet the fine box.
            sub = comp.index_range(flo - 0.5 * dx, fhi + 0.5 * dx, owned=True)
            if sub is None:
                continue
            s0 = np.maximum(sub[0], i0)
            s1 = np.minimum(sub[1], i1)
            if np.any(s1 < s0):
                continue
            boxes.append((flo, fhi))
            a = s0 - i0
            b = s1 - i0 + 1
            ox = _dual_overlap(x[a[0]:b[0]], dx[0], flo[0], fhi[0])
            oy = _dual_overlap(y[a[1]:b[1]], dx[1], flo[1], fhi[1])
            oz = _dual_overlap(z[a[2]:b[2]], dx[2], flo[2], fhi[2])
            covered[a[2]:b[2], a[1]:b[1], a[0]:b[0]] += (
                oz[:, None, None] * oy[None, :, None] * ox[None, None, :])
        if np.any(covered > cell * (1.0 + 1e-9)):
            raise ValueError(
                "level %d component %d: finer components cover a cell more "
                "than once - their owned boxes overlap" % (comp.level,
                                                            comp.comp))
        weight = cell - covered
        k, j, i = np.nonzero(edge & (weight > 1e-12 * cell))
        if k.size:
            d = np.stack([x[i] - centre[0], y[j] - centre[1],
                          z[k] - centre[2]], axis=1)
            frac = inside_fraction(d, dx, r_in, r_out, subsamples)
            wk = weight[k, j, i] * frac            # exact where uncovered
            part = covered[k, j, i] > 1e-12 * cell
            if part.any():
                wk[part] = cell * inside_fraction(d[part], dx, r_in, r_out,
                                                  subsamples, centre, boxes)
            weight[k, j, i] = wk
        mask &= weight > 1e-12 * cell
        if not mask.any():
            continue
        out.append(Selection(comp, i0, i1, mask, weight[mask]))
    return out


def selection_points(selections):
    """Positions ``(n, 3)``, weights, level and spacing of all selected points."""
    pts, w, lev, dx = [], [], [], []
    for s in selections:
        x, y, z = s.comp.coords(s.i0, s.i1)
        k, j, i = np.nonzero(s.mask)
        pts.append(np.stack([x[i], y[j], z[k]], axis=1))
        w.append(s.weight)
        lev.append(np.full(len(k), s.comp.level, dtype=np.int16))
        dx.append(np.full(len(k), float(s.comp.dx.min())))
    if not pts:
        return (np.zeros((0, 3)), np.zeros(0), np.zeros(0, dtype=np.int16),
                np.zeros(0))
    return (np.concatenate(pts), np.concatenate(w), np.concatenate(lev),
            np.concatenate(dx))


def read_selected(read, selections, names):
    """Values of each variable in ``names`` at the selected points.

    ``read(var, comp, i0, i1)`` returns the ``[k, j, i]`` slab
    (``Snapshot.read``). Returns ``{name: (n,) array}`` in the order of
    :func:`selection_points`.
    """
    out = {name: [] for name in names}
    for s in selections:
        for name in names:
            out[name].append(read(name, s.comp, s.i0, s.i1)[s.mask])
    return {name: (np.concatenate(v) if v else np.zeros(0))
            for name, v in out.items()}


def sqrt_gamma(gxx, gxy, gxz, gyy, gyz, gzz):
    det = (gxx * (gyy * gzz - gyz * gyz) - gxy * (gxy * gzz - gyz * gxz)
           + gxz * (gxy * gyz - gyy * gxz))
    return np.sqrt(det)


def integrands(points, calB, centre, pole, dx, r_in, r_out, axis_cells=4.0,
               subsamples=8):
    """Per point: ``|calB.phi_hat|/R``, ``|calB.r_hat|``, ``|calB.p|`` and zeta.

    ``calB`` is ``(n, 3)``, ``dx`` the spacing of each point (for the
    near-axis cells, see the module doc). Also returns how many cells got the
    near-axis treatment.
    """
    pole = np.asarray(pole, dtype=float)
    pole = pole / np.linalg.norm(pole)
    d = points - np.asarray(centre, dtype=float)
    r = np.linalg.norm(d, axis=1)
    zeta = d @ pole
    R2 = np.maximum(r * r - zeta * zeta, 0.0)
    pxd = np.cross(pole, d)                     # |p x d| = R
    with np.errstate(divide="ignore", invalid="ignore"):
        f_phi = np.abs(np.einsum("ij,ij->i", calB, pxd)) / R2
    with np.errstate(divide="ignore", invalid="ignore"):
        # At the centre itself the cell mean of |calB . r_hat| is |calB|/2.
        f_r = np.where(r > 0, np.abs(np.einsum("ij,ij->i", calB, d)) / r,
                       0.5 * np.linalg.norm(calB, axis=1))
    f_z = np.abs(calB @ pole)

    near = np.flatnonzero(R2 < (axis_cells * dx) ** 2)
    offsets = sub_offsets(subsamples)
    for q in near:
        ds = d[q] + offsets * dx[q]
        rs2 = np.einsum("ij,ij->i", ds, ds)
        inside = (rs2 >= r_in ** 2) & (rs2 < r_out ** 2)
        if not inside.any():
            continue
        ds = ds[inside]
        zs = ds @ pole
        Rs2 = np.maximum(rs2[inside] - zs * zs, 1e-300)
        f_phi[q] = np.mean(np.abs(np.cross(pole, ds) @ calB[q]) / Rs2)
    return f_phi, f_r, f_z, zeta, near.size


def class_sums(cls, n_class, weight, f_phi, f_r, f_z, zeta, r_in, r_out,
               n_bin):
    """The normalised integrals per class, and the Phi_z profile in zeta.

    ``cls`` is each point's class index. Returns ``{Phi_phi, Phi_r, Phi_z,
    volume: (n_class,), profile: (n_class, n_bin), z_edges: (n_bin + 1,)}``;
    the profile is the flux through the planes zeta = const, bin by bin, so
    its mean over the bins (which tile -r_out .. r_out) is Phi_z.
    """
    h = r_out
    out = {
        "Phi_phi": np.bincount(cls, weight * f_phi, n_class) / (2.0 * np.pi),
        "Phi_r": np.bincount(cls, weight * f_r, n_class) / (r_out - r_in),
        "Phi_z": np.bincount(cls, weight * f_z, n_class) / (2.0 * h),
        "volume": np.bincount(cls, weight, n_class),
    }
    edges = np.linspace(-r_out, r_out, n_bin + 1)
    b = np.clip(np.searchsorted(edges, zeta, side="right") - 1, 0, n_bin - 1)
    out["profile"] = (np.bincount(cls * n_bin + b, weight * f_z,
                                  n_class * n_bin).reshape(n_class, n_bin)
                      / np.diff(edges)[None, :])
    out["z_edges"] = edges
    return out
