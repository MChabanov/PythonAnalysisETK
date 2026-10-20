#!/usr/bin/env python3
"""Trace magnetic field lines through AMR data and classify them by their ends.

A field line through a point x is the curve dx/dl = s B(x)/|B(x)|, with l the
coordinate arc length and s = +1 (along B) or -1 (against it). Its shape does
not depend on the field's normalisation, so the Eulerian B^i is all that is
needed - no metric. From each seed the line is followed both ways until it

* enters a horizon spheroid (``horizons.py``): end ``BH1``, ``BH2``, ...;
* leaves the tracing box, or the data: end ``exit``;
* reaches the length cap ``max_length``: end ``cap`` (undetermined);
* meets a point where B vanishes or is not finite: end ``null``.

The pair of ends classifies the seed, e.g. ``BH1-exit`` for a point on an
open field line threading BH1's horizon, ``BH1-BH1`` for a closed loop on it,
``exit-exit`` for field that does not reach a horizon inside the box.

Integration: classical RK4 with a fixed step of ``step`` times the grid
spacing of the finest level at the start of the step, so the line is followed
at the resolution of the data wherever it goes. The field at each RK stage is
trilinear in the finest block that claims the point (``amr_field.py``) - the
``interp_order: 1`` of the resampling pipeline. A horizon is tested at the end
of every step; at a step of half a cell that cannot skip a horizon of
semi-axes ~0.7 M on a 1/32 M grid.

Seeds are independent, so the kernel runs over them with numba threads
(``numba.set_num_threads``), in a shuffled order: the cost per seed ranges from
a few steps to the length cap, and neighbouring seeds tend to cost alike.
"""

import math
import time

import numpy as np
from numba import njit, prange


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------

_UNSET = 255        # an extra-scale end not decided yet

@njit(cache=False)
def _locate(px, py, pz, blk_lo, blk_hi, lk_lo, lk_inv, lk_n, lk_start,
            lk_items):
    """The finest block claiming the point, or -1."""
    fx = (px - lk_lo[0]) * lk_inv
    fy = (py - lk_lo[1]) * lk_inv
    fz = (pz - lk_lo[2]) * lk_inv
    if not (fx >= 0.0 and fy >= 0.0 and fz >= 0.0):
        return -1
    cx = int(fx)
    cy = int(fy)
    cz = int(fz)
    if cx >= lk_n[0] or cy >= lk_n[1] or cz >= lk_n[2]:
        return -1
    cell = (cx * lk_n[1] + cy) * lk_n[2] + cz
    for q in range(lk_start[cell], lk_start[cell + 1]):
        b = lk_items[q]
        if (blk_lo[b, 0] <= px <= blk_hi[b, 0]
                and blk_lo[b, 1] <= py <= blk_hi[b, 1]
                and blk_lo[b, 2] <= pz <= blk_hi[b, 2]):
            return b
    return -1


@njit(cache=False)
def _interp(b, px, py, pz, data, blk_off, blk_x0, blk_inv, blk_n):
    """Trilinear interpolation of the 3 components in block ``b``."""
    nx = blk_n[b, 0]
    ny = blk_n[b, 1]
    nz = blk_n[b, 2]
    fx = (px - blk_x0[b, 0]) * blk_inv[b, 0]
    fy = (py - blk_x0[b, 1]) * blk_inv[b, 1]
    fz = (pz - blk_x0[b, 2]) * blk_inv[b, 2]
    i = int(math.floor(fx))
    j = int(math.floor(fy))
    k = int(math.floor(fz))
    i = min(max(i, 0), nx - 2)
    j = min(max(j, 0), ny - 2)
    k = min(max(k, 0), nz - 2)
    tx = fx - i
    ty = fy - j
    tz = fz - k
    sy = nx
    sz = nx * ny
    p = blk_off[b] + k * sz + j * sy + i
    w000 = (1.0 - tx) * (1.0 - ty) * (1.0 - tz)
    w100 = tx * (1.0 - ty) * (1.0 - tz)
    w010 = (1.0 - tx) * ty * (1.0 - tz)
    w110 = tx * ty * (1.0 - tz)
    w001 = (1.0 - tx) * (1.0 - ty) * tz
    w101 = tx * (1.0 - ty) * tz
    w011 = (1.0 - tx) * ty * tz
    w111 = tx * ty * tz
    out0 = 0.0
    out1 = 0.0
    out2 = 0.0
    for c in range(3):
        v = (w000 * data[p, c] + w100 * data[p + 1, c]
             + w010 * data[p + sy, c] + w110 * data[p + sy + 1, c]
             + w001 * data[p + sz, c] + w101 * data[p + sz + 1, c]
             + w011 * data[p + sz + sy, c] + w111 * data[p + sz + sy + 1, c])
        if c == 0:
            out0 = v
        elif c == 1:
            out1 = v
        else:
            out2 = v
    return out0, out1, out2


@njit(cache=False)
def _direction(px, py, pz, data, blk_off, blk_x0, blk_inv, blk_step, blk_n,
               blk_lo, blk_hi, lk_lo, lk_inv, lk_n, lk_start, lk_items):
    """``(status, ux, uy, uz, spacing)``: unit field direction at the point.

    status 0 = ok, 1 = no block claims the point, 2 = B zero or not finite.
    """
    b = _locate(px, py, pz, blk_lo, blk_hi, lk_lo, lk_inv, lk_n, lk_start,
                lk_items)
    if b < 0:
        return 1, 0.0, 0.0, 0.0, 0.0
    vx, vy, vz = _interp(b, px, py, pz, data, blk_off, blk_x0, blk_inv, blk_n)
    norm = math.sqrt(vx * vx + vy * vy + vz * vz)
    if not (norm > 0.0 and norm < math.inf):
        return 2, 0.0, 0.0, 0.0, 0.0
    return 0, vx / norm, vy / norm, vz / norm, blk_step[b]


@njit(cache=False)
def _in_horizon(px, py, pz, hc, ha, hie, hip, lim):
    """Index of the horizon spheroid containing the point, or -1.

    ``lim`` is the square of a scale factor on the semi-axes (1 for the
    spheroids as given).
    """
    for h in range(hc.shape[0]):
        dx = px - hc[h, 0]
        dy = py - hc[h, 1]
        dz = pz - hc[h, 2]
        zeta = dx * ha[h, 0] + dy * ha[h, 1] + dz * ha[h, 2]
        rho2 = dx * dx + dy * dy + dz * dz - zeta * zeta
        if rho2 * hie[h] + zeta * zeta * hip[h] < lim:
            return h
    return -1


@njit(cache=False)
def _mark_scaled(px, py, pz, hc, ha, hie, hip, xs2, xe):
    """Record the first enlarged horizon each extra scale's spheroids hold."""
    for k in range(xs2.shape[0]):
        if xe[k] == _UNSET:
            h = _in_horizon(px, py, pz, hc, ha, hie, hip, xs2[k])
            if h >= 0:
                xe[k] = h


@njit(cache=False)
def _finish_scaled(xe, end):
    """Extra scales whose spheroids the line never entered keep its end."""
    for k in range(xe.shape[0]):
        if xe[k] == _UNSET:
            xe[k] = end


@njit(cache=False)
def _outside(px, py, pz, box_lo, box_hi):
    return not (box_lo[0] <= px <= box_hi[0] and box_lo[1] <= py <= box_hi[1]
                and box_lo[2] <= pz <= box_hi[2])


@njit(cache=False)
def _trace_dir(x, y, z, sgn, h_fac, max_len, box_lo, box_hi, hc, ha, hie, hip,
               xs2, xe, data, blk_off, blk_x0, blk_inv, blk_step, blk_n, blk_lo,
               blk_hi, lk_lo, lk_inv, lk_n, lk_start, lk_items):
    """Follow one line one way: ``(end, length, x, y, z)`` at its end.

    ``xe`` receives the end with the horizons enlarged by each extra scale
    (``xs2``: squares of the scale ratios, all > 1).
    """
    nh = hc.shape[0]
    end_exit = nh
    end_cap = nh + 1
    end_null = nh + 2
    for k in range(xe.shape[0]):
        xe[k] = _UNSET
    _mark_scaled(x, y, z, hc, ha, hie, hip, xs2, xe)
    h0 = _in_horizon(x, y, z, hc, ha, hie, hip, 1.0)
    if h0 >= 0:
        _finish_scaled(xe, h0)
        return h0, 0.0, x, y, z
    if _outside(x, y, z, box_lo, box_hi):
        _finish_scaled(xe, end_exit)
        return end_exit, 0.0, x, y, z
    length = 0.0
    while True:
        st, k1x, k1y, k1z, dxl = _direction(
            x, y, z, data, blk_off, blk_x0, blk_inv, blk_step, blk_n, blk_lo,
            blk_hi, lk_lo, lk_inv, lk_n, lk_start, lk_items)
        if st != 0:
            end = end_exit if st == 1 else end_null
            _finish_scaled(xe, end)
            return end, length, x, y, z
        h = h_fac * dxl
        k1x *= sgn
        k1y *= sgn
        k1z *= sgn
        st, k2x, k2y, k2z, _ = _direction(
            x + 0.5 * h * k1x, y + 0.5 * h * k1y, z + 0.5 * h * k1z,
            data, blk_off, blk_x0, blk_inv, blk_step, blk_n, blk_lo, blk_hi,
            lk_lo, lk_inv, lk_n, lk_start, lk_items)
        if st != 0:
            end = end_exit if st == 1 else end_null
            _finish_scaled(xe, end)
            return end, length, x, y, z
        k2x *= sgn
        k2y *= sgn
        k2z *= sgn
        st, k3x, k3y, k3z, _ = _direction(
            x + 0.5 * h * k2x, y + 0.5 * h * k2y, z + 0.5 * h * k2z,
            data, blk_off, blk_x0, blk_inv, blk_step, blk_n, blk_lo, blk_hi,
            lk_lo, lk_inv, lk_n, lk_start, lk_items)
        if st != 0:
            end = end_exit if st == 1 else end_null
            _finish_scaled(xe, end)
            return end, length, x, y, z
        k3x *= sgn
        k3y *= sgn
        k3z *= sgn
        st, k4x, k4y, k4z, _ = _direction(
            x + h * k3x, y + h * k3y, z + h * k3z,
            data, blk_off, blk_x0, blk_inv, blk_step, blk_n, blk_lo, blk_hi,
            lk_lo, lk_inv, lk_n, lk_start, lk_items)
        if st != 0:
            end = end_exit if st == 1 else end_null
            _finish_scaled(xe, end)
            return end, length, x, y, z
        k4x *= sgn
        k4y *= sgn
        k4z *= sgn
        x += h * (k1x + 2.0 * k2x + 2.0 * k3x + k4x) / 6.0
        y += h * (k1y + 2.0 * k2y + 2.0 * k3y + k4y) / 6.0
        z += h * (k1z + 2.0 * k2z + 2.0 * k3z + k4z) / 6.0
        length += h
        _mark_scaled(x, y, z, hc, ha, hie, hip, xs2, xe)
        hh = _in_horizon(x, y, z, hc, ha, hie, hip, 1.0)
        if hh >= 0:
            _finish_scaled(xe, hh)
            return hh, length, x, y, z
        if _outside(x, y, z, box_lo, box_hi):
            _finish_scaled(xe, end_exit)
            return end_exit, length, x, y, z
        if length >= max_len:
            _finish_scaled(xe, end_cap)
            return end_cap, length, x, y, z


@njit(parallel=True, cache=False)
def _trace_seeds(seeds, order, h_fac, max_len, box_lo, box_hi, hc, ha, hie,
                 hip, xs2, data, blk_off, blk_x0, blk_inv, blk_step, blk_n,
                 blk_lo, blk_hi, lk_lo, lk_inv, lk_n, lk_start, lk_items, ends,
                 lengths, endpoints, xends):
    for q in prange(order.shape[0]):
        s = order[q]
        for d in range(2):
            sgn = 1.0 - 2.0 * d
            e, length, ex, ey, ez = _trace_dir(
                seeds[s, 0], seeds[s, 1], seeds[s, 2], sgn, h_fac, max_len,
                box_lo, box_hi, hc, ha, hie, hip, xs2, xends[s, d], data,
                blk_off, blk_x0, blk_inv, blk_step, blk_n, blk_lo, blk_hi,
                lk_lo, lk_inv, lk_n, lk_start, lk_items)
            ends[s, d] = e
            lengths[s, d] = length
            endpoints[s, d, 0] = ex
            endpoints[s, d, 1] = ey
            endpoints[s, d, 2] = ez


@njit(parallel=True, cache=False)
def _sample(points, data, blk_off, blk_x0, blk_inv, blk_step, blk_n, blk_lo,
            blk_hi, lk_lo, lk_inv, lk_n, lk_start, lk_items, out, block):
    for q in prange(points.shape[0]):
        b = _locate(points[q, 0], points[q, 1], points[q, 2], blk_lo, blk_hi,
                    lk_lo, lk_inv, lk_n, lk_start, lk_items)
        block[q] = b
        if b < 0:
            out[q, 0] = math.nan
            out[q, 1] = math.nan
            out[q, 2] = math.nan
        else:
            vx, vy, vz = _interp(b, points[q, 0], points[q, 1], points[q, 2],
                                 data, blk_off, blk_x0, blk_inv, blk_n)
            out[q, 0] = vx
            out[q, 1] = vy
            out[q, 2] = vz


# ---------------------------------------------------------------------------
# Python API
# ---------------------------------------------------------------------------

def sample(field, points):
    """The field at ``points`` ``(n, 3)`` and the level that supplied each.

    Points no block claims get NaN and level -1.
    """
    points = np.ascontiguousarray(points, dtype=np.float64).reshape(-1, 3)
    out = np.empty_like(points)
    block = np.empty(len(points), dtype=np.int64)
    _sample(points, *field.kernel_args(), out, block)
    level = np.where(block >= 0, field.blk_level[np.maximum(block, 0)], -1)
    return out, level


def end_names(horizons):
    """Names of the end codes: one per horizon, then exit, cap, null."""
    return list(horizons.names) + ["exit", "cap", "null"]


def class_names(ends_legend):
    """Names of the unordered end pairs, in class-index order."""
    n = len(ends_legend)
    return ["%s-%s" % (ends_legend[a], ends_legend[b])
            for a in range(n) for b in range(a, n)]


def classify(ends, n_end):
    """Class index of each seed from its ``(n, 2)`` end codes (order-free)."""
    a = np.minimum(ends[:, 0], ends[:, 1]).astype(np.int64)
    b = np.maximum(ends[:, 0], ends[:, 1]).astype(np.int64)
    return a * n_end - a * (a - 1) // 2 + (b - a)


class TraceResult(object):
    """Per seed: ``ends`` ``(n, 2)`` codes (along B, against B), the line
    ``lengths`` to each end and the ``endpoints`` ``(n, 2, 3)``; ``xends``
    ``(n, 2, n_extra)`` the ends had the horizons been larger by each of
    ``extra_scales``."""

    def __init__(self, ends, lengths, endpoints, legend, seconds, xends=None,
                 extra_scales=()):
        self.ends = ends
        self.lengths = lengths
        self.endpoints = endpoints
        self.legend = legend
        self.seconds = seconds
        self.xends = (np.zeros(ends.shape + (0,), dtype=np.uint8)
                      if xends is None else xends)
        self.extra_scales = tuple(float(x) for x in extra_scales)

    @property
    def n_end(self):
        return len(self.legend)

    def classes(self, extra=None):
        """Class per seed; with ``extra = k`` for the k-th enlarged horizon."""
        ends = self.ends if extra is None else self.xends[:, :, extra]
        return classify(ends, self.n_end)

    def class_names(self):
        return class_names(self.legend)


def trace(field, horizons, seeds, step=0.5, max_length=200.0, box=None,
          batches=1, shuffle_seed=12345, log=None, extra_scales=()):
    """Trace the line through every seed both ways; returns a :class:`TraceResult`.

    ``step`` is the RK4 step in units of the local grid spacing, ``max_length``
    the cap on each half-line's length, ``box`` ``(lo, hi)`` the tracing box
    (default: the field's lookup box). ``batches`` > 1 splits the (shuffled)
    seeds so progress can be logged through ``log``.

    ``extra_scales`` (each > 1) also gives, from the same lines, the ends the
    seeds would have with every horizon's semi-axes multiplied by that factor:
    a line is the same curve up to where it first enters an enlarged spheroid,
    which is then its end - or, if it never does, the end it has anyway. Exact,
    i.e. identical to tracing with the enlarged horizons.
    """
    seeds = np.ascontiguousarray(seeds, dtype=np.float64).reshape(-1, 3)
    n = len(seeds)
    extra = np.asarray(extra_scales, dtype=np.float64).ravel()
    if np.any(extra <= 1.0):
        raise ValueError("extra horizon scales must exceed 1 (got %s)" % extra)
    box_lo, box_hi = ((field.box_lo, field.box_hi) if box is None
                      else (np.asarray(box[0], dtype=float),
                            np.asarray(box[1], dtype=float)))
    order = np.arange(n, dtype=np.int64)
    if shuffle_seed is not None:
        np.random.default_rng(shuffle_seed).shuffle(order)
    ends = np.zeros((n, 2), dtype=np.uint8)
    lengths = np.zeros((n, 2), dtype=np.float64)
    endpoints = np.zeros((n, 2, 3), dtype=np.float64)
    xends = np.zeros((n, 2, extra.size), dtype=np.uint8)
    hargs = horizons.kernel_args()
    fargs = field.kernel_args()
    t0 = time.perf_counter()
    edges = np.linspace(0, n, max(1, int(batches)) + 1).astype(np.int64)
    for b in range(len(edges) - 1):
        part = order[edges[b]:edges[b + 1]]
        if part.size == 0:
            continue
        _trace_seeds(seeds, part, float(step), float(max_length),
                     np.ascontiguousarray(box_lo), np.ascontiguousarray(box_hi),
                     *hargs, extra ** 2, *fargs, ends, lengths, endpoints,
                     xends)
        if log is not None and len(edges) > 2:
            elapsed = time.perf_counter() - t0
            log("    traced %d / %d seeds, %.0f s" % (edges[b + 1], n, elapsed))
    return TraceResult(ends, lengths, endpoints, end_names(horizons),
                       time.perf_counter() - t0, xends, extra)
