#!/usr/bin/env python3
"""A vector field on AMR blocks, packed for the numba field-line tracer.

The tracer evaluates the field at arbitrary points along each line, always
from the finest refinement level that owns the point - the ownership rule of
the resampling backends (``Resample/amr_blocks.py``). This module holds the
field in the form the numba kernels of ``tracer.py`` read:

* one **block** per AMR component (or the part of one that meets the tracing
  box), with its origin, spacing, point counts and the box it **claims**: the
  component's owned box, clipped to the data actually loaded;
* all block data in one ``(n_points, 3)`` float64 array, each block's points in
  the HDF5 order ``[k, j, i]`` and the three vector components interleaved, so
  a trilinear stencil reads 8 x 3 neighbouring values;
* a uniform **lookup grid** over the tracing box: per cell, the blocks whose
  claimed box meets it, finest level first. Locating a point is one cell lookup
  and a few box tests; the first block that claims the point is the finest.

Blocks come from 3D Carpet output (:meth:`AMRVectorField.from_snapshot`) or,
for tests, from a function sampled on a synthetic hierarchy
(:meth:`AMRVectorField.from_function`).
"""

import numpy as np


class Block(object):
    """Geometry of one block; ``n`` is the point count in x, y, z."""

    __slots__ = ("level", "x0", "dx", "n", "claim_lo", "claim_hi", "source")

    def __init__(self, level, x0, dx, n, claim_lo, claim_hi, source=None):
        self.level = int(level)
        self.x0 = np.asarray(x0, dtype=float)
        self.dx = np.asarray(dx, dtype=float)
        self.n = np.asarray(n, dtype=np.int64)
        self.claim_lo = np.asarray(claim_lo, dtype=float)
        self.claim_hi = np.asarray(claim_hi, dtype=float)
        self.source = source

    @property
    def size(self):
        return int(np.prod(self.n))


class AMRVectorField(object):
    """A 3-vector field on blocks, with the lookup grid the tracer uses.

    ``blocks`` are :class:`Block` objects and ``data`` the ``(n_points, 3)``
    array holding them back to back in the order given (each ``[k, j, i]``).
    ``box_lo``/``box_hi`` bound the lookup grid of cell size ``lookup_cell``;
    points outside it are never located. ``names`` are the components' variable
    names, for the record.
    """

    def __init__(self, blocks, data, box_lo, box_hi, lookup_cell=0.5,
                 names=("x", "y", "z")):
        if data.ndim != 2 or data.shape[1] != 3:
            raise ValueError("data must be (n_points, 3)")
        if sum(b.size for b in blocks) != data.shape[0]:
            raise ValueError("data holds %d points, the blocks %d"
                             % (data.shape[0], sum(b.size for b in blocks)))
        if any(np.any(b.n < 2) for b in blocks):
            raise ValueError("every block needs at least 2 points per axis")
        self.names = tuple(names)
        self.box_lo = np.asarray(box_lo, dtype=float)
        self.box_hi = np.asarray(box_hi, dtype=float)

        offsets = np.zeros(len(blocks), dtype=np.int64)
        if blocks:
            offsets[1:] = np.cumsum([b.size for b in blocks])[:-1]
        # Finest first: the lookup takes the first block claiming a point.
        order = sorted(range(len(blocks)), key=lambda i: -blocks[i].level)
        self.blocks = [blocks[i] for i in order]
        self.data = np.ascontiguousarray(data, dtype=np.float64)
        self.blk_off = offsets[order]
        self.blk_level = np.array([b.level for b in self.blocks], dtype=np.int64)
        self.blk_x0 = _stack([b.x0 for b in self.blocks])
        self.blk_dx = _stack([b.dx for b in self.blocks])
        self.blk_inv = 1.0 / self.blk_dx if self.blocks else self.blk_dx
        self.blk_step = (self.blk_dx.min(axis=1) if self.blocks
                         else np.zeros(0))
        self.blk_n = np.array([b.n for b in self.blocks],
                              dtype=np.int64).reshape(-1, 3)
        self.blk_lo = _stack([b.claim_lo for b in self.blocks])
        self.blk_hi = _stack([b.claim_hi for b in self.blocks])
        self._build_lookup(float(lookup_cell))

    # -- the lookup grid ---------------------------------------------------

    def _build_lookup(self, cell):
        if cell <= 0.0:
            raise ValueError("lookup_cell must be positive")
        n = np.maximum(1, np.ceil((self.box_hi - self.box_lo) / cell)
                       .astype(np.int64))
        self.lk_lo = self.box_lo.copy()
        self.lk_inv = 1.0 / cell
        self.lk_n = n
        cells, items = [], []
        for b in range(len(self.blocks)):
            lo = np.floor((self.blk_lo[b] - self.lk_lo) * self.lk_inv)
            hi = np.floor((self.blk_hi[b] - self.lk_lo) * self.lk_inv)
            lo = np.maximum(lo, 0).astype(np.int64)
            hi = np.minimum(hi, n - 1).astype(np.int64)
            if np.any(hi < lo):
                continue
            ix, iy, iz = np.meshgrid(*[np.arange(lo[d], hi[d] + 1)
                                       for d in range(3)], indexing="ij")
            ids = ((ix * n[1] + iy) * n[2] + iz).ravel()
            cells.append(ids)
            items.append(np.full(ids.size, b, dtype=np.int32))
        n_cells = int(np.prod(n))
        if cells:
            cells = np.concatenate(cells)
            items = np.concatenate(items)
            # Stable: within a cell the blocks keep their finest-first order.
            perm = np.argsort(cells, kind="stable")
            cells, items = cells[perm], items[perm]
            counts = np.bincount(cells, minlength=n_cells)
        else:
            items = np.zeros(0, dtype=np.int32)
            counts = np.zeros(n_cells, dtype=np.int64)
        self.lk_start = np.zeros(n_cells + 1, dtype=np.int64)
        self.lk_start[1:] = np.cumsum(counts)
        self.lk_items = np.ascontiguousarray(items, dtype=np.int32)

    def kernel_args(self):
        """The arrays the numba kernels take, in their order."""
        return (self.data, self.blk_off, self.blk_x0, self.blk_inv,
                self.blk_step, self.blk_n, self.blk_lo, self.blk_hi,
                self.lk_lo, self.lk_inv, self.lk_n, self.lk_start,
                self.lk_items)

    def nbytes(self):
        return self.data.nbytes + self.lk_items.nbytes + self.lk_start.nbytes

    def describe(self):
        levels = sorted(set(int(lv) for lv in self.blk_level))
        rows = []
        for lv in levels:
            sel = self.blk_level == lv
            rows.append("level %d: %d block(s), %.3g Mpts, dx %g"
                        % (lv, int(sel.sum()),
                           sum(self.blocks[i].size for i in np.flatnonzero(sel))
                           / 1e6, float(self.blk_step[sel][0])))
        return rows

    # -- construction ------------------------------------------------------

    @classmethod
    def from_snapshot(cls, snap, names, box_lo, box_hi, levels=None,
                      margin=3, lookup_cell=0.5, log=None):
        """Load the components of ``names`` (three variables) meeting a box.

        ``snap`` is a ``Resample/amr_blocks.Snapshot``. Each component whose
        owned box meets ``[box_lo, box_hi]`` is read on the part of it within
        ``margin`` of its own cells of the box (whole points, ghosts included,
        so the trilinear stencil always has data); that part claims its owned
        box clipped to the points read. ``levels`` restricts the levels used.
        """
        box_lo = np.asarray(box_lo, dtype=float)
        box_hi = np.asarray(box_hi, dtype=float)
        plan = []
        for comp in snap.select(box_lo, box_hi, levels=levels, owned=True):
            rng = comp.index_range(box_lo - margin * comp.dx,
                                   box_hi + margin * comp.dx, owned=False)
            if rng is None:
                continue
            i0, i1 = rng
            n = i1 - i0 + 1
            if np.any(n < 2):
                continue
            x0 = comp.x0 + i0 * comp.dx
            own_lo, own_hi = comp.owned_box()
            claim_lo = np.maximum(own_lo, x0)
            claim_hi = np.minimum(own_hi, x0 + (n - 1) * comp.dx)
            if np.any(claim_hi < claim_lo):
                continue
            plan.append((comp, i0, i1, Block(comp.level, x0, comp.dx, n,
                                             claim_lo, claim_hi, source=comp)))
        total = sum(b.size for _, _, _, b in plan)
        if log is not None:
            log("  field: %d block(s), %.3g Mpts, %.2f GB"
                % (len(plan), total / 1e6, total * 24 / 1e9))
        data = np.empty((total, 3), dtype=np.float64)
        start = 0
        for comp, i0, i1, block in plan:
            view = data[start:start + block.size]
            for d, name in enumerate(names):
                view[:, d] = snap.read(name, comp, i0, i1).ravel()
            start += block.size
        return cls([b for _, _, _, b in plan], data, box_lo, box_hi,
                   lookup_cell=lookup_cell, names=names)

    @classmethod
    def from_function(cls, func, levels, box_lo, box_hi, nghost=3,
                      lookup_cell=0.5):
        """A synthetic hierarchy sampled from ``func(x, y, z) -> (fx, fy, fz)``.

        ``levels`` is a list of ``(dx, lo, hi, n_split)``: grid points
        ``lo + i dx`` up to ``hi`` (owned), split into ``n_split`` components
        along x, each padded by ``nghost`` ghost points. Mimics Carpet's
        layout: owned boxes of one level tile it, ghost zones overlap the
        neighbours. Returns the field; ``.components`` lists the geometry of
        each component as ``(level, x0, dx, shape, nghost)``.
        """
        blocks, chunks, comps = [], [], []
        for level, (dx, lo, hi, n_split) in enumerate(levels):
            lo = np.asarray(lo, dtype=float)
            hi = np.asarray(hi, dtype=float)
            npts = np.rint((hi - lo) / dx).astype(int) + 1
            edges = np.linspace(0, npts[0], int(n_split) + 1).astype(int)
            for s in range(int(n_split)):
                own0 = np.array([edges[s], 0, 0])
                own1 = np.array([edges[s + 1] - 1, npts[1] - 1, npts[2] - 1])
                i0 = own0 - nghost
                i1 = own1 + nghost
                x0 = lo + i0 * dx
                n = i1 - i0 + 1
                ng = np.full(3, nghost)
                comps.append((level, x0, np.full(3, dx), n, ng))
                ax = [x0[d] + dx * np.arange(n[d]) for d in range(3)]
                Z, Y, X = np.meshgrid(ax[2], ax[1], ax[0], indexing="ij")
                fx, fy, fz = func(X, Y, Z)
                chunks.append(np.stack([np.broadcast_to(f, X.shape).ravel()
                                        for f in (fx, fy, fz)], axis=1))
                own_lo = x0 + (nghost - 0.5) * dx
                own_hi = x0 + (n - 1 - nghost + 0.5) * dx
                blocks.append(Block(level, x0, np.full(3, dx), n, own_lo, own_hi))
        field = cls(blocks, np.concatenate(chunks), box_lo, box_hi,
                    lookup_cell=lookup_cell)
        field.components = comps
        return field


def _stack(rows):
    if not rows:
        return np.zeros((0, 3))
    return np.ascontiguousarray(np.array(rows, dtype=float).reshape(-1, 3))
