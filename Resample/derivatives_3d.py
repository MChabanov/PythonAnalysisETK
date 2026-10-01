#!/usr/bin/env python3
"""Spatial derivatives of gridded fields by centred finite differences.

Backend-agnostic (numpy only) and independent of any plane geometry: it
differentiates a uniformly spaced N-dimensional array - in practice one AMR
component, or a hyperslab of one, read from 3D output - and reports how many
cells the stencil consumed at each edge. The 3D resampling backend uses it to
evaluate derivatives on the *native* grid of every component before
interpolating them anywhere, so derivatives are never taken of interpolated
data. Nothing here knows about postcactus, planes or MPI, so it can be reused
for any other 3D analysis.

Derivatives
-----------
A derivative is named by the axes it is taken along, one letter per order:
``"x"`` is d/dx, ``"xy"`` is d^2/dxdy, ``"zz"`` is d^2/dz^2. Up to second order
in each axis is supported; a mixed derivative is the product of first-derivative
stencils along each axis it involves, and a repeated axis uses the proper
second-derivative stencil (not two first-derivative stencils in a row), so every
derivative of a given ``accuracy`` has the same stencil half-width along each
axis it involves: 1, 2, 3 cells for accuracy 2, 4, 6.

The derivative is with respect to the grid's own coordinates (the simulation's
x, y, z for 3D output), not any frame the result is later sampled in.

Stencil trimming
----------------
:func:`differentiate` returns only the points where the full stencil fits, i.e.
the input shrunk by the half-width on both sides of every axis it differentiates
along (:func:`trim`). The caller shifts the origin accordingly. Nothing is
extrapolated or filled with one-sided stencils, so every returned value has the
stated accuracy.

Config
------
:func:`parse_config` reads the ``derivatives`` block of a resampling config::

    derivatives:
      accuracy: 4          # 2, 4 or 6 (default 4)
      fields:
        Bx:  [x, y, z]     # -> dBx_dx, dBx_dy, dBx_dz
        gxx: [x, y, z, xy] # ...      plus d2gxx_dxdy

and turns every entry into a :class:`Derivative`, whose :attr:`~Derivative.name`
is the output variable name.
"""

import numpy as np

AXES = "xyz"

# Centred stencils: {derivative order: {accuracy: coefficients}}, offsets
# -w..+w. Standard Fornberg weights; each row sums to 0.
_STENCILS = {
    1: {
        2: [-1.0 / 2, 0.0, 1.0 / 2],
        4: [1.0 / 12, -2.0 / 3, 0.0, 2.0 / 3, -1.0 / 12],
        6: [-1.0 / 60, 3.0 / 20, -3.0 / 4, 0.0, 3.0 / 4, -3.0 / 20, 1.0 / 60],
    },
    2: {
        2: [1.0, -2.0, 1.0],
        4: [-1.0 / 12, 4.0 / 3, -5.0 / 2, 4.0 / 3, -1.0 / 12],
        6: [1.0 / 90, -3.0 / 20, 3.0 / 2, -49.0 / 18, 3.0 / 2, -3.0 / 20,
            1.0 / 90],
    },
}

ACCURACIES = (2, 4, 6)
DEFAULT_ACCURACY = 4


def parse_axes(spec):
    """Turn ``"xy"`` / ``["x", "y"]`` into a sorted tuple of axis indices.

    Raises ValueError for unknown letters, an empty spec, or more than two
    derivatives along one axis.
    """
    letters = "".join(spec) if not isinstance(spec, str) else spec
    letters = letters.strip().lower()
    if not letters:
        raise ValueError("empty derivative spec")
    bad = [c for c in letters if c not in AXES]
    if bad:
        raise ValueError("derivative %r: unknown axis %r (use x, y, z)"
                         % (spec, bad[0]))
    axes = tuple(sorted(AXES.index(c) for c in letters))
    if any(axes.count(a) > 2 for a in set(axes)):
        raise ValueError("derivative %r: at most second order along one axis"
                         % (spec,))
    return axes


def axis_counts(axes, ndim=3):
    """How many times each axis is differentiated: ``(0, 2, 0)`` for ``yy``."""
    return tuple(axes.count(a) for a in range(ndim))


def half_width(accuracy):
    """Stencil half-width in cells for a centred stencil of this accuracy."""
    if accuracy not in ACCURACIES:
        raise ValueError("accuracy must be one of %s (got %r)"
                         % (ACCURACIES, accuracy))
    return accuracy // 2


def trim(axes, accuracy, ndim=3):
    """Cells lost at *each* edge of each axis: an int array of length ``ndim``."""
    w = half_width(accuracy)
    return np.array([w if n else 0 for n in axis_counts(axes, ndim)], dtype=int)


def differentiate(data, dx, axes, accuracy=DEFAULT_ACCURACY):
    """Centred finite-difference derivative of a uniformly spaced array.

    Parameters
    ----------
    data : ndarray
        The field, axis order matching ``dx`` (x, y, z for postcactus data).
    dx : sequence of float
        Grid spacing along each axis.
    axes : tuple of int
        From :func:`parse_axes`; e.g. ``(0, 1)`` for d^2/dxdy.
    accuracy : int
        Order of accuracy of the stencils (2, 4 or 6).

    Returns
    -------
    (result, cut) where ``cut`` is :func:`trim` and ``result`` has shape
    ``data.shape - 2 * cut``: the derivative at input points
    ``cut[k] .. shape[k] - 1 - cut[k]`` along each axis. Raises ValueError if
    the array is too small for the stencil along some axis.
    """
    data = np.asarray(data, dtype=float)
    ndim = data.ndim
    counts = axis_counts(axes, ndim)
    cut = trim(axes, accuracy, ndim)
    if any(data.shape[k] <= 2 * cut[k] for k in range(ndim)):
        raise ValueError("array of shape %s too small for a %d-point stencil"
                         % (data.shape, 2 * half_width(accuracy) + 1))

    out = data
    for k, order in enumerate(counts):
        if order == 0:
            continue
        coeffs = _STENCILS[order][accuracy]
        w = cut[k]
        n_out = out.shape[k] - 2 * w
        acc = np.zeros(out.shape[:k] + (n_out,) + out.shape[k + 1:])
        for j, c in enumerate(coeffs):
            if c == 0.0:
                continue
            sl = [slice(None)] * ndim
            sl[k] = slice(j, j + n_out)
            acc += c * out[tuple(sl)]
        out = acc / float(dx[k]) ** order

    # Axes not differentiated along are untouched (cut = 0 there).
    return out, cut


class Derivative(object):
    """One requested derivative: a source field and the axes to take it along."""

    def __init__(self, field, axes):
        self.field = str(field)
        self.axes = parse_axes(axes) if not isinstance(axes, tuple) else axes

    @property
    def letters(self):
        """``"xy"`` for d^2/dxdy."""
        return "".join(AXES[a] for a in self.axes)

    @property
    def order(self):
        return len(self.axes)

    @property
    def name(self):
        """Output variable name: ``dBx_dx``, ``d2gxx_dxdy``, ``d2gxx_dzdz``."""
        prefix = "d" if self.order == 1 else "d%d" % self.order
        return "%s%s_%s" % (prefix, self.field,
                            "".join("d" + c for c in self.letters))

    def as_dict(self):
        return {"field": self.field, "axes": self.letters}

    def __repr__(self):
        return "Derivative(%r, %r)" % (self.field, self.letters)


def parse_config(block):
    """Parse a config ``derivatives`` block into ``(accuracy, [Derivative])``.

    ``block`` may be None/empty (no derivatives). Duplicate requests are
    dropped, keeping the first. Raises ValueError on anything malformed.
    """
    if not block:
        return DEFAULT_ACCURACY, []
    if not isinstance(block, dict):
        raise ValueError("`derivatives` must be a mapping with `fields` "
                         "(and optionally `accuracy`)")
    unknown = set(block) - {"accuracy", "fields"}
    if unknown:
        raise ValueError("`derivatives`: unknown key(s) %s"
                         % ", ".join(sorted(map(repr, unknown))))

    accuracy = int(block.get("accuracy", DEFAULT_ACCURACY))
    half_width(accuracy)  # validates

    fields = block.get("fields") or {}
    if not isinstance(fields, dict):
        raise ValueError("`derivatives.fields` must map a field name to a "
                         "list of derivatives, e.g. `Bx: [x, y, z]`")
    result, seen = [], set()
    for field, specs in fields.items():
        if isinstance(specs, str):
            raise ValueError("`derivatives.fields.%s` must be a list, e.g. "
                             "[x, y, z] (a bare string is ambiguous: is 'xy' "
                             "d/dx and d/dy, or d^2/dxdy?)" % field)
        for spec in specs or []:
            deriv = Derivative(field, str(spec))
            if deriv.name not in seen:
                seen.add(deriv.name)
                result.append(deriv)
    return accuracy, result
