#!/usr/bin/env python3
"""Read the output of ``shell_integrals.py`` and group its classes (numpy + h5py).

    from integrals_tools import load_integral, group

    phi = load_integral("Phi_r__HHR_newcool_30M_flux_shell_BH1_r4-5.h5")
    phi["times"]                         # (n_iter,)
    phi["classes"]                       # ["BH1-BH1", "BH1-BH2", "BH1-exit", ...]

    phi["data"]                          # (n_iter, n_hscale, n_class)
    total = group(phi, "total")          # every point: the unmasked integral
    outside = group(phi, "no_horizon")   # points whose line reaches no horizon
    bh1 = group(phi, "BH1")              # lines with an end on BH1
    bh1_big = group(phi, "BH1", horizon_scale=1.1)   # horizons 10% larger

    from integrals_tools import load_points
    pts = load_points("points__<label>__it204800.h5")   # one iteration, per point

Groups (each a sum over classes, ``(n_iter,)``; for ``Phi_z`` also over the
``profile`` with ``key="profile"``):

``total``          all classes
``horizon``        at least one end on a horizon - the holes' magnetospheres
``BH1`` / ``BH2``  at least one end on that horizon
``no_horizon``     no end on a horizon (``free`` + ``undetermined``)
``free``           both ends leave the tracing box
``undetermined``   no horizon end, at least one capped or null end
"""

import h5py
import numpy as np

GROUPS = ("total", "horizon", "BH1", "BH2", "no_horizon", "free",
          "undetermined")


def _strings(values):
    return [v.decode() if isinstance(v, bytes) else str(v) for v in values]


def load_integral(path):
    """One output file as a dict: datasets as arrays, attributes as values."""
    out = {}
    with h5py.File(path, "r") as h5:
        for name, dset in h5.items():
            out[name] = dset[()]
        for name, val in h5.attrs.items():
            if name in out:                # a dataset wins over an attribute
                continue
            out[name] = _strings(val) if name in ("classes", "ends") else val
    return out


def load_points(path):
    """One ``points__`` file: per point positions, volumes, levels, end codes.

    Adds ``classes`` (class index per point, first horizon scale) and
    ``class_names``; ``xclasses[:, k]`` for the k-th extra horizon scale.
    """
    out = load_integral(path)
    legend = _strings(out["ends_legend"])
    n = len(legend)

    def classify(ends):
        a = np.minimum(ends[:, 0], ends[:, 1]).astype(np.int64)
        b = np.maximum(ends[:, 0], ends[:, 1]).astype(np.int64)
        return a * n - a * (a - 1) // 2 + (b - a)

    out["ends_legend"] = legend
    out["class_names"] = ["%s-%s" % (legend[a], legend[b])
                          for a in range(n) for b in range(a, n)]
    out["classes"] = classify(out["ends"])
    out["xclasses"] = np.stack([classify(out["xends"][:, :, k]) for k in
                                range(out["xends"].shape[2])], axis=1) \
        if out["xends"].shape[2] else np.zeros((len(out["ends"]), 0), int)
    return out


def class_mask(classes, name):
    """Which classes (``"A-B"`` names) belong to the group ``name``."""
    horizons = set()
    for c in classes:
        for end in c.split("-"):
            if end not in ("exit", "cap", "null"):
                horizons.add(end)
    mask = []
    for c in classes:
        ends = c.split("-")
        on = [e for e in ends if e in horizons]
        if name == "total":
            mask.append(True)
        elif name == "horizon":
            mask.append(bool(on))
        elif name in horizons:
            mask.append(name in ends)
        elif name == "no_horizon":
            mask.append(not on)
        elif name == "free":
            mask.append(ends == ["exit", "exit"])
        elif name == "undetermined":
            mask.append(not on and ends != ["exit", "exit"])
        else:
            raise KeyError("unknown group %r (known: %s, or a horizon name)"
                           % (name, ", ".join(GROUPS)))
    return np.array(mask)


def group(result, name, key="data", horizon_scale=None):
    """Sum of ``result[key]`` over the classes of group ``name``, ``(n_iter, ...)``.

    ``horizon_scale`` picks the horizon size (one of ``result["horizon_scales"]``;
    default the first, the one traced with).
    """
    arr = np.asarray(result[key])
    scales = np.atleast_1d(result.get("horizon_scales", [1.0]))
    j = 0
    if horizon_scale is not None:
        hit = np.flatnonzero(np.isclose(scales, horizon_scale))
        if hit.size == 0:
            raise KeyError("horizon_scale %g not in %s" % (horizon_scale, scales))
        j = int(hit[0])
    mask = class_mask(result["classes"], name)
    return arr[:, j][:, mask].sum(axis=1)
