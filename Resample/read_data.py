#!/usr/bin/env python3
"""Helpers for reading the HDF5 files produced by the resample_2d pipeline.

Each file holds one variable as a stack of resampled 2D slices:

    data        (n_iter, nx, ny)   the resampled field
    iterations  (n_iter,)          ET iteration numbers
    times       (n_iter,)          physical times
    x, y        (nx,), (ny,)       coordinate axes
    attrs:      variable, label, plane, simdir, interp_order, resolution

Example
-------
    from read_data import load_variable, slice_at_iteration

    rho = load_variable("resampled/rho_b__dU_10_15_linear_HR.h5")
    print(rho["times"][0], rho["data"][0].shape)

    field, t = slice_at_iteration(
        "resampled/rho_b__dU_10_15_linear_HR.h5", iteration=1024
    )
"""

import numpy as np
import h5py


def load_variable(path):
    """Load an entire variable file into memory as a dict of numpy arrays.

    Returns keys: data, iterations, times, x, y, plus the file attributes.
    For large files prefer `slice_at_iteration` / `iter_slices` to avoid
    pulling the whole stack into RAM.
    """
    out = {}
    with h5py.File(path, "r") as h5:
        out["data"] = h5["data"][:]
        out["iterations"] = h5["iterations"][:]
        out["times"] = h5["times"][:]
        out["x"] = h5["x"][:]
        out["y"] = h5["y"][:]
        out["attrs"] = dict(h5.attrs)
    return out


def slice_at_iteration(path, iteration):
    """Return (field, time) for a single iteration without loading the rest."""
    with h5py.File(path, "r") as h5:
        iters = h5["iterations"][:]
        idx = np.where(iters == iteration)[0]
        if idx.size == 0:
            raise KeyError("iteration %d not in %s" % (iteration, path))
        i = int(idx[0])
        return h5["data"][i], float(h5["times"][i])


def iter_slices(path):
    """Yield (iteration, time, field) one slice at a time (memory friendly)."""
    with h5py.File(path, "r") as h5:
        iters = h5["iterations"][:]
        times = h5["times"][:]
        for i in range(iters.shape[0]):
            yield int(iters[i]), float(times[i]), h5["data"][i]


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("usage: python read_data.py <file.h5>")
        sys.exit(1)

    data = load_variable(sys.argv[1])
    print("variable :", data["attrs"].get("variable"))
    print("plane    :", data["attrs"].get("plane"))
    print("shape    :", data["data"].shape)
    print("iters    :", data["iterations"][:5], "...")
    print("times    :", data["times"][:5], "...")
