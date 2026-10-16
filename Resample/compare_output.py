#!/usr/bin/env python3
"""Compare two resampled HDF5 files produced by the Resample pipeline.

The point of this is cross-validation: an axis-aligned plane cut out of the 3D
output by ``resample_3d.py`` must reproduce the same plane resampled from
Carpet's own 2D output by ``resample_2d.py``, because ``plane: xy|xz|yz`` uses
exactly the same in-plane axes and the same interpolation. Running this after a
change to the pipeline is the cheapest way to confirm nothing drifted.

It also works for any other pair with a common grid - two backends, two
resolutions of the same run (with ``--iterations`` to line them up), before and
after a code change.

Usage
-----
    python compare_output.py A.h5 B.h5
    python compare_output.py A.h5 B.h5 --tol 1e-10 --quiet

Compares only the iterations present in both files. Exits non-zero if the
grids are incompatible or the largest relative difference exceeds ``--tol``.
"""

import argparse
import sys

import numpy as np
import h5py


def _relative_difference(a, b):
    """Elementwise |a - b| / max(|a|, |b|), defined as 0 where both are 0."""
    scale = np.maximum(np.abs(a), np.abs(b))
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = np.where(scale > 0, np.abs(a - b) / scale, 0.0)
    return rel


def compare(path_a, path_b, tol=1e-9, quiet=False):
    """Compare two output files; return the largest relative difference."""
    with h5py.File(path_a, "r") as fa, h5py.File(path_b, "r") as fb:
        for axis in ("x", "y"):
            xa, xb = fa[axis][:], fb[axis][:]
            if xa.shape != xb.shape:
                sys.exit("ERROR: %s axis has %d points in A but %d in B - the "
                         "two files are on different grids"
                         % (axis, xa.shape[0], xb.shape[0]))
            if not np.allclose(xa, xb, rtol=0, atol=1e-12):
                sys.exit("ERROR: the %s axes differ (A spans %g..%g, B spans "
                         "%g..%g)" % (axis, xa[0], xa[-1], xb[0], xb[-1]))

        its_a, its_b = fa["iterations"][:], fb["iterations"][:]
        index_b = {int(it): i for i, it in enumerate(its_b)}
        shared = [(i, index_b[int(it)]) for i, it in enumerate(its_a)
                  if int(it) in index_b]
        if not shared:
            sys.exit("ERROR: the two files share no iterations (A has %d..%d, "
                     "B has %d..%d)" % (its_a[0], its_a[-1], its_b[0], its_b[-1]))

        if not quiet:
            print("A: %s  [%s, %s, %d iterations]"
                  % (path_a, fa.attrs.get("backend"), fa.attrs.get("plane"),
                     its_a.size))
            print("B: %s  [%s, %s, %d iterations]"
                  % (path_b, fb.attrs.get("backend"), fb.attrs.get("plane"),
                     its_b.size))
            print("comparing %d shared iteration(s) on a %s grid"
                  % (len(shared), tuple(fa["data"].shape[1:])))

        worst = -1.0          # so the first iteration always sets worst_it
        worst_it = None
        for ia, ib in shared:                       # one slice at a time
            a, b = fa["data"][ia], fb["data"][ib]
            rel = _relative_difference(a, b)
            peak = float(rel.max())
            if not quiet:
                print("  it %8d   max rel %.3e   median rel %.3e   "
                      "identical: %s"
                      % (its_a[ia], peak, float(np.median(rel)),
                         np.array_equal(a, b)))
            if peak > worst:
                worst, worst_it = peak, int(its_a[ia])

    print("worst relative difference: %.3e (iteration %s), tolerance %.1e"
          % (worst, worst_it, tol))
    return worst


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file_a")
    parser.add_argument("file_b")
    parser.add_argument("--tol", type=float, default=1e-9,
                        help="Fail above this relative difference "
                             "(default 1e-9; the same plane resampled two ways "
                             "typically agrees to ~1e-13).")
    parser.add_argument("--quiet", action="store_true",
                        help="Only print the summary line.")
    args = parser.parse_args()

    worst = compare(args.file_a, args.file_b, tol=args.tol, quiet=args.quiet)
    if worst > args.tol:
        print("MISMATCH")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
