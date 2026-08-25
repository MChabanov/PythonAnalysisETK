#!/usr/bin/env python3
"""Show what plane a 3D resampling config actually cuts, without resampling.

A rotating plane is easy to get subtly wrong - above all through
``plane.time_offset``, which shifts the binary's phase and so rotates every
frame by a constant angle with no other symptom. This prints the frame table
the pipeline would build, so it can be checked in seconds instead of after an
hour-long job.

The most useful check: the printed ``bh1``/``bh2`` positions at t = 0 should
match the puncture positions in the simulation's parfile (``$x_punc1`` and
friends), and ``bh1_uv``/``bh2_uv`` should put both bodies at v = 0 whenever the
plane was built to contain the separation vector.

Usage
-----
    # times taken from an existing output file (the surest source)
    python inspect_plane.py config_3d.yaml --times-from ../2d/rho_b__run_xz.h5

    # or a synthetic cadence: start, stop, number of frames
    python inspect_plane.py config_3d.yaml --times 0 5068.8 100

No MPI, no data read. ``auto`` values need the simulation's parfile, which is
taken from the config's ``simdir_pickle`` when it has one, or from a fresh
directory scan with ``--scan``.
"""

import argparse
import os
import pickle
import sys

import numpy as np
import yaml

import plane_motion


def _load_simdir(cfg, scan):
    """A SimDir for resolving ``auto`` values, or None if not available."""
    pickled = (cfg.get("simdir_pickle") or {})
    path = pickled.get("path")
    if pickled.get("pickled") and path and os.path.isfile(path):
        print("SimDir: unpickled from %s" % path)
        with open(path, "rb") as f:
            return pickle.load(f)

    if not scan:
        return None

    from postcactus.simdir import SimDir
    exclude = cfg.get("simdir_exclude") or {}
    print("SimDir: scanning %s (this is the slow part; a pickle avoids it)"
          % cfg["simdir"])
    kwargs = {"max_depth": int(cfg.get("scan_max_depth", 8))}
    if exclude.get("dirs"):
        kwargs["exclude_dirs"] = list(exclude["dirs"])
    if exclude.get("files"):
        kwargs["exclude_files"] = list(exclude["files"])
    return SimDir(cfg["simdir"], **kwargs)


def _times(args):
    """The (iterations, times) to build frames for."""
    if args.times_from:
        import h5py
        with h5py.File(args.times_from, "r") as h5:
            iterations = np.asarray(h5["iterations"][:], dtype=np.int64)
            times = np.asarray(h5["times"][:], dtype=float)
        print("times: %d frame(s) from %s, t = %g .. %g"
              % (len(times), args.times_from, times[0], times[-1]))
        return iterations, times

    start, stop, count = args.times
    count = int(count)
    times = np.linspace(float(start), float(stop), count)
    iterations = np.arange(count, dtype=np.int64)
    print("times: %d synthetic frame(s), t = %g .. %g (iteration numbers are "
          "placeholders)" % (count, times[0], times[-1]))
    return iterations, times


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="The 3D resampling YAML config.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--times-from", metavar="FILE.h5",
                        help="Take the iteration/time list from an existing "
                             "output file of the same simulation.")
    source.add_argument("--times", nargs=3, metavar=("START", "STOP", "COUNT"),
                        help="Use this many evenly spaced simulation times.")
    parser.add_argument("--scan", action="store_true",
                        help="Scan the simulation directory when no usable "
                             "SimDir pickle is configured (needed only to "
                             "resolve `auto` values from the parfile).")
    parser.add_argument("--rows", type=int, default=10,
                        help="How many frames to print (default 10).")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f) or {}
    plane = cfg.get("plane", "xy")
    kind = plane_motion.plane_kind(plane)
    print("plane kind: %s" % kind)

    iterations, times = _times(args)

    sim = None
    needs_parfile = kind == "trajectory" and "auto" in (
        str(plane.get("file", "auto")), str(plane.get("time_offset", "auto")))
    if needs_parfile:
        sim = _load_simdir(cfg, args.scan)
        if sim is None:
            sys.exit(
                "ERROR: this config uses `auto` for plane.file or "
                "plane.time_offset, which is read from the simulation's "
                "parfile. Either pass --scan, point simdir_pickle at an "
                "existing pickle, or set both values explicitly.")

    print("\nbuilding frames:")
    try:
        table = plane_motion.build_frame_table(plane, iterations, times,
                                               sim=sim, log=print)
    except ValueError as exc:
        sys.exit("ERROR: plane: %s" % exc)

    step = max(1, len(iterations) // max(1, args.rows))
    rows = list(range(0, len(iterations), step))
    fmt3 = lambda v: "[%7.4f %7.4f %7.4f]" % tuple(v)

    print("\nframes (every %d of %d):" % (step, len(iterations)))
    header = "%8s %10s  %-25s %-25s" % ("it", "t", "normal", "u_axis")
    has_bodies = "bh1_uv" in table.extras
    if has_bodies:
        header += "  %-17s %-17s %8s" % ("bh1 (u,v)", "bh2 (u,v)", "sep")
    print(header)
    for i in rows:
        line = "%8d %10.3f  %-25s %-25s" % (
            table.iterations[i], table.times[i],
            fmt3(table.n_hat[i]), fmt3(table.u_hat[i]))
        if has_bodies:
            line += "  %-17s %-17s %8.4f" % (
                "[%7.3f %6.3f]" % tuple(table.extras["bh1_uv"][i]),
                "[%7.3f %6.3f]" % tuple(table.extras["bh2_uv"][i]),
                table.extras["separation"][i])
        print(line)

    if has_bodies:
        print("\nbody positions in simulation coordinates:")
        for i in rows[:3]:
            print("  t = %10.3f   bh1 = %s   bh2 = %s"
                  % (table.times[i], fmt3(table.extras["bh1_position"][i]),
                     fmt3(table.extras["bh2_position"][i])))
        print("  (compare the t = %g row with $x_punc1/$y_punc1 in the parfile "
              "to confirm plane.time_offset)" % table.times[0])

    total, largest, flips = table.continuity_report()
    print("\nframe turns %.1f degrees over %d frame(s); largest step %.2f "
          "degrees; reversals %d" % (total, len(iterations), largest, flips))
    if table.moving:
        print("output datasets that will carry the geometry: %s"
              % ", ".join(sorted(table.per_iteration_data(iterations[:1]))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
