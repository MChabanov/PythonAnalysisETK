#!/usr/bin/env python3
"""How the classification of the shell's points depends on the tracer settings.

    python check_tracer.py config_shell.yaml --iteration 204800 [--seeds 200000]

On one iteration, traces the field lines through a random subset of the
shell's points with the config's settings and with each setting changed on its
own: half the step, twice the length cap, a box 4 M wider, horizons 10%
larger. Reports per variant the seeds per second, the share of seeds whose
class changes, and the volume share and the three integrals of the points with
a horizon end, without one, and undetermined (cap or null, no horizon end) -
scaled from the subset to the shell. One rank (plain ``python``), all cores
through numba threads.
"""

import argparse
import os
import resource
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import shell_integrals as si          # noqa: E402  (sets up the import paths)
from amr_blocks import Snapshot      # noqa: E402
import tracer                        # noqa: E402


def groups(res):
    """Masks of the seeds with a horizon end / none, both exit / undetermined."""
    n_h = len(res.legend) - 3
    ends = res.ends
    horizon = np.any(ends < n_h, axis=1)
    free = np.all(ends == n_h, axis=1)
    return {"horizon": horizon, "free": free,
            "undetermined": ~horizon & ~free}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("config")
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--seeds", type=int, default=200000)
    parser.add_argument("--box-extra", type=float, default=4.0)
    args = parser.parse_args()

    cfg = si.load_config(args.config)
    cfg["iteration_min"] = cfg["iteration_max"] = args.iteration
    tr = cfg["tracer"]
    print("numba: %d thread(s), %d core(s)" % si.set_threads(cfg), flush=True)
    backend, xyz, tables = si.prepare(cfg)
    row = 0
    it = int(tables["iterations"][row])
    centre, pole = tables["centre"][row], tables["pole"][row]
    print("iteration %d, t = %g, shell centre %s"
          % (it, tables["times"][row], np.round(centre, 4).tolist()), flush=True)

    t0 = time.perf_counter()
    snap = Snapshot(xyz, it, cfg["fields"]["B"][0], tocs=backend._tocs)
    sp = si.ShellPoints(cfg, snap, centre, pole)
    print("shell: %d points, volume %.6g (exact %.6g), %.0f s"
          % (len(sp), sp.weight.sum(), 4 * np.pi / 3 * (
              cfg["shell"]["r_out"] ** 3 - cfg["shell"]["r_in"] ** 3),
             time.perf_counter() - t0), flush=True)
    rng = np.random.default_rng(1)
    pick = np.sort(rng.choice(len(sp), size=min(args.seeds, len(sp)),
                              replace=False))
    seeds = sp.points[pick]
    scale = len(sp) / len(pick)
    norms = {"Phi_phi": 2 * np.pi,
             "Phi_r": cfg["shell"]["r_out"] - cfg["shell"]["r_in"],
             "Phi_z": 2 * cfg["shell"]["r_out"]}
    f = {"Phi_phi": sp.f_phi[pick], "Phi_r": sp.f_r[pick], "Phi_z": sp.f_z[pick]}
    w = sp.weight[pick]
    totals = {q: scale * np.sum(w * f[q]) / norms[q] for q in f}
    print("unmasked (from the subset): " + ", ".join(
        "%s %.5g" % (q, v) for q, v in totals.items()), flush=True)

    def report(name, res, base=None):
        g = groups(res)
        rate = len(seeds) / max(res.seconds, 1e-9)
        line = "%-22s %8.0f seeds/s" % (name, rate)
        if base is not None:
            line += "  changed %6.3f%%" % (100 * np.mean(
                res.classes() != base.classes()))
        print(line, flush=True)
        for gname, m in g.items():
            vol = np.sum(w[m]) / np.sum(w)
            vals = ", ".join("%s %.4g (%.1f%%)"
                             % (q, scale * np.sum(w[m] * f[q][m]) / norms[q],
                                100 * np.sum(w[m] * f[q][m]) / np.sum(w * f[q]))
                             for q in f)
            print("    %-13s volume %5.1f%%  %s" % (gname, 100 * vol, vals),
                  flush=True)
        return res

    variants = [("base", {}), ("step / 2", {"step": 0.5 * float(tr["step"])}),
                ("max_length x 2",
                 {"max_length": 2 * float(tr["max_length"])}),
                ("horizons x 1.1", {"scale": 1.1})]
    t0 = time.perf_counter()
    field = si.tracing_field(cfg, snap, centre)
    print("field: %.2f GB in %.0f s; %s" % (
        field.nbytes() / 1e9, time.perf_counter() - t0,
        "; ".join(field.describe())), flush=True)
    base = None
    for name, opt in variants:
        hz = si.horizons_at(tables, row, opt.get("scale", 1.0))
        res = tracer.trace(field, hz, seeds,
                           step=opt.get("step", float(tr["step"])),
                           max_length=opt.get("max_length",
                                              float(tr["max_length"])))
        report(name, res, base)
        if base is None:
            base = res
            print("    classes: " + ", ".join(
                "%s %.2f%%" % (n, 100 * c / len(seeds)) for n, c in zip(
                    res.class_names(),
                    np.bincount(res.classes(), minlength=len(
                        res.class_names()))) if c), flush=True)
            print("    median half-line length %.2f M, capped half-lines "
                  "%.2f%%" % (np.median(res.lengths),
                              100 * np.mean(res.ends == len(res.legend) - 2)),
                  flush=True)
    del field
    half = float(tr["box_half_width"]) + args.box_extra
    field = si.tracing_field(cfg, snap, centre, half)
    res = tracer.trace(field, si.horizons_at(tables, row), seeds,
                       step=float(tr["step"]), max_length=float(tr["max_length"]))
    report("box half-width %g" % half, res, base)
    snap.close()
    print("peak memory %.1f GB" % (resource.getrusage(resource.RUSAGE_SELF)
                                   .ru_maxrss / 2 ** 20), flush=True)


if __name__ == "__main__":
    main()
