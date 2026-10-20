#!/usr/bin/env python3
"""End-to-end test of the shell-integral driver on synthetic data.

    python test_driver.py     (one line per check, then ALL OK)

Runs ``shell_integrals.process`` on one iteration of a synthetic hierarchy (the
one of ``test_shell_cells.py``, radial field, a uniform B along z traced with
a unit-sphere "BH1" at the shell centre), writes the outputs, and checks: the
classes (lines through R < 1 reach BH1) and their volumes; the enlarged
horizon (1.2x) recorded from the same lines; a second run reading the ends
from the ``points__`` file instead of tracing, with identical integrals; a
change of settings forcing a new trace; the output layout and
``integrals_tools``.
"""

import os
import sys
import tempfile

import numpy as np
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import shell_integrals as si          # noqa: E402  (sets up the import paths)
import test_shell_cells as T         # noqa: E402
from amr_field import AMRVectorField  # noqa: E402
import integrals_tools               # noqa: E402

FAILURES = []
CENTRE = np.array([0.37, -0.21, 0.13])


def check(name, ok, detail=""):
    print("%s %s%s" % ("ok  " if ok else "FAIL", name,
                       ("  (%s)" % detail) if detail else ""), flush=True)
    if not ok:
        FAILURES.append(name)


class FakeSnapshot(object):
    """The ``amr_blocks.Snapshot`` interface over the synthetic hierarchy."""

    def __init__(self, xyz, it, ref_field, tocs=None):
        self.components = T.hierarchy(0.5)
        self._read = T.make_read("radial", CENTRE, [0.0, 0.0, 1.0])
        self.closed = False

    def read(self, var, comp, i0, i1):
        return self._read(var, comp, i0, i1)

    def close(self):
        self.closed = True


def fake_field(cfg, snap, centre, half=None):
    half = float(cfg["tracer"]["box_half_width"])

    def uniform(x, y, z):
        return 0 * x, 0 * x, 1 + 0 * x

    return AMRVectorField.from_function(
        uniform, [(0.25, centre - half - 1, centre + half + 1, 2)],
        centre - half, centre + half)


def config(tmp, **tracer):
    raw = {"simdir": "/nonexistent", "label": "synthetic", "output_dir": tmp,
           "shell": {"r_in": 4.0, "r_out": 5.0, "centre": CENTRE.tolist()},
           "tracer": dict({"box_half_width": 6.0, "step": 0.5,
                           "max_length": 50.0, "extra_horizon_scales": [1.2],
                           "batches": 1}, **tracer),
           "save_points": [7]}
    path = os.path.join(tmp, "cfg.yaml")
    with open(path, "w") as f:
        yaml.safe_dump(raw, f)
    return si.load_config(path)


def tables():
    hz = (np.array([CENTRE, CENTRE + 100.0]), np.array([[0, 0, 1.0]] * 2),
          np.array([1.0, 1.0]), np.array([1.0, 1.0]))
    return {"iterations": np.array([7]), "times": np.array([1.5]),
            "centre": CENTRE[None], "pole": np.array([[0, 0, 1.0]]),
            "phi_zero": np.array([[1.0, 0, 0]]),
            "bh1_position": CENTRE[None], "bh2_position": CENTRE[None] + 100,
            "horizons": [hz]}


def main():
    si.Snapshot = FakeSnapshot
    si.tracing_field = fake_field
    with tempfile.TemporaryDirectory() as tmp:
        cfg = config(tmp)
        tab = tables()
        first = si.process(cfg, None, None, tab, 0)
        legend = first["classes"]
        V = 4 * np.pi / 3 * (5 ** 3 - 4 ** 3)
        vol = dict(zip(legend, first["volume"][0]))
        # Lines are vertical: those with R < 1 reach BH1 - the shell's two
        # polar plugs, of volume 2 int_0^1 2 pi R (sqrt(25-R^2) - sqrt(16-R^2)) dR.
        def plugs(radius):
            Rs = np.linspace(0, radius, 20001)
            h = np.sqrt(25 - Rs ** 2) - np.sqrt(16 - Rs ** 2)
            return 2 * np.trapz(2 * np.pi * Rs * h, Rs)

        plug = plugs(1.0)
        check("BH1-exit volume = the polar plugs R < 1",
              abs(vol["BH1-exit"] / plug - 1) < 0.02,
              "%.4f vs %.4f" % (vol["BH1-exit"], plug))
        check("the rest exit-exit, volumes add up",
              abs(vol["exit-exit"] + vol["BH1-exit"] - V) / V < 1e-3)
        plug12 = plugs(1.2)
        vol12 = dict(zip(legend, first["volume"][1]))["BH1-exit"]
        check("1.2x horizon from the same lines: plugs R < 1.2",
              abs(vol12 / plug12 - 1) < 0.02, "%.4f vs %.4f" % (vol12, plug12))
        path = si.points_path(cfg, 7)
        pts = integrals_tools.load_points(path)
        check("points file: every point, heavy data for save_points",
              len(pts["points"]) == first["n_points"][0].sum()
              and "endpoints" in pts and first["traced"] == 1)

        again = si.process(cfg, None, None, tab, 0)
        same = all(np.array_equal(again[q], first[q])
                   for q in si.QUANTITIES + ("profile", "n_points"))
        check("rerun reads the ends, identical integrals",
              again["traced"] == 0 and same and np.all(again["timings"][3] == 0))

        changed = config(tmp, step=0.4)
        third = si.process(changed, None, None, tab, 0)
        check("other tracer settings: traced again", third["traced"] == 1)

        si.write_outputs(cfg, tab, {0: first})
        res = integrals_tools.load_integral(
            os.path.join(tmp, "Phi_r__synthetic.h5"))
        ok = (res["data"].shape == (1, 2, len(legend))
              and np.allclose(res["horizon_scales"], [1.0, 1.2]))
        total = integrals_tools.group(res, "total")
        parts = (integrals_tools.group(res, "horizon")
                 + integrals_tools.group(res, "no_horizon"))
        big = integrals_tools.group(res, "BH1", horizon_scale=1.2)
        check("output layout and groups", ok and np.allclose(total, parts)
              and big[0] > integrals_tools.group(res, "BH1")[0],
              "data %s" % (res["data"].shape,))
        z = integrals_tools.load_integral(
            os.path.join(tmp, "Phi_z__synthetic.h5"))
        check("profile per horizon scale, mean = data",
              np.allclose(z["profile"].mean(axis=-1), z["data"]))
        tr = integrals_tools.load_integral(
            os.path.join(tmp, "tracing__synthetic.h5"))
        check("tracing file", tr["n_points"].shape == (1, 2, len(legend))
              and tr["traced"][0] == 1 and tr["mean_length"].shape[-1]
              == len(legend))
    if FAILURES:
        print("FAILED: %d check(s): %s" % (len(FAILURES), "; ".join(FAILURES)))
        sys.exit(1)
    print("ALL OK")


if __name__ == "__main__":
    main()
