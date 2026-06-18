#!/usr/bin/env python3
"""Resample 2D slices from an Einstein Toolkit simulation into HDF5 files.

kuibit backend. This is the kuibit (https://github.com/Sbozzolo/kuibit) port of
``resample_2d_data_postcactus.py`` (which uses the older postcactus). The two
backends are interchangeable and produce files readable by the same
``read_data.py``: all orchestration (MPI distribution, YAML config, pickled
SimDir cache, the serial *and* parallel iteration query, warm-state gather,
round-robin scheduling, HDF5 streaming output and the on-disk schema) lives in
``resample_common.py``. This module contains *only* the coupling to kuibit,
collected in :class:`KuibitBackend`.

In kuibit the whole postcactus read chain collapses into:

    sim.gridfunctions[plane][variable][iteration].to_UniformGridData(...)

Usage
-----
    mpirun -n <N> python resample_2d_data_kuibit.py config.yaml

(or via the launcher: ``mpirun -n <N> python resample_2d.py config.yaml`` with
``backend: kuibit`` in the config.)

See ``config_example.yaml`` for all options. ``interp_order >= 1`` selects
multilinear resampling (kuibit ``resample=True``); ``interp_order == 0`` selects
nearest-neighbour (``resample=False``).
"""

import numpy as np

from kuibit.simdir import SimDir

from resample_common import Backend, grid_bounds, log, run


class KuibitBackend(Backend):
    """All kuibit-specific coupling for the resampling pipeline.

    The "plane index" is ``sim.gridfunctions[plane]`` (an ``AllGridFunctions``).
    Its per-variable warm state is a single object - the ``OneGridFunctionH5``
    cached at ``plane_index._vars[var]`` (with its parsed ``alldata`` and
    ``restarts_data``) - so gathering warm state is just moving those objects.
    """

    name = "kuibit"

    def scan(self, cfg):
        if cfg["simdir_exclude"]["dirs"] or cfg["simdir_exclude"]["files"]:
            log("  WARNING: simdir_exclude is not supported by the kuibit "
                "backend and will be ignored")
        return SimDir(cfg["simdir"])

    def plane_index(self, sim, cfg):
        return sim.gridfunctions[cfg["plane"]]

    def query(self, plane_index, var):
        reader = plane_index[var]
        return reader.available_iterations, reader.available_times

    def extract_warm_state(self, plane_index, variables):
        return {var: plane_index._vars.get(var) for var in variables}

    def inject_warm_state(self, plane_index, warm):
        for var, reader in warm.items():
            if reader is not None:
                plane_index._vars[var] = reader

    def read_slice(self, plane_index, var, it, cfg):
        resolution, min_corner, max_corner = grid_bounds(cfg["grid"])
        # kuibit: resample=True is multilinear interpolation, False is nearest.
        resample = int(cfg["interp_order"]) >= 1
        hgd = plane_index[var][int(it)]                 # read from disk
        ugd = hgd.to_UniformGridData(                   # merge + interpolate
            resolution, min_corner, max_corner,
            resample=resample, iteration=int(it),
        )
        return np.asarray(ugd.data)


def main():
    run(KuibitBackend())


if __name__ == "__main__":
    main()
