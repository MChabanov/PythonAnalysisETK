#!/usr/bin/env python3
"""Single entry point for the 3D -> sphere resampling pipeline.

The counterpart of ``resample_3d.py``: that launcher cuts a plane out of the
full 3D output, this one samples spheres on a uniform (theta, phi) grid (see
``resample_sphere_data_postcactus.py``).

    mpirun -n <N> python resample_sphere.py config_sphere.yaml

``backend`` in the config selects the reader library, exactly as for 2D/3D:

    backend: postcactus   ->  resample_sphere_data_postcactus.py

If ``backend`` is omitted it defaults to ``postcactus``, currently the only
sphere backend. The output follows the usual schema - one HDF5 file per
variable, ``x``/``y`` holding theta/phi - with a ``radius`` axis added when
several radii are sampled.
"""

import argparse
import importlib
import sys

import yaml

BACKENDS = {
    "postcactus": "resample_sphere_data_postcactus",
}

DEFAULT_BACKEND = "postcactus"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Path to the YAML configuration file.")
    args = parser.parse_args()

    # Peek only the backend key here; the backend module re-reads and validates
    # the full config itself.
    with open(args.config) as f:
        cfg = yaml.safe_load(f) or {}
    backend = str(cfg.get("backend") or DEFAULT_BACKEND).lower()

    if backend not in BACKENDS:
        sys.exit(
            "ERROR: unknown sphere backend %r (choose one of: %s)"
            % (backend, ", ".join(sorted(BACKENDS)))
        )

    # Importing the backend module initialises MPI and pulls in its heavy
    # dependency (postcactus) only for the backend actually selected.
    module = importlib.import_module(BACKENDS[backend])
    module.main()


if __name__ == "__main__":
    main()
