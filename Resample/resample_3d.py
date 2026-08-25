#!/usr/bin/env python3
"""Single entry point for the 3D -> plane resampling pipeline.

The counterpart of ``resample_2d.py``: that launcher reads slices Carpet
already wrote in 2D, this one reads the full 3D output and cuts an arbitrary
plane out of it (see ``resample_3d_data_postcactus.py``).

    mpirun -n <N> python resample_3d.py config_3d.yaml

``backend`` in the config selects the reader library, exactly as for 2D:

    backend: postcactus   ->  resample_3d_data_postcactus.py

If ``backend`` is omitted it defaults to ``postcactus``, currently the only 3D
backend. The output schema is the same as the 2D pipeline's, so ``read_data.py``
and ``../Frames/make_frames.py`` work on it unchanged.
"""

import argparse
import importlib
import sys

import yaml

BACKENDS = {
    "postcactus": "resample_3d_data_postcactus",
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
            "ERROR: unknown 3D backend %r (choose one of: %s)"
            % (backend, ", ".join(sorted(BACKENDS)))
        )

    # Importing the backend module initialises MPI and pulls in its heavy
    # dependency (postcactus) only for the backend actually selected.
    module = importlib.import_module(BACKENDS[backend])
    module.main()


if __name__ == "__main__":
    main()
