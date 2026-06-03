#!/usr/bin/env python3
"""Single entry point for the 2D resampling pipeline.

Reads the ``backend`` key from the YAML config and dispatches to the matching
implementation, so you launch the same way regardless of which library is
available in your environment:

    mpirun -n <N> python resample_2d.py config.yaml

``backend`` in the config selects the implementation:

    backend: postcactus   ->  resample_2d_data_postcactus.py
    backend: kuibit       ->  resample_2d_data_kuibit.py

If ``backend`` is omitted it defaults to ``postcactus`` (the original behaviour).
Each backend script can still be run directly if you prefer.
"""

import argparse
import importlib
import sys

import yaml

BACKENDS = {
    "postcactus": "resample_2d_data_postcactus",
    "kuibit": "resample_2d_data_kuibit",
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
            "ERROR: unknown backend %r (choose one of: %s)"
            % (backend, ", ".join(sorted(BACKENDS)))
        )

    # Importing the backend module initialises MPI and pulls in its heavy
    # dependency (postcactus or kuibit) only for the backend actually selected.
    module = importlib.import_module(BACKENDS[backend])
    module.main()


if __name__ == "__main__":
    main()
