#!/usr/bin/env python3
"""Example plane module for ``plane: {kind: analytic, module: ...}``.

Copy this next to your config, edit it, and point ``plane.module`` at it:

    plane:
      kind: analytic
      module: ./my_plane.py

The contract - every function optional except ``normal``:

    normal(t, it)    -> (nx, ny, nz)   required; need not be a unit vector
    origin(t, it)    -> (x, y, z)      default (0, 0, 0)
    u_axis(t, it)    -> (ux, uy, uz)   default: see below

``t`` is the simulation time of the frame and ``it`` its iteration number. Both
are passed, so the plane can be driven by whichever is more natural.

Only rank 0 imports this file - the frames are precomputed there and broadcast -
so it may do anything, including read files of its own. It is called once per
output frame; if it happens to accept numpy arrays it will be called once with
all times at once instead, which is a bonus rather than a requirement.

Leaving out ``u_axis`` lets the pipeline pick the coordinate axis least aligned
with the plane normal over the whole run and project that in; this is continuous
in time, unlike the static plane's convention. Define ``u_axis`` when you want a
specific in-plane orientation - e.g. so a feature stays put across the movie.

A module can also supply pieces of a *trajectory* plane, by naming ``module``
alongside ``second_vector: module``, ``normal: module``, ``origin: module`` or
``u_axis: module``; the functions used are then ``second_vector(t, it)`` and the
ones above.
"""

import numpy as np

# --- Edit these -------------------------------------------------------------

TILT = np.radians(30.0)     # angle of the normal away from the z axis
PERIOD = 1076.9             # code-time units per full turn of the plane
ORIGIN = (0.0, 0.0, 0.0)    # the plane's centre


def normal(t, it):
    """The plane normal at simulation time ``t``.

    This one precesses: the normal keeps a fixed angle ``TILT`` to the z axis
    and sweeps once around it every ``PERIOD``.
    """
    phase = 2.0 * np.pi * t / PERIOD
    return (np.sin(TILT) * np.cos(phase),
            np.sin(TILT) * np.sin(phase),
            np.cos(TILT) * np.ones_like(phase))


def origin(t, it):
    """Where the in-plane coordinates are centred (constant here)."""
    return np.broadcast_to(np.asarray(ORIGIN, dtype=float),
                           np.shape(t) + (3,) if np.ndim(t) else (3,))


def u_axis(t, it):
    """The +u direction, i.e. the output's ``x`` axis.

    Keeping u in the xy plane and perpendicular to the normal's azimuth makes
    the tilt appear as a pure "nodding" of the frame rather than a rotation.
    """
    phase = 2.0 * np.pi * t / PERIOD
    zero = np.zeros_like(phase)
    return (-np.sin(phase), np.cos(phase), zero)
