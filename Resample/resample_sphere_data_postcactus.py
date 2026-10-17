#!/usr/bin/env python3
"""Resample spheres out of 3D Einstein Toolkit output onto (theta, phi) grids.

postcactus backend. The sphere counterpart of ``resample_3d_data_postcactus.py``,
of which it is a thin subclass: only the sampled surface differs. Each output
point is interpolated from the finest refinement level that covers it, exactly
as for a plane cut - same component culling, hyperslab reads, ownership rule,
derivatives, caches, MPI structure and chunk merge.

Usage
-----
    mpirun -n <N> python resample_sphere.py config_sphere.yaml
    mpirun -n <N> python resample_sphere_data_postcactus.py config_sphere.yaml

See ``config_example_sphere.yaml`` for all options.

The sphere block
----------------
::

    sphere:
      radius: 3.0              # or several: [2.0, 3.0, 5.0]
      resolution: [300, 600]   # n_theta, n_phi
      centre: bh1              # [x, y, z] (fixed; default the origin) |
                               # bh1 | bh2 | com | midpoint | module
      pole: z                  # x | y | z | [a, b, c] (fixed; default z) |
                               # spin1 | spin2 | orbital | module
      phi_zero: x              # optional; x | y | z | [a, b, c] (fixed) |
                               # separation | spin1 | spin2 | module

A centre, pole or phi_zero that follows the binary (``bh1``, ``spin1``,
``separation`` ...) is read from the trajectory file, as for a
``kind: trajectory`` plane (``file`` and ``time_offset`` as there, default
``auto``). ``module`` values come from a Python module (``module:``) as for an
analytic plane: ``origin(t, it)``, ``normal(t, it)`` (the pole) and
``u_axis(t, it)`` (phi_zero). Without ``phi_zero`` the coordinate axis least
aligned with the pole is used (x for a pole along z).

The frame is built by ``plane_motion`` - the sphere's centre, pole and phi = 0
direction are a plane's origin, normal and u axis - so a turning pole is made
continuous and checked in the same way, before any field data is read. The one
difference: the sphere's ``e2 = pole x phi_zero`` always completes a
right-handed frame (theta from the pole, phi from phi_zero towards e2).

Output
------
One file per variable, the usual schema:

    data      (n_iter, n_theta, n_phi)            one radius
              (n_iter, n_radii, n_theta, n_phi)   several radii
    x, y      theta, phi in radians (theta cell-centred, phi from 0)
    radius    (n_radii,)

plus the attributes ``surface = sphere``, ``sphere_kind``, ``sphere_radii``.
A fixed sphere stores its frame as the attributes ``sphere_centre``,
``sphere_pole`` and ``sphere_phi_zero``; a moving one as ``(n_iter, 3)``
datasets of the same names, with ``bh1_position``, ``bh2_position`` and
``separation`` when it follows the binary.

All radii in one file cost nothing: each (iteration, radius) sheet is its own
HDF5 chunk, so reading one radius at one time reads exactly what a file per
radius would, and the chunk merge still copies chunks verbatim. Reading gets
cheaper instead - the 3D data are read once per (variable, iteration) for all
radii together, so the per-file metadata that dominates a cold slice is paid
once, and a component that several radii cross is read once.

``refinement_level`` (default yes here) adds the output variable
``refinement_level``: the level that supplied each point, from component
metadata alone. Choose the resolution from it: the grid spacing at radius r is
``r pi / n_theta`` and, at the equator, ``2 pi r / n_phi``, to be compared with
the spacing ``dx`` of that level.
"""

import os

import numpy as np

from resample_common import abort, log, run
from resample_3d_data_postcactus import Postcactus3DBackend
import plane_motion
import sphere_geom
from sphere_geom import SphereSpec, phi_grid, theta_grid

_SPHERE_KEYS = {"radius", "resolution", "centre", "pole", "phi_zero", "file",
                "time_offset", "module"}

# Values that tie part of the frame to the binary (kind trajectory).
_CENTRE_BODIES = ("bh1", "bh2", "com", "midpoint")
_POLE_PRESETS = plane_motion.NORMAL_PRESETS            # orbital, spin1, spin2
_PHI_ZERO_PRESETS = ("separation", "spin1", "spin2")


def _named(value, names):
    return isinstance(value, str) and value.strip().lower() in names


def sphere_kind(sphere):
    """``static``, ``trajectory`` or ``analytic``, from the values given.

    Anything that follows the binary makes the whole frame trajectory-driven;
    otherwise a ``module`` value makes it analytic; otherwise it is fixed.
    """
    centre = sphere.get("centre")
    pole = sphere.get("pole", "z")
    phi_zero = sphere.get("phi_zero")
    if (_named(centre, _CENTRE_BODIES) or _named(pole, _POLE_PRESETS)
            or _named(phi_zero, _PHI_ZERO_PRESETS)):
        return "trajectory"
    if any(_named(v, ("module",)) for v in (centre, pole, phi_zero)):
        return "analytic"
    return "static"


def frame_spec(sphere):
    """The ``plane`` mapping whose frame table is the sphere's frame.

    centre -> origin, pole -> normal, phi_zero -> u_axis.
    """
    kind = sphere_kind(sphere)
    centre = sphere.get("centre", [0.0, 0.0, 0.0])
    pole = sphere.get("pole", "z")
    phi_zero = sphere.get("phi_zero")

    if kind == "static":
        if phi_zero is None:
            # The coordinate axis least aligned with the pole, as a moving
            # frame without phi_zero gets it (plane_motion's stable axis).
            tilt = np.abs(sphere_geom._as_vector(pole, "sphere pole"))
            phi_zero = "xyz"[int(np.argmin(tilt))]
        return {"normal": pole, "origin": centre, "u_axis": phi_zero}

    if kind == "trajectory":
        plane = {"kind": "trajectory", "origin": centre, "normal": pole,
                 "u_axis": phi_zero,
                 "file": sphere.get("file", "auto"),
                 "time_offset": sphere.get("time_offset", "auto")}
        if sphere.get("module") is not None:
            plane["module"] = sphere["module"]
        return plane

    if not _named(pole, ("module",)):
        raise ValueError(
            "a module-driven sphere takes its pole from the module too "
            "(pole: module, with normal(t, it) defined there); return a "
            "constant from it for a fixed pole")
    plane = {"kind": "analytic", "module": sphere.get("module")}
    if not _named(centre, ("module",)):
        plane["origin"] = centre
    if phi_zero is not None and not _named(phi_zero, ("module",)):
        plane["u_axis"] = phi_zero
    return plane


def _radii(value):
    """The ``radius`` value as a list of floats, positive and increasing."""
    radii = np.atleast_1d(np.asarray(value, dtype=float))
    if radii.ndim != 1 or radii.size == 0 or not np.all(np.isfinite(radii)):
        raise ValueError("radius must be a number or a list of numbers (got %r)"
                         % (value,))
    if np.any(radii <= 0.0):
        raise ValueError("radii must be positive (got %r)" % (value,))
    if np.any(np.diff(radii) <= 0.0):
        raise ValueError("list the radii in increasing order, each once "
                         "(got %r)" % (value,))
    return [float(r) for r in radii]


class Postcactus3DSphereBackend(Postcactus3DBackend):
    """postcactus coupling for spheres cut through 3D data.

    Everything but the surface is inherited: the frame table (built from the
    ``plane`` mapping :func:`frame_spec` derives), the component walk, the
    caches and the reads. Here only the grid, the :class:`SphereSpec`, the
    output layout and the attributes differ.
    """

    name = "postcactus_sphere"
    surface = "sphere"

    # -- configuration -----------------------------------------------------

    def validate_config(self, cfg):
        """Parse the ``sphere`` block; then the checks shared with planes."""
        sphere = cfg.get("sphere")
        if not isinstance(sphere, dict):
            abort("a sphere config needs a `sphere` mapping with at least "
                  "`radius` and `resolution` (see config_example_sphere.yaml)")
        sphere = dict(sphere)
        if "center" in sphere:
            if "centre" in sphere:
                abort("sphere: give `centre` or `center`, not both")
            sphere["centre"] = sphere.pop("center")
        unknown = sorted(set(sphere) - _SPHERE_KEYS)
        if unknown:
            abort("unknown sphere key(s): %s (known: %s)"
                  % (", ".join(unknown), ", ".join(sorted(_SPHERE_KEYS))))
        for key in ("radius", "resolution"):
            if sphere.get(key) is None:
                abort("sphere.%s is required" % key)

        try:
            radii = _radii(sphere["radius"])
        except ValueError as exc:
            abort("sphere.radius: %s" % exc)
        resolution = sphere["resolution"]
        if (not isinstance(resolution, (list, tuple)) or len(resolution) != 2
                or any(int(n) != n or n < 1 for n in resolution)):
            abort("sphere.resolution must be [n_theta, n_phi], two positive "
                  "integers (got %r)" % (resolution,))
        n_theta, n_phi = int(resolution[0]), int(resolution[1])

        try:
            kind = sphere_kind(sphere)
            plane = frame_spec(sphere)
        except ValueError as exc:
            abort("sphere: %s" % exc)
        self._check_module(sphere, kind)

        cfg["sphere"] = sphere
        cfg["sphere_kind"] = kind
        cfg["sphere_radii"] = radii
        cfg["plane"] = plane
        cfg["grid"]["resolution"] = [n_theta, n_phi]
        cfg.setdefault("refinement_level", True)

        if kind == "static":
            try:
                spec = self._fixed_spec(cfg)
            except ValueError as exc:
                abort("sphere: %s" % exc)
            log("Sphere (fixed): %s" % spec.describe())
        else:
            log("Sphere (%s): centre %s, pole %s, phi = 0 along %s; resolved "
                "per iteration after the query"
                % (kind, sphere.get("centre", [0.0, 0.0, 0.0]),
                   sphere.get("pole", "z"),
                   sphere.get("phi_zero") or "the axis least aligned with the "
                                             "pole"))
        log("Sphere grid: %d x %d (theta cell-centred, phi periodic), %s"
            % (n_theta, n_phi, "one radius" if len(radii) == 1 else
               "%d radii in one file per variable" % len(radii)))
        for r in radii:
            log("  r = %g: spacing r dtheta = %.4g, r dphi = %.4g at the "
                "equator" % (r, r * np.pi / n_theta, r * 2 * np.pi / n_phi))

        self._validate_3d_common(cfg)

    @staticmethod
    def _check_module(sphere, kind):
        """A `module` value needs a module file defining the right function."""
        wanted = [name for key, name in (("centre", "origin"),
                                         ("pole", "normal"),
                                         ("phi_zero", "u_axis"))
                  if _named(sphere.get(key), ("module",))]
        if not wanted:
            return
        path = sphere.get("module")
        if path is None:
            abort("sphere: a `module` value needs sphere.module, the path to a "
                  "Python file (see plane_module_example.py)")
        if not os.path.isfile(os.path.abspath(os.path.expanduser(str(path)))):
            abort("sphere.module not found: %r (paths are relative to the "
                  "working directory)" % path)
        module = plane_motion.load_plane_module(path)
        missing = [name for name in wanted if not callable(getattr(module, name,
                                                                   None))]
        if missing:
            abort("sphere.module %s does not define %s"
                  % (path, ", ".join("%s(t, it)" % m for m in missing)))

    # -- geometry ----------------------------------------------------------

    def _fixed_spec(self, cfg):
        """The :class:`SphereSpec` of a fixed sphere."""
        plane = self._static_spec(cfg)
        return SphereSpec(plane.origin, cfg["sphere_radii"], plane.u_hat,
                          plane.n_hat)

    def _spec_at(self, cfg, table, it):
        origin, u_hat, _, n_hat = table.frame_at(it)
        return SphereSpec(origin, cfg["sphere_radii"], u_hat, n_hat)

    def _axes(self, cfg):
        """``(theta, phi)`` - the output's ``x``/``y``."""
        if self._axes_cache is None:
            n_theta, n_phi = cfg["grid"]["resolution"]
            self._axes_cache = (theta_grid(n_theta), phi_grid(n_phi))
        return self._axes_cache

    def coordinates(self, cfg):
        theta, phi = self._axes(cfg)
        return {"x": theta, "y": phi,
                "radius": np.asarray(cfg["sphere_radii"], dtype=float)}

    def slice_shape(self, cfg):
        n_theta, n_phi = cfg["grid"]["resolution"]
        n_radii = len(cfg["sphere_radii"])
        if n_radii == 1:
            return (int(n_theta), int(n_phi))
        return (n_radii, int(n_theta), int(n_phi))

    def read_slice(self, plane_index, var, it, cfg):
        """All radii of one iteration in one pass; one radius -> 2D."""
        out = Postcactus3DBackend.read_slice(self, plane_index, var, it, cfg)
        return out[0] if len(cfg["sphere_radii"]) == 1 else out

    # -- output description ------------------------------------------------

    def extra_output_attrs(self, cfg, var=None):
        kind = cfg["sphere_kind"]
        attrs = {
            "source_dims": "xyz",
            "surface": "sphere",
            "plane": "sphere",
            "sphere_kind": kind,
            "sphere_radii": np.asarray(cfg["sphere_radii"], dtype=float),
            "x_axis": "theta",
            "y_axis": "phi",
            "angles": "radians; theta (cell-centred) from sphere_pole, phi "
                      "from sphere_phi_zero towards sphere_pole x "
                      "sphere_phi_zero",
            "layout": ("data[iteration, theta, phi]"
                       if len(cfg["sphere_radii"]) == 1 else
                       "data[iteration, radius, theta, phi]"),
            "outside_value": float(cfg["outside_value"]),
        }
        if kind == "static":
            spec = self._fixed_spec(cfg)
            attrs.update({"sphere_centre": spec.centre, "sphere_pole": spec.e3,
                          "sphere_phi_zero": spec.e1})
        attrs.update(self._variable_attrs(cfg, var))
        return attrs

    _RENAME = {"plane_origin": "sphere_centre", "plane_normal": "sphere_pole",
               "plane_u_axis": "sphere_phi_zero"}

    def per_iteration_data(self, cfg, var, iters):
        """The moving frame as ``sphere_*`` datasets (none for a fixed one).

        The plane's v axis is dropped: the sphere's e2 is pole x phi_zero.
        """
        data = Postcactus3DBackend.per_iteration_data(self, cfg, var, iters)
        return {self._RENAME.get(name, name): values
                for name, values in data.items() if name != "plane_v_axis"}


def main():
    run(Postcactus3DSphereBackend())


if __name__ == "__main__":
    main()
