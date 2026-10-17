#!/usr/bin/env python3
"""Read and use the output of the sphere pipeline (numpy + h5py only).

    from sphere_tools import load_sphere, unit_vectors, area_weights

    bx = load_sphere("Bx__run_sphere_BH1.h5")
    bx["data"]            # (n_iter, n_radii, n_theta, n_phi), always 4D
    bx["theta"], bx["phi"], bx["radius"], bx["times"]
    bx["pole"][i], bx["phi_zero"][i], bx["centre"][i]   # frame at iteration i

    r_hat, th_hat, ph_hat = unit_vectors(bx["theta"], bx["phi"],
                                         bx["pole"][i], bx["phi_zero"][i])
    B_r = r_hat[0] * Bx + r_hat[1] * By + r_hat[2] * Bz      # (n_theta, n_phi)
    w = area_weights(len(bx["theta"]), len(bx["phi"]), r)     # sums to 4 pi r^2

The unit vectors are flat-space ones in simulation coordinates. For the flux
of a densitized vector, e.g. the magnetic flux ``sum(w * sqrt_gamma * B_r)``,
that is exact: ``sqrt(gamma) B^i`` is divergence-free with the coordinate
divergence, so its flux through the coordinate sphere needs no other metric
factor.
"""

import numpy as np
import h5py

from sphere_geom import area_weights, sphere_frame  # noqa: F401 (re-export)


def load_sphere(path, iterations=None):
    """Load a sphere output file as a dict of numpy arrays.

    ``data`` is always ``(n_iter, n_radii, n_theta, n_phi)``, also for a file
    with one radius. ``iterations`` (a list) restricts what is read. The frame
    (``centre``, ``pole``, ``phi_zero``) is ``(n_iter, 3)`` whether the file
    stores it per iteration or, for a fixed sphere, as attributes.
    """
    with h5py.File(path, "r") as h5:
        if h5.attrs.get("surface") != "sphere":
            raise ValueError("%s is not output of the sphere pipeline" % path)
        its = h5["iterations"][:]
        if iterations is None:
            rows = np.arange(its.size)
            data = h5["data"][:]
        else:
            index = {int(it): i for i, it in enumerate(its)}
            missing = [it for it in iterations if int(it) not in index]
            if missing:
                raise KeyError("iteration(s) %s not in %s" % (missing, path))
            rows = np.array([index[int(it)] for it in iterations], dtype=int)
            # h5py reads increasing indices only; read sorted, then reorder.
            order = np.argsort(rows)
            data = np.empty((rows.size,) + h5["data"].shape[1:])
            data[order] = h5["data"][rows[order]] if rows.size else data
        if data.ndim == 3:
            data = data[:, None]
        out = {
            "data": data,
            "iterations": its[rows],
            "times": h5["times"][:][rows],
            "theta": h5["x"][:],
            "phi": h5["y"][:],
            "radius": h5["radius"][:],
            "attrs": dict(h5.attrs),
        }
        for name, key in (("centre", "sphere_centre"), ("pole", "sphere_pole"),
                          ("phi_zero", "sphere_phi_zero")):
            if key in h5:
                out[name] = h5[key][:][rows]
            else:
                out[name] = np.repeat(np.asarray(h5.attrs[key], float)[None],
                                      rows.size, axis=0)
        for name in ("bh1_position", "bh2_position", "separation"):
            if name in h5:
                out[name] = h5[name][:][rows]
    return out


def unit_vectors(theta, phi, pole, phi_zero):
    """``(r_hat, theta_hat, phi_hat)``, each ``(3, n_theta, n_phi)``.

    Flat-space unit vectors in simulation coordinates for the sphere's frame:
    theta from ``pole``, phi from ``phi_zero`` towards ``pole x phi_zero``.
    """
    e1, e2, e3 = sphere_frame(pole, phi_zero)
    st = np.sin(theta)[:, None]
    ct = np.cos(theta)[:, None]
    sp = np.sin(phi)[None, :]
    cp = np.cos(phi)[None, :]

    def combine(a, b, c):
        return np.stack([a * e1[d] + b * e2[d] + c * e3[d] for d in range(3)])

    one = np.ones_like(st * cp)
    r_hat = combine(st * cp, st * sp, ct * one)
    theta_hat = combine(ct * cp, ct * sp, -st * one)
    phi_hat = combine(-sp * one, cp * one, 0.0 * one)
    return r_hat, theta_hat, phi_hat


def sqrt_gamma(gxx, gxy, gxz, gyy, gyz, gzz):
    """``sqrt(det(gamma_ij))`` from the six resampled metric components."""
    det = (gxx * (gyy * gzz - gyz * gyz) - gxy * (gxy * gzz - gyz * gxz)
           + gxz * (gxy * gyz - gyy * gxz))
    return np.sqrt(det)
