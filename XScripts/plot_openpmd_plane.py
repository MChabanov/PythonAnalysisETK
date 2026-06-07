#!/usr/bin/env python3
"""Minimal standalone plotter for PlanesX openPMD plane output.

The PlanesX (and CarpetX 3D) openPMD writer declares each AMR level's mesh
extent as the *full refined domain* at that level, but only stores the boxes
that actually intersect the plane. Reading the full declared extent therefore
returns a huge, mostly-unwritten array (backend fill/garbage) -- that is why a
naive `mesh[comp][()]` fails to plot. The fix is to load only the written
chunks (ADIOS2 `available_chunks()`; HDF5 reports the whole extent as one chunk,
so we clip the cell-centred fill row/column) and draw each at its own offset.

Usage:
    python plot_openpmd_plane.py <sim>.<tag>.it%08T.<ext> [component] [--save out.png]

Example:
    python plot_openpmd_plane.py sim.xy_z0.it%08T.bp5
    python plot_openpmd_plane.py sim.xy_z0.it%08T.bp5 hydrobasex_rho_lev00 --save rho.png
"""

import argparse
import re

import matplotlib.pyplot as plt
import numpy as np
import openpmd_api as io


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pattern", help="openPMD series pattern, e.g. sim.xy_z0.it%%08T.bp5")
    ap.add_argument("component", nargs="?", default=None,
                    help="mesh component name (default: first found)")
    ap.add_argument("--iteration", type=int, default=None,
                    help="iteration to plot (default: last)")
    ap.add_argument("--save", default=None, help="save figure to this path instead of showing")
    args = ap.parse_args()

    series = io.Series(args.pattern, io.Access.read_only)

    it_keys = list(series.iterations)
    if not it_keys:
        raise SystemExit("no iterations in series")
    it_index = args.iteration if args.iteration is not None else it_keys[-1]
    it = series.iterations[it_index]

    fig, ax = plt.subplots()
    drawn = 0
    pending = []  # (level, x_edges, y_edges, data)

    for mesh_name in it.meshes:
        mesh = it.meshes[mesh_name]
        ggo = list(mesh.grid_global_offset)  # [off_b, off_a]  (b = first/slow axis)
        gsp = list(mesh.grid_spacing)        # [sp_b, sp_a]
        # Level number is encoded in the mesh name as _levNN; used only for
        # draw order (finer on top).
        m = re.search(r"_lev(\d+)", mesh_name)
        level = int(m.group(1)) if m else 0

        comp_names = list(mesh)
        if not comp_names:
            continue
        if args.component is None:
            comp_name = comp_names[0]
        elif args.component in comp_names:
            comp_name = args.component
        else:
            continue  # this mesh/level does not carry the requested component
        rc = mesh[comp_name]

        pos = list(rc.position)   # [pos_b, pos_a]; 0.5 => cell-centred on that axis
        shape = list(rc.shape)    # [N_b, N_a] declared vertex counts (full domain)

        for ch in rc.available_chunks():
            off = [int(v) for v in ch.offset]   # [o_b, o_a]
            ext = [int(v) for v in ch.extent]   # [c_b, c_a]
            # Clip the cell-centred fill index (declared extent is vertex count
            # N; only N-1 cells were written). No-op for ADIOS2 box chunks.
            for d in (0, 1):
                if pos[d] != 0.0:  # cell-centred on this axis
                    valid = shape[d] - 1
                    ext[d] = min(ext[d], max(0, valid - off[d]))
            if ext[0] == 0 or ext[1] == 0:
                continue

            data = rc.load_chunk(off, ext)
            series.flush()
            data = np.asarray(data).reshape(ext[0], ext[1])

            # Cell edges in world coordinates (pcolormesh wants N+1 edges).
            # Centre of index k on axis is ggo + (off + k + pos)*gsp; edges are
            # half a spacing on either side.
            nb, na = ext[0], ext[1]
            y0 = ggo[0] + (off[0] + pos[0]) * gsp[0]
            x0 = ggo[1] + (off[1] + pos[1]) * gsp[1]
            y_edges = y0 + (np.arange(nb + 1) - 0.5) * gsp[0]
            x_edges = x0 + (np.arange(na + 1) - 0.5) * gsp[1]
            pending.append((level, x_edges, y_edges, data))

    if not pending:
        raise SystemExit("no chunks found for the requested component")

    # Shared color scale; draw coarse-to-fine so finer levels sit on top.
    vmin = min(d.min() for _, _, _, d in pending)
    vmax = max(d.max() for _, _, _, d in pending)
    for level, x_edges, y_edges, data in sorted(pending, key=lambda t: t[0]):
        mesh_art = ax.pcolormesh(x_edges, y_edges, data, vmin=vmin, vmax=vmax,
                                 shading="flat")
        drawn += 1

    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("iteration %d (%d chunks)" % (it_index, drawn))
    fig.colorbar(mesh_art, ax=ax)

    del series
    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches="tight")
        print("wrote", args.save)
    else:
        plt.show()


if __name__ == "__main__":
    main()
