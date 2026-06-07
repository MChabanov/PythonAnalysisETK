#!/usr/bin/env python3
"""Inspect actual written data: chunks, extents, and point counts (per-chunk safe).

Shows the actual data structure without loading unwritten padding from declared extents.
"""

import sys
import os
import numpy as np
import openpmd_common as opc


def inspect_iteration(filepath, variable_pattern=""):
    """Inspect all chunks in a single iteration."""
    try:
        import openpmd_api as io
    except ImportError:
        print("ERROR: openpmd_api not available")
        return

    try:
        series = io.Series(filepath, io.Access.read_only)
    except Exception as e:
        print(f"ERROR opening {filepath}: {e}")
        return

    iterations = list(series.iterations)
    if not iterations:
        print("No iterations found")
        series.close()
        return

    it = iterations[-1]  # Last iteration
    itobj = series.iterations[it]

    print(f"{'=' * 100}")
    print(f"Iteration {it}")
    print(f"{'=' * 100}\n")

    entries = {}

    for mesh_name in sorted(itobj.meshes):
        mesh_info = opc.parse_amr_mesh_name(mesh_name)
        if mesh_info is None:
            continue

        mesh = itobj.meshes[mesh_name]
        comp_names = list(mesh)
        if not comp_names:
            continue

        for comp_name in comp_names:
            if not opc.mesh_matches_variable(mesh_name, comp_name, variable_pattern, mesh_info):
                continue

            rc = mesh[comp_name]

            # Get metadata
            sp = np.array(mesh.get_attribute("gridSpacing"))
            off = np.array(mesh.get_attribute("gridGlobalOffset"))
            pos = np.array(rc.position)
            shape = np.array([int(s) for s in rc.shape])

            declared_points = np.prod(shape)

            # Iterate chunks
            chunks_info = []
            total_actual_points = 0

            for ch in rc.available_chunks():
                ch_off = np.array([int(v) for v in ch.offset])
                ch_ext = np.array([int(v) for v in ch.extent])

                # Clip cell-centred fill
                ext_clipped = ch_ext.copy()
                for d in range(len(ch_ext)):
                    if pos[d] != 0.0:
                        valid = shape[d] - 1
                        ext_clipped[d] = min(ch_ext[d], max(0, valid - ch_off[d]))

                if any(v == 0 for v in ext_clipped):
                    continue

                actual_points = np.prod(ext_clipped)
                total_actual_points += actual_points

                # World coordinates
                world_off = off + ch_off * sp
                world_extent = ch_ext * sp

                chunks_info.append({
                    'offset': ch_off,
                    'extent': ch_ext,
                    'extent_clipped': ext_clipped,
                    'points': actual_points,
                    'world_offset': world_off,
                    'world_extent': world_extent,
                })

            label = opc.component_label(mesh_info, comp_name, len(comp_names))
            entries[(label, mesh_info["level"], mesh_info["patch"])] = {
                'mesh_name': mesh_name,
                'component': comp_name,
                'declared_points': declared_points,
                'declared_shape': tuple(shape),
                'chunks': chunks_info,
                'mesh_info': mesh_info,
            }

    # Print results
    if not entries:
        print(f"No data found matching '{variable_pattern}'")
        series.close()
        return

    for label, level, patch in sorted(entries):
        info = entries[(label, level, patch)]
        mesh_name = info['mesh_name']
        comp_name = info['component']
        declared = info['declared_points']
        declared_shape = info['declared_shape']
        chunks = info['chunks']

        actual = sum(c['points'] for c in chunks)
        pct_sparse = 100 * (1 - actual / declared) if declared > 0 else 0

        print(f"{label} level {level} patch {patch}: {mesh_name}/{comp_name}")
        print(f"  Declared extent: {declared_shape} → {declared:,} points")
        print(f"  Actual written: {actual:,} points ({100*actual/declared:.2f}%)")
        print(f"  Sparsity: {pct_sparse:.1f}% unwritten")
        print(f"  Number of chunks: {len(chunks)}")

        if len(chunks) <= 5:
            for i, ch in enumerate(chunks):
                print(f"    Chunk {i}: offset={ch['offset']}, extent={ch['extent']}, "
                      f"points={ch['points']:,}")
        else:
            # Show first and last
            for i, ch in enumerate(chunks[:2]):
                print(f"    Chunk {i}: offset={ch['offset']}, extent={ch['extent']}, "
                      f"points={ch['points']:,}")
            print(f"    ... ({len(chunks)-4} more chunks) ...")
            for i, ch in enumerate(chunks[-2:], len(chunks)-2):
                print(f"    Chunk {i}: offset={ch['offset']}, extent={ch['extent']}, "
                      f"points={ch['points']:,}")
        print()

    series.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_chunks.py <filepath> [variable_pattern]")
        print("Example: python inspect_chunks.py sim.xy_z0.it%08T.bp5 rho")
        sys.exit(1)

    filepath = sys.argv[1]
    variable_pattern = sys.argv[2] if len(sys.argv) > 2 else ""

    inspect_iteration(filepath, variable_pattern)
