#!/usr/bin/env python3
"""Report the toes (and the fused sole, when built): extents, volume and a mass estimate. Needs trimesh.

    uv run --with trimesh python hardware/tools/foot_report.py [build_dir]

Mass = volume x density x infill; density and infill come from params.scad.
"""
import re
import sys
from pathlib import Path

import trimesh

here = Path(__file__).resolve().parent.parent
build = Path(sys.argv[1]) if len(sys.argv) > 1 else here / "build"
params = dict(re.findall(r"^(\w+)\s*=\s*([-\d.]+);", (here / "params.scad").read_text(), re.M))
density, infill = float(params["density_g_cm3"]), float(params["infill_fraction"])

for side in ("left", "right"):
    toes = trimesh.load(build / f"toes_{side}.stl")
    (x0, y0, z0), (x1, y1, z1) = toes.bounds
    mass = toes.volume / 1000.0 * density * infill
    print(f"{side:5s} toes: x {x0:.1f}..{x1:.1f} mm (reach {x1 - float(params['front_wall_x']):.1f} mm past the front wall), "
          f"y {y0:.1f}..{y1:.1f}, z {z0:.1f}..{z1:.1f} mm; {toes.volume / 1000:.2f} cm3, about {mass:.1f} g at {infill:.0%} infill")
    fused = build / f"sole_with_toes_{side}.stl"
    if fused.exists():
        mesh = trimesh.load(fused)
        print(f"      sole+toes: watertight {mesh.is_watertight}, volume {mesh.volume / 1000:.2f} cm3, "
              f"bounds x {mesh.bounds[0][0]:.1f}..{mesh.bounds[1][0]:.1f} mm, z {mesh.bounds[0][2]:.1f}..{mesh.bounds[1][2]:.1f} mm")
