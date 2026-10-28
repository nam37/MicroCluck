#!/usr/bin/env python3
"""Report the shell: size, volume and an estimated mass (needs trimesh).

    uv run --with trimesh python hardware/tools/shell_report.py [build_dir]

Mass is the full volume of the printed walls times the PETG density from torso.scad (an upper bound: a real
print with a sparse infill is lighter only in thick sections, and these walls are thin).
"""
import re
import sys
from pathlib import Path

import trimesh

here = Path(__file__).resolve().parent.parent
build = Path(sys.argv[1]) if len(sys.argv) > 1 else here / "build"
density = float(re.search(r"^density_g_cm3\s*=\s*([\d.]+);", (here / "torso.scad").read_text(), re.M).group(1))

shell = trimesh.load(build / "shell.stl")
env = trimesh.load(build / "shell_envelope.stl")
(x0, y0, z0), (x1, y1, z1) = shell.bounds
print(f"shell: x {x0:.1f}..{x1:.1f} ({x1 - x0:.1f} long), y {y0:.1f}..{y1:.1f} ({y1 - y0:.1f} wide), "
      f"z {z0:.1f}..{z1:.1f} ({z1 - z0:.1f} tall) mm")
print(f"       watertight {shell.is_watertight}, wall volume {shell.volume / 1000:.2f} cm3 "
      f"-> about {shell.volume / 1000 * density:.0f} g at {density} g/cm3; outer envelope {env.volume / 1000:.1f} cm3")
print(f"       center of the shell mass (x, y, z) mm: {[round(float(v), 1) for v in shell.center_mass]}")
