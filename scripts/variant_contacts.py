"""Which robot parts overlap each other at the home pose in a stand-up model XML? (Penetrations explain blown-up sweeps.)

    uv run python scripts/variant_contacts.py src/mjlab_microduck/robot/microcluck/robot_standup.xml [...more xml]
"""

import sys

import mujoco
import numpy as np

sys.path.insert(0, "src")
from mjlab_microduck.robot.microcluck import build_walk_model as b  # noqa: E402

for path in sys.argv[1:]:
    spec = mujoco.MjSpec.from_file(path)
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05])
    m = spec.compile()
    d = mujoco.MjData(m)
    d.qpos[2] = 0.30  # well above the floor: only self-overlaps matter
    d.qpos[3:7] = [1, 0, 0, 0]
    d.qpos[7:17] = np.array([float(v) for v in b.stand_qpos(False).split()])
    mujoco.mj_forward(m, d)
    name = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom{g}"
    body = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g])
    pairs = {}
    for c in d.contact[: d.ncon]:
        if c.dist >= 0:
            continue
        key = (f"{name(c.geom1)}[{body(c.geom1)}]", f"{name(c.geom2)}[{body(c.geom2)}]")
        pairs[key] = min(pairs.get(key, 0.0), c.dist)
    print(f"== {path.split('/')[-1]}: {len(pairs)} overlapping geom pairs at the home pose")
    for (g1, g2), dist in sorted(pairs.items(), key=lambda kv: kv[1])[:10]:
        print(f"   {1000 * dist:6.1f} mm  {g1}  x  {g2}")
