"""Solve the home-pose trim (whole-body centre of mass over the ankle axis) for a walking-model variant XML.

The stock solver (robot/microcluck/trim_home.py) loads the stock scene. A variant with the leg chain moved (hips raised,
moved back) needs its own trim: hip pitch and ankle are trimmed together so the foot stays flat, until the centre of mass
is the stock distance (0.6 mm) ahead of the ankle axis, then the trunk height is set so the soles just touch the floor.
Prints the trim and trunk height to pass as MICROCLUCK_STAND_TRIM, and the joint limits the pose has to respect.

    uv run python scripts/trim_variant.py src/mjlab_microduck/robot/microcluck/robot_walk_back29_clean.xml
"""

import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, "src")
from mjlab_microduck.robot.microcluck import build_walk_model as b  # noqa: E402

STOCK_COM_OFFSET = 0.0006  # m, whole-body CoM x minus ankle-axis x in Pollen's STAND key


def jq(m, name):
    return m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)]


def lowest_sole_z(m, d):
    zmin = 1e9
    for side in ("left", "right"):
        gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_foot_collision")
        mid = m.geom_dataid[gid]
        v0, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
        world = m.mesh_vert[v0:v0 + n] @ d.geom_xmat[gid].reshape(3, 3).T + d.geom_xpos[gid]
        zmin = min(zmin, world[:, 2].min())
    return zmin


def pose(m, trim, z=0.12):
    d = mujoco.MjData(m)
    d.qpos[3:7] = [1, 0, 0, 0]
    d.qpos[2] = z
    h, k, a = b.HIP_PITCH0 + trim, b.KNEE0, b.ANKLE0 + trim
    for name, v in (("left_hip_roll", -b.HIP_ROLL0), ("right_hip_roll", b.HIP_ROLL0), ("left_hip_pitch", -h),
                    ("right_hip_pitch", h), ("left_knee", -k), ("right_knee", k), ("left_ankle", a), ("right_ankle", -a)):
        d.qpos[jq(m, name)] = v
    mujoco.mj_forward(m, d)
    return d


def com_offset(m, trim):
    d = pose(m, trim)
    return d.subtree_com[1][0] - d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "ankle_left")][0]


def solve(m):
    lo, hi = -0.6, 0.9
    f_lo, f_hi = com_offset(m, lo) - STOCK_COM_OFFSET, com_offset(m, hi) - STOCK_COM_OFFSET
    if f_lo * f_hi >= 0:
        return None
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        f_mid = com_offset(m, mid) - STOCK_COM_OFFSET
        if f_lo * f_mid <= 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return 0.5 * (lo + hi)


for path in sys.argv[1:]:
    m = mujoco.MjModel.from_xml_path(path)
    print(f"== {Path(path).name}: total mass {m.body_subtreemass[1]:.4f} kg")
    print(f"   CoM minus ankle axis at the stock trim ({b.STAND_TRIM}): {1000 * com_offset(m, b.STAND_TRIM):+.1f} mm "
          f"(stock duck {1000 * STOCK_COM_OFFSET:+.1f} mm)")
    trim = solve(m)
    if trim is None:
        print("   no trim in -0.6..0.9 rad puts the centre of mass over the ankle axis")
        continue
    z = 0.12 - lowest_sole_z(m, pose(m, trim)) + 0.0005
    d = pose(m, trim, z)
    print(f"   trim = {trim:+.6f} rad ({np.degrees(trim):+.1f} deg): hip pitch {np.degrees(b.HIP_PITCH0 + trim):.1f} deg, "
          f"ankle {np.degrees(b.ANKLE0 + trim):.1f} deg, trunk z {z:.5f} m, whole-body CoM {np.round(d.subtree_com[1], 4)}")
    for name in ("left_hip_pitch", "left_ankle", "left_knee"):
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        print(f"   {name}: range {np.round(m.jnt_range[j], 3)} rad, at this pose {d.qpos[jq(m, name)]:+.3f}")
