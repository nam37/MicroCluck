#!/usr/bin/env python3
"""Solve MicroCluck's home-pose trim (CoM over the ankle axis) on the CPU.

Pollen's STAND key puts the whole-body CoM over the ankle axis (stock offset +0.6 mm). MicroCluck's
mass distribution differs (no head, screen torso, tail), so hip pitch and ankle are trimmed together
(the foot stays flat) until the CoM offset matches the stock one, then the trunk height is set so the
soles just touch the floor. Copy the printed values into build_walk_model.py and rebuild.

This is geometry only. Whether the pose really holds is checked in the mjlab environment with the
BAM actuator model (a plain XML position actuator cannot hold even Pollen's stock stand, so a CPU
stand test with it is meaningless).

    uv run python src/mjlab_microduck/robot/microcluck/trim_home.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_walk_model as b  # noqa: E402

OUR_SCENE = HERE / "scene_walk.xml"
STOCK_COM_OFFSET = 0.0006  # m, whole-body CoM x minus ankle-axis x in Pollen's STAND key


def jq(m, name):
    return m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)]


def lowest_sole_z(m, d):
    zmin = 1e9
    for side in ("left", "right"):
        gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_foot_collision")
        mid = m.geom_dataid[gid]
        v0, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
        verts = m.mesh_vert[v0:v0 + n]
        world = verts @ d.geom_xmat[gid].reshape(3, 3).T + d.geom_xpos[gid]
        zmin = min(zmin, world[:, 2].min())
    return zmin


def pose(m, trim, z=0.12):
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "STAND"))
    h, a = b.HIP_PITCH0 + trim, b.ANKLE0 + trim
    d.qpos[jq(m, "left_hip_pitch")], d.qpos[jq(m, "left_ankle")] = -h, a
    d.qpos[jq(m, "right_hip_pitch")], d.qpos[jq(m, "right_ankle")] = h, -a
    d.qpos[2] = z
    mujoco.mj_forward(m, d)
    return d


def com_offset(m, trim):
    d = pose(m, trim)
    ank = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "ankle_left")
    return d.subtree_com[1][0] - d.xpos[ank][0]


def solve_trim(m):
    lo, hi = -0.30, 0.30
    f_lo, f_hi = com_offset(m, lo) - STOCK_COM_OFFSET, com_offset(m, hi) - STOCK_COM_OFFSET
    assert f_lo * f_hi < 0, f"no sign change over trim range: {f_lo:.4f} {f_hi:.4f}"
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        f_mid = com_offset(m, mid) - STOCK_COM_OFFSET
        if f_lo * f_mid <= 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return 0.5 * (lo + hi)


def main():
    m = mujoco.MjModel.from_xml_path(str(OUR_SCENE))
    print(f"MicroCluck total mass {m.body_subtreemass[1]:.4f} kg, nq={m.nq}, nu={m.nu}")
    d0 = pose(m, 0.0)
    print(f"CoM minus ankle axis at the stock home angles: {com_offset(m, 0.0) * 1000:+.1f} mm "
          f"(stock duck: {STOCK_COM_OFFSET * 1000:+.1f} mm); whole-body CoM height {d0.subtree_com[1][2]:.4f} m")
    trim = solve_trim(m)
    z = 0.12 - lowest_sole_z(m, pose(m, trim)) + 0.0005
    d = pose(m, trim, z)
    print(f"trim = {trim:+.6f} rad ({np.degrees(trim):+.2f} deg): hip pitch {np.degrees(b.HIP_PITCH0 + trim):.2f} deg, "
          f"ankle {np.degrees(b.ANKLE0 + trim):.2f} deg, trunk z {z:.5f} m")
    print(f"whole-body CoM at the trimmed pose: {np.round(d.subtree_com[1], 4)}")
    print(f"current build constants: STAND_TRIM={b.STAND_TRIM}, STAND_TRUNK_Z={b.STAND_TRUNK_Z}")


if __name__ == "__main__":
    main()
