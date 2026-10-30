"""Did a found get-up motion cheat? Replay a saved standup_search result and measure what the simulation does not collide.

The stand-up model only collides the soles, the lower leg (shank), the hip bracket and the torso pieces (bezel, shell hull,
comb, tail, wings). The thighs and the servos on them are visual only, so a search could pass them through the floor or the
shell without penalty. This replays the best candidate on N face-down starts, records the pose every step, and then, in
plain MuJoCo, measures for the visual-only leg parts:

  - the lowest mesh vertex against the floor (negative = below the floor)
  - the distance to the torso collision pieces (negative = overlap; thigh meshes are not convex, so MuJoCo uses their convex
    hull and overlaps are slightly overstated)

    uv run python scripts/standup_validity.py --play ~/microcluck_search/centred_clean.pt --n 16
"""

import argparse

import mujoco
import numpy as np
import torch

import mjlab_microduck.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.torch import configure_torch_backends

ap = argparse.ArgumentParser()
ap.add_argument("--task", default="Mjlab-StandUp-FaceDown-Flat-MicroCluck")
ap.add_argument("--play", required=True)
ap.add_argument("--n", type=int, default=16)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--every", type=int, default=3, help="distance checks every this many steps (the floor check uses every step)")
a = ap.parse_args()

configure_torch_backends()
torch.manual_seed(a.seed)
dev = "cuda:0"
saved = torch.load(a.play, map_location=dev)
K, seconds, asym = saved["keyframes"], saved["seconds"], saved["asym"]
theta = saved["best"].to(dev)
theta10 = theta if asym else torch.cat([theta, -theta], dim=-1)

cfg = load_env_cfg(a.task)
cfg.scene.num_envs = a.n
cfg.seed = a.seed
cfg.episode_length_s = seconds + 2.0
cfg.curriculum.clear()
cfg.events.pop("push_robot", None)
env = ManagerBasedRlEnv(cfg=cfg, device=dev)
robot = env.scene["robot"]
steps = round(seconds / env.step_dt)
last_second = round(1.0 / env.step_dt)

env.reset()
rec_pos, rec_quat, rec_q, up_hist = [], [], [], []
with torch.inference_mode():
    for i in range(steps):
        pos = i / (steps - 1) * (K - 1)
        k0 = min(int(pos), K - 2)
        frac = pos - k0
        act = (theta10[k0] * (1 - frac) + theta10[k0 + 1] * frac).unsqueeze(0).expand(a.n, -1)
        env.step(act)
        rec_pos.append(robot.data.root_link_pos_w.cpu().numpy().copy())
        rec_quat.append(robot.data.root_link_quat_w.cpu().numpy().copy())
        rec_q.append(robot.data.joint_pos.cpu().numpy().copy())
        z = robot.data.root_link_pos_w[:, 2] + saved.get("z_offset", 0.0)
        gz = robot.data.projected_gravity_b[:, 2]
        up_hist.append((z > 0.10) & (gz < -0.94))
up = torch.stack(up_hist[-last_second:]).all(0)
print(f"replay: up for the last second {int(up.sum())}/{a.n}")
env.close()

# ---- plain MuJoCo on the same model, pose by pose
import os  # noqa: E402

xml = os.environ["MICROCLUCK_STANDUP_XML"]
m = mujoco.MjModel.from_xml_path(xml)
d = mujoco.MjData(m)
bname = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g])
gname = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom{g}"
mname = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, m.geom_dataid[g]) or ""

# visual-only leg parts: everything meshed on the hip and thigh chain (the shank and sole already collide)
chain = {"yaw2roll", "bearing_roll", "hip_l", "hip_l_2", "upper_leg_left", "upper_leg_right"}
parts = [g for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and bname(g) in chain
         and m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0]
torso = [g for g in range(m.ngeom) if gname(g) in ("bezel_collision", "shell_collision", "comb_collision", "tail_collision",
                                                   "wing_left_collision", "wing_right_collision")]
shank = {bname(g) for g in range(m.ngeom) if gname(g).endswith("_foot_collision")}
verts = {g: m.mesh_vert[m.mesh_vertadr[m.geom_dataid[g]]:m.mesh_vertadr[m.geom_dataid[g]] + m.mesh_vertnum[m.geom_dataid[g]]]
         for g in parts}
print(f"{len(parts)} visual-only leg geoms checked against the floor, {len(torso)} torso collision pieces")

n_steps = len(rec_pos)
floor_min = np.zeros((n_steps, a.n))  # lowest vertex of any visual-only leg part, per step and robot (m)
floor_who = [[""] * a.n for _ in range(n_steps)]
dist = {}  # (part body, torso geom) -> minimum distance (m)
fromto = np.zeros(6)
for t in range(n_steps):
    for r in range(a.n):
        d.qpos[0:3] = rec_pos[t][r]
        d.qpos[3:7] = rec_quat[t][r]
        d.qpos[7:7 + rec_q[t].shape[1]] = rec_q[t][r]
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        low, who = 9.0, ""
        for g in parts:
            zmin = float((d.geom_xpos[g] + verts[g] @ d.geom_xmat[g].reshape(3, 3).T)[:, 2].min())
            if zmin < low:
                low, who = zmin, f"{bname(g)}:{mname(g)}"
        floor_min[t, r], floor_who[t][r] = low, who
        if t % a.every == 0 and r < 8:
            for g in parts:
                for h in torso:
                    key = (bname(g), gname(h))
                    dd = mujoco.mj_geomDistance(m, d, g, h, 0.05, fromto)
                    if dd < dist.get(key, 9.0):
                        dist[key] = dd

mm = lambda v: f"{1000 * v:7.1f} mm"
print(f"\nlowest visual-only leg vertex over the replay, all robots: {mm(floor_min.min())} "
      f"(at step {np.unravel_index(floor_min.argmin(), floor_min.shape)[0]}, {floor_who[np.unravel_index(floor_min.argmin(), floor_min.shape)[0]][np.unravel_index(floor_min.argmin(), floor_min.shape)[1]]})")
worst = floor_min.min(0)
print(f"per robot, deepest below the floor (mm): median {1000 * np.median(worst):.1f}, 90th percentile {1000 * np.percentile(worst, 10):.1f}, "
      f"robots with a part more than 5 mm below the floor: {int((worst < -0.005).sum())}/{a.n}, more than 15 mm: {int((worst < -0.015).sum())}/{a.n}")
print("\ndeepest overlap with the torso collision pieces (negative = overlap), 8 robots, by thigh-chain body:")
for (b, h), v in sorted(dist.items(), key=lambda kv: kv[1])[:12]:
    print(f"   {mm(v)}  {b} x {h}")
