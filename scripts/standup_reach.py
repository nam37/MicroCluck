"""Static reach: how far can a constant leg pose raise MicroCluck from lying face-down?

Sweeps a grid of mirrored leg poses (hip pitch, knee, ankle offsets from the home pose; yaw and roll at home) with
thousands of face-down robots at once, holds each pose for a few seconds and reports how upright and how high the trunk
ends up. It shows the geometric limit of what the legs can do alone, without the dynamics a search has to find.
The best pose is saved in the format of standup_search.py so it can be rendered:

    uv run python scripts/standup_reach.py --out reach_best.pt
    uv run python scripts/standup_search.py --play reach_best.pt --n 1 --video reach_best.mp4
"""

import argparse

import numpy as np
import torch

import mjlab_microduck.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.torch import configure_torch_backends

ap = argparse.ArgumentParser()
ap.add_argument("--task", default="Mjlab-StandUp-FaceDown-Flat-MicroCluck")
ap.add_argument("--per-axis", type=int, default=16)
ap.add_argument("--seconds", type=float, default=3.0)
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--out", default="reach_best.pt")
a = ap.parse_args()

configure_torch_backends()
n = a.per_axis
hip = torch.linspace(-1.2, 1.6, n)
knee = torch.linspace(-1.6, 1.6, n)
ankle = torch.linspace(-1.2, 1.2, n)
grid = torch.cartesian_prod(hip, knee, ankle).to(a.device)
n_env = len(grid)

cfg = load_env_cfg(a.task)
cfg.scene.num_envs = n_env
cfg.seed = a.seed
cfg.episode_length_s = a.seconds + 2.0
cfg.curriculum.clear()
cfg.events.pop("push_robot", None)
env = ManagerBasedRlEnv(cfg=cfg, device=a.device)
robot = env.scene["robot"]
print("joints:", list(robot.joint_names))

# 0-4 left leg (hip yaw, hip roll, hip pitch, knee, ankle), 5-9 right leg; the right is the left mirrored (sign -1)
act = torch.zeros(n_env, 10, device=a.device)
act[:, 2], act[:, 3], act[:, 4] = grid[:, 0], grid[:, 1], grid[:, 2]
act[:, 7], act[:, 8], act[:, 9] = -grid[:, 0], -grid[:, 1], -grid[:, 2]

steps = round(a.seconds / env.step_dt)
with torch.inference_mode():
    env.reset()
    for _ in range(steps):
        env.step(act)
    z = robot.data.root_link_pos_w[:, 2] * 1000
    gz = robot.data.projected_gravity_b[:, 2]
    tilt = torch.rad2deg(torch.arccos((-gz).clamp(0, 1)))  # 90 = lying, 0 = upright
    elev = -torch.rad2deg(torch.arcsin(gz.clamp(-1, 1)))  # trunk head-end elevation: +90 upright, 0 horizontal, negative = head end down
    # Extreme poses can blow the simulation up (a trunk below the floor, or flung far up). Those are not results.
    # The valid band is for the stock layout, whose trunk frame rests at about 36 mm lying down; variants with the legs
    # mounted higher stand lower, but lying down the trunk is still above the floor.
    valid = torch.isfinite(z) & torch.isfinite(gz) & (z > 25) & (z < 160)
    tilt = torch.where(valid, tilt, torch.full_like(tilt, 180.0))
    elev = torch.where(valid, elev, torch.full_like(elev, -180.0))
    z = torch.where(valid, z, torch.zeros_like(z))

print(f"{n_env} static poses held for {a.seconds:g} s from face-down (rest: trunk 36 mm, tilt 90 deg); "
      f"{int((~valid).sum())} blew up (trunk below 25 mm or above 160 mm) and are excluded below")
print(f"tilt under 85 deg: {int((tilt < 85).sum())}, under 70: {int((tilt < 70).sum())}, under 50: {int((tilt < 50).sum())}, "
      f"under 20: {int((tilt < 20).sum())};  trunk above 60 mm: {int((z > 60).sum())}, above 90: {int((z > 90).sum())}")
print(f"head end raised above horizontal by 10+ deg: {int((elev > 10).sum())}; trunk above 90 mm and within 15 deg of horizontal "
      f"(a flat plank held by the legs): {int(((z > 90) & (elev.abs() < 15)).sum())}")
for title, order in (("most upright", tilt.argsort()), ("highest trunk", z.argsort(descending=True))):
    print(f"-- {title}:  hip pitch / knee / ankle offsets (rad)  ->  tilt deg, head-end elevation deg (negative = head down), trunk mm")
    for i in order[:6].tolist():
        print(f"   {grid[i, 0]:+.2f} {grid[i, 1]:+.2f} {grid[i, 2]:+.2f}  ->  {float(tilt[i]):5.1f} deg, {float(elev[i]):+6.1f} deg, {float(z[i]):5.1f} mm")
best = int(tilt.argsort()[0])
theta = torch.zeros(2, 5)  # two identical keyframes of (yaw, roll, hip pitch, knee, ankle), the mirrored-leg format
theta[:, 2:5] = grid[best].cpu()
torch.save({"best": theta, "mean": theta, "keyframes": 2, "seconds": a.seconds, "asym": False, "best_score": 0.0}, a.out)
print("saved", a.out)
env.close()
