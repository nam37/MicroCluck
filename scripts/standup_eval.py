"""How often does a policy get MicroCluck up from lying on the floor?

Spawns n robots in one starting pose (face-down, face-up, or standing), runs the policy for a few seconds with the
training randomization on but no pushes, and counts the ones standing at the end: trunk higher than 0.10 m and tilted
less than 20 degrees from vertical for the whole last second. Also reports the time of the first upright moment.
Without --checkpoint the actions are zero (the legs hold the home pose), which is the "do nothing" baseline.
"""

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

import mjlab_microduck.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

ap = argparse.ArgumentParser()
ap.add_argument("--task", default="Mjlab-StandUp-Flat-MicroCluck")
ap.add_argument("--checkpoint", default=None)
ap.add_argument("--start", choices=["face_down", "face_up", "sitting", "standing"], default="face_down")
ap.add_argument("--roll-max", type=float, default=0.0, help="face-up only: random partial roll about the long axis, degrees")
ap.add_argument("--n", type=int, default=64)
ap.add_argument("--seconds", type=float, default=6.0)
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--video", default=None, help="write an mp4 of env 0")
ap.add_argument("--distance", type=float, default=0.7, help="video camera distance, m")
a = ap.parse_args()

configure_torch_backends()
cfg = load_env_cfg(a.task)
cfg.scene.num_envs = a.n
cfg.seed = a.seed
cfg.episode_length_s = a.seconds + 1.0
cfg.curriculum.clear()
cfg.events.pop("push_robot", None)
mix = {"standing_prob": 0.0, "sitting_prob": 0.0, "face_down_prob": 0.0, "face_up_prob": 0.0}
mix[{"face_down": "face_down_prob", "face_up": "face_up_prob", "sitting": "sitting_prob", "standing": "standing_prob"}[a.start]] = 1.0
cfg.events["set_ground_state"].params.update(mix)
cfg.events["set_ground_state"].params["face_up_roll_max"] = float(np.radians(a.roll_max))
if a.video:
    cfg.viewer.width, cfg.viewer.height = 1280, 720
    cfg.viewer.distance, cfg.viewer.elevation, cfg.viewer.azimuth = a.distance, -20.0, 130.0

base = ManagerBasedRlEnv(cfg=cfg, device=a.device, render_mode="rgb_array" if a.video else None)
env = RslRlVecEnvWrapper(base, clip_actions=load_rl_cfg(a.task).clip_actions)
policy = None
if a.checkpoint:
    agent_cfg = load_rl_cfg(a.task)
    runner = (load_runner_cls(a.task) or MjlabOnPolicyRunner)(env, asdict(agent_cfg), device=a.device)
    runner.load(str(Path(a.checkpoint).expanduser()), load_cfg={"actor": True}, strict=True, map_location=a.device)
    policy = runner.get_inference_policy(device=a.device)

robot = base.scene["robot"]
steps = round(a.seconds / base.step_dt)
last_second = round(1.0 / base.step_dt)
obs = env.get_observations()
upright_hist, first_up = [], torch.full((a.n,), float("inf"), device=a.device)
frames = []
with torch.inference_mode():
    for i in range(steps):
        act = policy(obs) if policy else torch.zeros(a.n, base.action_manager.total_action_dim, device=a.device)
        obs, _, _, _ = env.step(act)
        z = robot.data.root_link_pos_w[:, 2]
        gz = robot.data.projected_gravity_b[:, 2]
        up = (z > 0.10) & (gz < -0.94)
        upright_hist.append(up)
        first_up = torch.where(up & torch.isinf(first_up), torch.full_like(first_up, i * base.step_dt), first_up)
        if a.video:
            frames.append(base.render())
up = torch.stack(upright_hist)
stood = up[-last_second:].all(0)
z = robot.data.root_link_pos_w[:, 2] * 1000
gx = robot.data.projected_gravity_b[:, 0]
print(f"{a.task} {Path(a.checkpoint).name if a.checkpoint else 'zero actions'}: start {a.start}"
      f"{f' (roll up to {a.roll_max:g} deg)' if a.start == 'face_up' else ''}, {a.n} robots, {a.seconds:g} s, no pushes")
print(f"standing for the last second: {int(stood.sum())}/{a.n} ({100 * stood.float().mean():.0f}%)")
reached = ~torch.isinf(first_up)
print(f"upright at some point: {int(reached.sum())}/{a.n}; median time to first upright "
      f"{float(first_up[reached].median()) if reached.any() else float('nan'):.2f} s")
print(f"final trunk height mm: median {float(z.median()):.0f}, min {float(z.min()):.0f}, max {float(z.max()):.0f}; "
      f"still face-down (gravity x > 0.7): {int((gx > 0.7).sum())}, still face-up (x < -0.7): {int((gx < -0.7).sum())}")
gy = robot.data.projected_gravity_b[:, 1]
gz_end = robot.data.projected_gravity_b[:, 2]
screen_tilt = torch.rad2deg(torch.arccos((-gx).clamp(-1, 1)))  # angle between the screen normal and straight up (0 = screen up)
print("screen normal away from straight up (deg) at the end, 10th / 50th / 90th percentile: "
      + " ".join(f"{q:.0f}" for q in torch.quantile(screen_tilt, torch.tensor([0.1, 0.5, 0.9], device=a.device)).tolist()))
lying = gz_end.abs() < 0.5
roll_end = torch.rad2deg(torch.atan2(gy.abs(), -gx))  # 0 = flat on the back, 90 = on its side (read it for robots lying down)
print(f"resting on a side (|gravity y| > 0.7): {int((gy.abs() > 0.7).sum())}; of the {int(lying.sum())} lying down, roll away from flat "
      f"on the back (deg): " + (" ".join(f"{q:.0f}" for q in torch.quantile(roll_end[lying], torch.tensor([0.1, 0.5, 0.9], device=a.device)).tolist())
                                if lying.any() else "n/a") + " (10th, 50th, 90th percentile)")
if a.video and frames:
    import mediapy as media

    media.write_video(a.video, np.stack(frames), fps=round(1 / base.step_dt))
    print("wrote", a.video)
env.close()
