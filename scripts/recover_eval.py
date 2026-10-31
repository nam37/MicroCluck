"""How often does the recovery get the robot up, with and without the learned correction, under named physics?

Runs n robots through the whole 11 s recovery (roll onto the front, then stand-up) from the side, the back, or half of each, on the
Nominal task (the stock stand-up randomization) plus whatever --stress-* options name, and counts the robots up as standup_search.py
counts them: trunk higher than 0.10 m and tilted under 20 degrees for the whole last second of the 11 s. Without --checkpoint the
policy's output is zero, which plays the open-loop motions (the same ones as standup_search.py --play) and is the baseline.

  uv run python scripts/recover_eval.py --checkpoint logs/rsl_rl/microcluck_recover/<run>/model_1500.pt --start side --stress-com 8 0 0
"""

import argparse
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

import mjlab_microduck.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab_microduck.tasks.microcluck_stress import add_stress_args, apply_stress, stress_from_args

ap = argparse.ArgumentParser()
ap.add_argument("--task", default="Mjlab-Recover-Nominal-MicroCluck")
ap.add_argument("--checkpoint", default=None, help="a trained policy; without it the correction is zero (the open-loop motions)")
ap.add_argument("--start", choices=["side", "back", "mix"], default="mix")
ap.add_argument("--side-roll", type=float, nargs=2, default=[90.0, 90.0], metavar=("LO", "HI"), help="degrees from flat on the back, side starts")
ap.add_argument("--back-roll-max", type=float, default=30.0, help="degrees from flat on the back, to a random side, back starts")
ap.add_argument("--n", type=int, default=256)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--video", default=None, help="write an mp4 of env 0")
ap.add_argument("--distance", type=float, default=0.5)
ap.add_argument("--label", default="")
add_stress_args(ap)
a = ap.parse_args()

configure_torch_backends()
cfg = load_env_cfg(a.task)
cfg.scene.num_envs = a.n
cfg.seed = a.seed
cfg.events["set_ground_state"].params.update(
    side_prob={"side": 1.0, "back": 0.0, "mix": 0.5}[a.start], side_roll=(math.radians(a.side_roll[0]), math.radians(a.side_roll[1])),
    back_roll_max=math.radians(a.back_roll_max))
apply_stress(cfg, stress_from_args(a))
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

term = base.action_manager.get_term("joint_pos")
robot = base.scene["robot"]
motion_steps, last_second, hold_steps = term.n_steps, round(1.0 / base.step_dt), round((cfg.episode_length_s - 1.0) / base.step_dt) - term.n_steps
obs = env.get_observations()
chain = term.chain.clone()  # fixed at the reset, before the first step
held, tilt_end = [], None
frames = []
with torch.inference_mode():
    for i in range(motion_steps + hold_steps):
        act = policy(obs) if policy else torch.zeros(a.n, base.action_manager.total_action_dim, device=a.device)
        obs, _, _, _ = env.step(act)
        z_raw, g = robot.data.root_link_pos_w[:, 2], robot.data.projected_gravity_b
        ok = torch.isfinite(z_raw) & torch.isfinite(g).all(dim=-1) & (z_raw > 0.022) & (z_raw < 0.16)
        held.append(ok & (z_raw + term.z_offset > 0.10) & (g[:, 2] < -0.94))
        if a.video:
            frames.append(base.render())
held = torch.stack(held)
up = held[motion_steps - last_second:motion_steps].all(0)  # as standup_search.py: the last second of the 11 s
up_later = held[-last_second:].all(0)  # and still up after the pose has been held for one more second
ever = held[term.roll_steps:].any(0)
parts = []
for c, (name, _, _) in enumerate(__import__("mjlab_microduck.tasks.microcluck_recover_env_cfg", fromlist=["CHAINS"]).CHAINS):
    m = chain == c
    if m.any():
        parts.append(f"{name} {int((up & m).sum())}/{int(m.sum())}")
print(f"recover {a.label or a.start}: {Path(a.checkpoint).parent.name + '/' + Path(a.checkpoint).name if a.checkpoint else 'open loop'}: "
      f"up for the last second {int(up.sum())}/{a.n} ({100 * up.float().mean():.0f}%) [{', '.join(parts)}]; "
      f"still up one second later {int(up_later.sum())}/{a.n}; up at some point {int(ever.sum())}/{a.n}", flush=True)
if a.video and frames:
    import mediapy as media

    media.write_video(a.video, np.stack(frames), fps=round(1 / base.step_dt))
    print("wrote", a.video)
env.close()
