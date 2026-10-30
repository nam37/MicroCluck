"""Render one clean walking clip of a trained policy as an MP4.

    MUJOCO_GL=egl uv run python scripts/render_walk_video.py Mjlab-Velocity-Flat-MicroCluck \
        --checkpoint ~/microcluck_sim/logs/rsl_rl/microcluck_velocity/<run>/model_4999.pt --out walk.mp4

Runs a single environment with a fixed forward command, no pushes and the robot facing +x at reset, with a
camera that tracks the trunk from a fixed side of the world. If the robot falls before the clip is full, the
take is thrown away and a new one starts from the reset, up to --max-attempts. The clip is written at the
control rate (50 fps). Frames are held in memory until the end: about 2.8 GB for 20 s at 1280x720.
"""

import argparse
from dataclasses import asdict
from pathlib import Path

import mediapy as media
import numpy as np
import torch

import mjlab_microduck.tasks  # noqa: F401  (registers the MicroDuck / MicroCluck tasks)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--speed", type=float, default=0.3, help="forward command in m/s (the policy trained on +-0.4)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--azimuth", type=float, default=130.0, help="degrees; the robot faces +x, 90 is a right-side profile, 130 a front-right three-quarter view")
    ap.add_argument("--elevation", type=float, default=-12.0)
    ap.add_argument("--distance", type=float, default=0.7, help="metres from the tracked trunk")
    ap.add_argument("--lookat-z", type=float, default=0.0, help="metres above the trunk to aim at")
    ap.add_argument("--pushes", action="store_true", help="keep the play configuration's random pushes")
    ap.add_argument("--max-attempts", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    configure_torch_backends()
    env_cfg = load_env_cfg(a.task, play=True)
    env_cfg.scene.num_envs = 1
    env_cfg.seed = a.seed
    env_cfg.episode_length_s = max(env_cfg.episode_length_s, 4 * a.seconds)  # no timeout reset inside a take
    if not a.pushes:
        env_cfg.events.pop("push_robot", None)
    env_cfg.events["reset_base"].params["pose_range"]["yaw"] = (0.0, 0.0)
    for name in [k for k in env_cfg.curriculum if "standing" in k]:
        env_cfg.curriculum.pop(name)  # it would rewrite rel_standing_envs at reset
    twist = env_cfg.commands["twist"]
    twist.ranges.lin_vel_x = (a.speed, a.speed)
    twist.ranges.lin_vel_y = (0.0, 0.0)
    twist.ranges.ang_vel_z = (0.0, 0.0)
    twist.rel_standing_envs = 0.0
    twist.rel_heading_envs = 0.0
    twist.rel_turn_in_place_envs = 0.0
    twist.resampling_time_range = (1e9, 1e9)
    for cmd in env_cfg.commands.values():  # no debug arrows in the picture
        if hasattr(cmd, "debug_vis"):
            cmd.debug_vis = False
    view = env_cfg.viewer
    view.width, view.height = a.width, a.height
    view.distance, view.elevation, view.azimuth = a.distance, a.elevation, a.azimuth
    view.lookat = (0.0, 0.0, a.lookat_z)
    view.max_extra_envs = 0

    agent_cfg = load_rl_cfg(a.task)
    base = ManagerBasedRlEnv(cfg=env_cfg, device=a.device, render_mode="rgb_array")
    env = RslRlVecEnvWrapper(base, clip_actions=agent_cfg.clip_actions)
    runner = (load_runner_cls(a.task) or MjlabOnPolicyRunner)(env, asdict(agent_cfg), device=a.device)
    runner.load(str(Path(a.checkpoint).expanduser()), load_cfg={"actor": True}, strict=True, map_location=a.device)
    policy = runner.get_inference_policy(device=a.device)

    robot = base.scene["robot"]
    steps = round(a.seconds / base.step_dt)
    fps = round(1.0 / base.step_dt)
    print(f"command (vx, vy, wz) = {base.command_manager.get_command('twist')[0].tolist()}, {steps} steps at {fps} fps")

    frames = np.empty((steps, a.height, a.width, 3), np.uint8)
    obs = env.get_observations()
    n, attempt, start_xy = 0, 1, robot.data.root_link_pos_w[0, :2].clone()
    with torch.inference_mode():
        while n < steps:
            obs, _, _, _ = env.step(policy(obs))
            if bool(base.reset_terminated[0]):
                print(f"take {attempt}: fell after {n * base.step_dt:.1f} s, starting a new take")
                attempt += 1
                if attempt > a.max_attempts:
                    raise SystemExit(f"no clean take of {a.seconds:g} s in {a.max_attempts} attempts")
                n = 0
                start_xy = robot.data.root_link_pos_w[0, :2].clone()
                continue
            frames[n] = base.render()
            n += 1
    walked = (robot.data.root_link_pos_w[0, :2] - start_xy).norm().item()
    print(f"clean take {attempt}: walked {walked:.2f} m in {a.seconds:g} s ({walked / a.seconds:.2f} m/s, commanded {a.speed:g})")

    out = Path(a.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    media.write_video(str(out), frames, fps=fps)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {a.width}x{a.height}, {steps} frames)")
    env.close()


if __name__ == "__main__":
    main()
