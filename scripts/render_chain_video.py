"""Render the whole recovery chain as one video: lying on a side, rolling onto the front, standing up, walking.

Not one policy. The first three stages replay two open-loop motions found by scripts/standup_search.py (the roll onto the
front, then the stand-up searched with that roll played first, both as their mean motion), in the stand-up task's
environment. When the robot is standing, the simulator state is copied into the walking task's environment (same model,
same scene) and the trained walking policy takes over with a forward command. Run both tasks on the same XML:

    MICROCLUCK_STANDUP_XML=<xml> MICROCLUCK_WALK_XML=<xml> MICROCLUCK_LEG_DZ=0.07 MICROCLUCK_STAND_TRIM=0.258849 \
    MUJOCO_GL=egl uv run python scripts/render_chain_video.py --prefix roll.pt --standup chain.pt \
        --walk-checkpoint model_999.pt --out chain.mp4

The captions come from the measured orientation of the robot, not from timers. A take that does not end up standing, or
that falls while walking, is thrown away and a new one starts (the reset randomization differs between takes).
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

STAND_TASK = "Mjlab-StandUp-FaceUp-Flat-MicroCluck"
WALK_TASK = "Mjlab-Velocity-Flat-MicroCluck"
LABELS = ["Lying on its side", "Rolling onto its front", "Face down", "Getting up", "Standing", "Walking"]
SUBTITLES = ["searched open-loop motion"] * 5 + ["trained walking policy, 0.3 m/s forward command"]


def load_motion(path: str, dev: str):
    """The mean motion of a saved search as (1, K, 10) joint-target offsets, and its length in seconds."""
    d = torch.load(Path(path).expanduser(), map_location=dev)
    theta = d["mean"].to(dev)
    if not d["asym"]:
        theta = torch.cat([theta, -theta], dim=-1)  # mirrored legs
    return theta.unsqueeze(0), float(d["seconds"])


def caption(frame: np.ndarray, stage: int, t: float) -> np.ndarray:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return frame
    im = Image.fromarray(frame)
    d = ImageDraw.Draw(im)
    try:
        big = ImageFont.truetype("DejaVuSans-Bold.ttf", max(18, frame.shape[0] // 18))
        small = ImageFont.truetype("DejaVuSans.ttf", max(12, frame.shape[0] // 32))
    except OSError:
        big = small = ImageFont.load_default()
    pad = frame.shape[0] // 40
    d.rectangle([0, 0, frame.shape[1], 3 * pad + big.size + small.size], fill=(0, 0, 0))
    d.text((pad, pad), f"{stage + 1}  {LABELS[stage]}", font=big, fill=(255, 255, 255))
    d.text((pad, 2 * pad + big.size), f"{SUBTITLES[stage]}   |   simulation, t = {t:4.1f} s", font=small, fill=(190, 190, 190))
    return np.asarray(im)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefix", required=True, help="saved search: the roll onto the front")
    ap.add_argument("--standup", required=True, help="saved search: the stand-up searched with the roll played first")
    ap.add_argument("--walk-checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--roll", type=float, default=90.0, help="start rolled this far from flat on the back, degrees (90 = on a side)")
    ap.add_argument("--side", type=float, default=1.0, help="+1 or -1: which side")
    ap.add_argument("--z-offset", type=float, default=0.07, help="metres added to the trunk height for the standing test (legs mounted higher)")
    ap.add_argument("--hold", type=float, default=1.5, help="seconds standing with a zero command before walking")
    ap.add_argument("--walk-seconds", type=float, default=8.0)
    ap.add_argument("--speed", type=float, default=0.3)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=576)
    ap.add_argument("--azimuth", type=float, default=130.0)
    ap.add_argument("--elevation", type=float, default=-15.0)
    ap.add_argument("--distance", type=float, default=0.75)
    ap.add_argument("--attempts", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    configure_torch_backends()
    dev = a.device

    def tune_viewer(cfg):
        v = cfg.viewer
        v.width, v.height = a.width, a.height
        v.distance, v.elevation, v.azimuth = a.distance, a.elevation, a.azimuth
        v.lookat = (0.0, 0.0, 0.0)
        v.max_extra_envs = 0

    # stand-up environment (stages 1 to 3)
    cfg_su = load_env_cfg(STAND_TASK)
    cfg_su.scene.num_envs = 1
    cfg_su.seed = a.seed
    cfg_su.episode_length_s = 120.0
    cfg_su.curriculum.clear()
    cfg_su.events.pop("push_robot", None)
    tune_viewer(cfg_su)

    # walking environment (stage 4), as in render_walk_video.py
    cfg_w = load_env_cfg(WALK_TASK, play=True)
    cfg_w.scene.num_envs = 1
    cfg_w.seed = a.seed
    cfg_w.episode_length_s = 120.0
    cfg_w.events.pop("push_robot", None)
    for name in [k for k in cfg_w.curriculum if "standing" in k]:
        cfg_w.curriculum.pop(name)
    twist = cfg_w.commands["twist"]
    twist.ranges.lin_vel_x = (a.speed, a.speed)
    twist.ranges.lin_vel_y = (0.0, 0.0)
    twist.ranges.ang_vel_z = (0.0, 0.0)
    twist.rel_standing_envs = 0.0
    twist.rel_heading_envs = 0.0
    twist.rel_turn_in_place_envs = 0.0
    twist.resampling_time_range = (1e9, 1e9)
    for cmd in cfg_w.commands.values():
        if hasattr(cmd, "debug_vis"):
            cmd.debug_vis = False
    tune_viewer(cfg_w)

    def action_signature(cfg):
        return {k: (getattr(v, "scale", None), getattr(v, "use_default_offset", None)) for k, v in cfg.actions.items()}

    print("stand-up action terms:", action_signature(cfg_su), "\nwalking action terms: ", action_signature(cfg_w), flush=True)
    if action_signature(cfg_su) != action_signature(cfg_w):
        raise SystemExit("the stand-up and walking tasks scale their actions differently: the motions cannot be replayed in the walking environment")

    su = ManagerBasedRlEnv(cfg=cfg_su, device=dev, render_mode="rgb_array")
    wbase = ManagerBasedRlEnv(cfg=cfg_w, device=dev, render_mode="rgb_array")
    if abs(su.step_dt - wbase.step_dt) > 1e-9:
        raise SystemExit(f"control periods differ: {su.step_dt} against {wbase.step_dt}")
    agent_cfg = load_rl_cfg(WALK_TASK)
    wenv = RslRlVecEnvWrapper(wbase, clip_actions=agent_cfg.clip_actions)
    runner = (load_runner_cls(WALK_TASK) or MjlabOnPolicyRunner)(wenv, asdict(agent_cfg), device=dev)
    runner.load(str(Path(a.walk_checkpoint).expanduser()), load_cfg={"actor": True}, strict=True, map_location=dev)
    policy = runner.get_inference_policy(device=dev)

    roll_theta, roll_s = load_motion(a.prefix, dev)
    up_theta, up_s = load_motion(a.standup, dev)
    dt = su.step_dt
    fps = round(1.0 / dt)
    su_robot, w_robot = su.scene["robot"], wbase.scene["robot"]
    last_second = round(1.0 / dt)

    def place_on_side():
        """Spawn on the back, heading +x, rolled about the spine by --roll toward --side (as standup_search --start-roll)."""
        su.reset()
        cy, sy, s = 1.0, 0.0, 2.0**-0.5
        w, x, y, z = s * cy, s * sy, -s * cy, s * sy
        th = a.side * np.radians(a.roll)
        ct, st = np.cos(th / 2), np.sin(th / 2)
        q = [w * ct - z * st, x * ct + y * st, y * ct - x * st, w * st + z * ct]
        su.sim.data.qpos[:, 3:7] = torch.tensor(q, device=dev, dtype=su.sim.data.qpos.dtype).expand(su.sim.data.qpos.shape[0], -1)
        su.sim.data.qvel[:, :6] = 0.0

    for attempt in range(1, a.attempts + 1):
        frames: list = []
        stage = 0
        up_hist: list = []

        def record(robot, env, t):
            nonlocal stage
            gx = float(robot.data.projected_gravity_b[0, 0])
            gz = float(robot.data.projected_gravity_b[0, 2])
            z = float(robot.data.root_link_pos_w[0, 2]) + a.z_offset
            up = z > 0.10 and gz < -0.94
            up_hist.append(up)
            if stage == 0 and t >= 0.5:
                stage = 1
            if stage <= 1 and gx > 0.9:
                stage = 2
            if stage <= 2 and t > roll_s:  # the stand-up motion starts when the roll motion has played out
                stage = 3
            if stage <= 3 and up and t > roll_s:
                stage = 4
            frames.append(caption(env.render(), stage, t))

        with torch.inference_mode():
            place_on_side()
            t = 0.0
            for theta, secs in ((roll_theta, roll_s), (up_theta, up_s)):
                steps = round(secs / dt)
                K = theta.shape[1]
                for i in range(steps):
                    pos = i / (steps - 1) * (K - 1)
                    k0 = min(int(pos), K - 2)
                    frac = pos - k0
                    su.step(theta[:, k0] * (1 - frac) + theta[:, k0 + 1] * frac)
                    t += dt
                    record(su_robot, su, t)
            stood = len(up_hist) >= last_second and all(up_hist[-last_second:])
            print(f"take {attempt}: standing after the stand-up: {stood} ({t:.1f} s)", flush=True)
            if not stood:
                continue

            # hand the simulator state to the walking environment
            qpos, qvel = su.sim.data.qpos.clone(), su.sim.data.qvel.clone()
            wenv.reset()
            wbase.sim.data.qpos[:] = qpos
            wbase.sim.data.qvel[:] = qvel
            if hasattr(wbase.sim, "forward"):
                wbase.sim.forward()
            obs = wenv.get_observations()
            term = wbase.command_manager.get_term("twist")
            hold_steps, walk_steps = round(a.hold / dt), round(a.walk_seconds / dt)
            fell = False
            start_xy = None
            for k in range(hold_steps + walk_steps):
                term.vel_command_b[:] = 0.0
                if k >= hold_steps:
                    term.vel_command_b[:, 0] = a.speed
                    if start_xy is None:
                        start_xy = w_robot.data.root_link_pos_w[0, :2].clone()
                        stage = 5
                obs, _, _, _ = wenv.step(policy(obs))
                t += dt
                if bool(wbase.reset_terminated[0]):
                    print(f"take {attempt}: fell after {k * dt:.1f} s of the walking environment, starting a new take", flush=True)
                    fell = True
                    break
                if k < hold_steps:
                    stage = 4
                record(w_robot, wbase, t)
            if fell:
                continue
            walked = (w_robot.data.root_link_pos_w[0, :2] - start_xy).norm().item()
            print(f"clean take {attempt}: walked {walked:.2f} m in {a.walk_seconds:g} s ({walked / a.walk_seconds:.2f} m/s, commanded {a.speed:g})", flush=True)

        out = Path(a.out).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        media.write_video(str(out), np.stack(frames), fps=fps)
        print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {len(frames)} frames, {len(frames) / fps:.1f} s)")
        break
    else:
        raise SystemExit(f"no clean take in {a.attempts} attempts")
    su.close()
    wenv.close()


if __name__ == "__main__":
    main()
