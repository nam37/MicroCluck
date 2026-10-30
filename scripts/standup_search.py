"""Open-loop get-up search: does ANY leg motion take MicroCluck from lying face-down to standing?

Not a policy. A candidate is K keyframes of joint-target offsets from the home pose, played back by linear interpolation
over a few seconds, the same for every start. The cross-entropy method searches the keyframes with thousands of face-down
robots simulated at once (one candidate per robot, each start randomized as in training). It answers "is there a way up for
this body" without depending on RL exploration, and a found motion is a reference for seeding a policy.

  search:  uv run python scripts/standup_search.py --out best.pt
  replay:  uv run python scripts/standup_search.py --play best.pt --n 64 --video best.mp4

The legs are mirrored by default (right joint = minus the left one, as in the home pose and the stock sit keyframe).
A robot counts as up when its trunk is above 0.10 m and tilted under 20 degrees for the whole last second.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

import mjlab_microduck.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.torch import configure_torch_backends

ap = argparse.ArgumentParser()
ap.add_argument("--task", default="Mjlab-StandUp-FaceDown-Flat-MicroCluck")
ap.add_argument("--pop", type=int, default=2048, help="candidates (robots) per generation")
ap.add_argument("--gens", type=int, default=40)
ap.add_argument("--keyframes", type=int, default=6)
ap.add_argument("--seconds", type=float, default=5.0)
ap.add_argument("--elite", type=float, default=0.05)
ap.add_argument("--sigma", type=float, default=0.5, help="initial std of the keyframe offsets, rad")
ap.add_argument("--limit", type=float, default=1.6, help="clip of the keyframe offsets, rad")
ap.add_argument("--asym", action="store_true", help="search all 10 joints instead of mirroring the legs")
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--out", default="standup_search_best.pt")
ap.add_argument("--play", default=None, help="replay a saved result instead of searching")
ap.add_argument("--n", type=int, default=64, help="robots for --play")
ap.add_argument("--video", default=None, help="--play: mp4 of env 0")
ap.add_argument("--distance", type=float, default=0.5)
ap.add_argument("--freeze-hip", action="store_true",
                help="hold hip yaw and hip roll at the home pose (a leg with only hip pitch, knee and ankle)")
ap.add_argument("--z-offset", type=float, default=0.0,
                help="metres added to the trunk height before it is scored, for variants whose standing trunk is lower "
                     "(legs mounted dz higher: pass dz)")
a = ap.parse_args()

configure_torch_backends()
torch.manual_seed(a.seed)
saved = torch.load(a.play, map_location=a.device) if a.play else None
if saved:
    a.keyframes, a.seconds, a.asym = saved["keyframes"], saved["seconds"], saved["asym"]
    a.z_offset = saved.get("z_offset", 0.0)
n_env = a.n if saved else a.pop

cfg = load_env_cfg(a.task)
cfg.scene.num_envs = n_env
cfg.seed = a.seed
cfg.episode_length_s = a.seconds + 2.0
cfg.curriculum.clear()
cfg.events.pop("push_robot", None)
if a.video:
    cfg.viewer.width, cfg.viewer.height = 1280, 720
    cfg.viewer.distance, cfg.viewer.elevation, cfg.viewer.azimuth = a.distance, -20.0, 130.0
env = ManagerBasedRlEnv(cfg=cfg, device=a.device, render_mode="rgb_array" if a.video else None)
robot = env.scene["robot"]
dev = env.device
steps = round(a.seconds / env.step_dt)
last_second = round(1.0 / env.step_dt)
settle_steps = round(0.5 / env.step_dt)
n_free = 10 if a.asym else 5
joint_mask = torch.ones(n_free, device=dev)
if a.freeze_hip:
    joint_mask[[0, 1, 5, 6] if a.asym else [0, 1]] = 0.0  # yaw and roll of each leg


def tilt(p) -> float:
    """Trunk tilt from vertical in degrees, from p = cos(tilt)."""
    return float(np.degrees(np.arccos(min(max(float(p), 0.0), 1.0))))


def expand(theta: torch.Tensor) -> torch.Tensor:
    """(N, K, free) -> (N, K, 10): mirrored legs unless --asym."""
    return theta if a.asym else torch.cat([theta, -theta], dim=-1)


def rollout(theta10: torch.Tensor, frames=None, trace=None):
    """Play each robot's keyframes from a fresh face-down start. Returns score, up-at-end, peak height, peak uprightness.

    The score gives partial credit so the search has something to climb. Each step is worth c = p * (0.3 + 0.7 * h),
    where p = -gravity_z in the body frame is the cosine of the trunk's tilt from vertical (1 upright, 0 lying down on
    its front, back or side) and h is the height (0 at the face-down rest height, 1 at 0.11 m). The score is
    0.3 * the best step + 0.7 * the mean over the last second + 0.5 when the robot is up for the last second, so ending
    up matters more than passing through. Two earlier versions were wrong: needing height and uprightness together scored
    0.000 for every candidate (no gradient), and measuring pitch with atan2(gx, -gz) read a robot lying on its side or
    on its back as fully upright, which a search with independent legs found in its first generation."""
    env.reset()
    K = theta10.shape[1]
    best = torch.zeros(n_env, device=dev)
    final = torch.zeros(n_env, device=dev)
    peak_z = torch.zeros(n_env, device=dev)
    peak_p = torch.zeros(n_env, device=dev)
    held = []
    for i in range(steps):
        pos = i / (steps - 1) * (K - 1)
        k0 = min(int(pos), K - 2)
        frac = pos - k0
        act = theta10[:, k0] * (1 - frac) + theta10[:, k0 + 1] * frac
        env.step(act)
        z_raw = robot.data.root_link_pos_w[:, 2]
        z = z_raw + a.z_offset
        gz = robot.data.projected_gravity_b[:, 2]
        # a blown-up simulation (trunk under the floor or flung up, or non-finite) must never score
        ok = torch.isfinite(z_raw) & torch.isfinite(gz) & (z_raw > 0.022) & (z_raw < 0.16)
        h = ((z - 0.04) / 0.07).clamp(0, 1)
        p = (-gz).clamp(0, 1)
        c = torch.where(ok, p * (0.3 + 0.7 * h), torch.zeros_like(p))
        if i >= settle_steps:  # the robots spawn up to 90 mm high and drop to rest first
            best = torch.maximum(best, c)
            peak_z = torch.maximum(peak_z, z)
            peak_p = torch.maximum(peak_p, p)
        if i >= steps - last_second:
            final += c / last_second
        held.append(ok & (z > 0.10) & (gz < -0.94))
        if trace is not None:
            trace.append((z_raw.clone(), gz.clone(), ok.clone()))
        if frames is not None:
            frames.append(env.render())
    up = torch.stack(held[-last_second:]).all(0)
    return 0.3 * best + 0.7 * final + 0.5 * up.float(), up, peak_z, peak_p


if saved:
    theta = saved["best"].to(dev).unsqueeze(0).expand(n_env, -1, -1)
    frames = [] if a.video else None
    trace: list = []
    with torch.inference_mode():
        score, up, peak, prog = rollout(expand(theta), frames, trace)
    print(f"replay of {a.play}: {n_env} robots, {a.seconds:g} s: up for the last second {int(up.sum())}/{n_env} "
          f"({100 * up.float().mean():.0f}%), peak trunk height median {1000 * float(peak.median()):.0f} mm, "
          f"max {1000 * float(peak.max()):.0f} mm, closest to upright (tilt from vertical, 90 = lying) median "
          f"{tilt(prog.median()):.0f} deg, best {tilt(prog.max()):.0f} deg")
    print("trace, median over robots every 0.2 s: time s | trunk-frame height mm | tilt deg (90 = lying) | robots in the valid band")
    for i in range(0, len(trace), round(0.2 / env.step_dt)):
        z_t, gz_t, ok_t = trace[i]
        t_deg = torch.rad2deg(torch.arccos((-gz_t).clamp(0, 1)))
        print(f"   {i * env.step_dt:4.2f} | {1000 * float(z_t.median()):6.1f} | {float(t_deg.median()):5.1f} | {int(ok_t.sum())}/{n_env}")
    Z = torch.stack([t[0] for t in trace])  # steps x robots, raw trunk-frame height
    i_peak = Z[settle_steps:].argmax(0) + settle_steps
    print(f"peak trunk height: at step {int(i_peak.float().median())} (t = {float(i_peak.float().median()) * env.step_dt:.2f} s) in the median robot; "
          f"heights in mm around robot 0's peak: " + " ".join(f"{1000 * float(Z[j, 0]):.0f}" for j in range(max(0, int(i_peak[0]) - 4), min(len(Z), int(i_peak[0]) + 5))))
    if a.video:
        import mediapy as media

        media.write_video(a.video, np.stack(frames), fps=round(1 / env.step_dt))
        print("wrote", a.video)
    env.close()
    raise SystemExit

K = a.keyframes
mu = torch.zeros(K, n_free, device=dev)
sigma = torch.full((K, n_free), a.sigma, device=dev)
n_elite = max(2, int(a.elite * n_env))
best_ever, best_theta = -1.0, mu.clone()
print(f"search: {n_env} candidates x {a.gens} generations, {K} keyframes over {a.seconds:g} s, "
      f"{'asymmetric' if a.asym else 'mirrored'} legs ({K * n_free} numbers), {steps} steps per rollout", flush=True)
t0 = time.time()
with torch.inference_mode():
    for g in range(a.gens):
        theta = (mu + sigma * torch.randn(n_env, K, n_free, device=dev)).clamp(-a.limit, a.limit) * joint_mask
        score, up, peak, prog = rollout(expand(theta))
        order = score.argsort(descending=True)
        elite = theta[order[:n_elite]]
        if float(score[order[0]]) > best_ever:
            best_ever, best_theta = float(score[order[0]]), theta[order[0]].clone()
        mu = 0.3 * mu + 0.7 * elite.mean(0)
        sigma = (0.3 * sigma + 0.7 * elite.std(0)).clamp(min=0.05)
        print(f"[{time.strftime('%H:%M')}] gen {g + 1:3d}/{a.gens}: best score {float(score[order[0]]):.3f} (ever {best_ever:.3f}), "
              f"elite mean {float(score[order[:n_elite]].mean()):.3f}, up at end {int(up.sum())}/{n_env}, "
              f"highest trunk {1000 * float(peak.max()):.0f} mm, closest to upright {tilt(prog.max()):.0f} deg of tilt, "
              f"noise {float(sigma.mean()):.2f}, {time.time() - t0:.0f} s", flush=True)
        torch.save({"best": best_theta.cpu(), "mean": mu.cpu(), "keyframes": K, "seconds": a.seconds, "asym": a.asym,
                    "best_score": best_ever, "z_offset": a.z_offset}, a.out)
print(f"saved {a.out}; best score {best_ever:.3f} (1.5 = up and upright for the last second)")
env.close()
