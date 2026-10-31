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
from mjlab.envs.mdp import dr
from mjlab.managers import EventTermCfg, SceneEntityCfg
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
ap.add_argument("--use-mean", action="store_true", help="--play: replay the search distribution's mean motion instead of the best single candidate")
ap.add_argument("--video", default=None, help="--play: mp4 of env 0")
ap.add_argument("--distance", type=float, default=0.5)
ap.add_argument("--freeze-hip", action="store_true",
                help="hold hip yaw and hip roll at the home pose (a leg with only hip pitch, knee and ankle)")
ap.add_argument("--z-offset", type=float, default=0.0,
                help="metres added to the trunk height before it is scored, for variants whose standing trunk is lower "
                     "(legs mounted dz higher: pass dz)")
ap.add_argument("--start-roll", type=float, default=None,
                help="face-up task: start every robot on its back rolled by exactly this many degrees to a random side "
                     "(90 = lying on a side), instead of the task's random start. Overrides --roll-max")
ap.add_argument("--play-start-roll", type=float, default=None,
                help="--play: start every robot rolled by exactly this many degrees from flat on the back instead of the saved start "
                     "(tests a saved motion at other angles)")
ap.add_argument("--prefix", default=None,
                help="a saved search whose motion is played first (its mean motion, or the best candidate with --prefix-best) before each "
                     "candidate, e.g. the roll onto the front; only what happens after it is scored, and --seconds is the candidate's part")
ap.add_argument("--prefix-best", action="store_true", help="play the prefix's best single candidate instead of its mean motion")
ap.add_argument("--goal", choices=["stand", "prone"], default="stand",
                help="stand: be upright for the last second. prone: be lying face-down (screen to the floor) for the last second, "
                     "the hand-off to the face-down stand-up")
ap.add_argument("--roll-max", type=float, default=None,
                help="face-up starts only: random partial roll toward the side, up to this many degrees (the stock task uses 90; "
                     "0 = flat on the back). Default: leave the task's setting")
ap.add_argument("--stress-mu", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                help="--play: the floor and every robot geom get one friction coefficient per robot, drawn from [LO, HI] "
                     "(the training randomization only moves the feet, 0.7 to 1.3 against a floor of 1.0, and MuJoCo takes the larger of the two)")
ap.add_argument("--stress-vin", type=float, nargs="+", default=None, metavar="V",
                help="battery voltage, V: one value for every robot, or LO HI drawn per robot (training: 6.5 to 8.2)")
ap.add_argument("--stress-drop", type=float, nargs="+", default=None, metavar="G",
                help="voltage sag per N*m of summed joint torque, V: one value or LO HI (training: 0 to 0.2)")
ap.add_argument("--stress-lag", type=int, nargs="+", default=None, metavar="STEPS",
                help="servo command delay, physics steps of 5 ms: one value or LO HI (training: 3 to 6)")
ap.add_argument("--stress-mass", type=float, nargs="+", default=None, metavar="G",
                help="grams added to the trunk body (density scaled up, centre of mass unchanged): one value or LO HI")
ap.add_argument("--stress-com", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
                help="trunk centre of mass shifted by this many millimetres along x (forward), y (left), z (up)")
ap.add_argument("--stress-com-range", type=float, default=None, metavar="MM",
                help="trunk centre of mass moved by a random amount in [-MM, MM] millimetres along each axis, redrawn at every reset")
ap.add_argument("--log-actuators", action="store_true", help="--play: print the peak torque and speed of each joint over the replay")
ap.add_argument("--log-contacts", action="store_true",
                help="--play: print which body parts touch which (trunk parts against leg parts, leg against leg, parts against the floor)")
ap.add_argument("--replicas", type=int, default=1,
                help="search: every candidate motion is simulated in this many robots (each with its own randomized physics, "
                     "friction, mass, battery, delay) and scored on their mean; --pop counts robots, so there are pop / replicas candidates")
a = ap.parse_args()

configure_torch_backends()
torch.manual_seed(a.seed)
saved = torch.load(a.play, map_location=a.device) if a.play else None
if saved:
    a.keyframes, a.seconds, a.asym = saved["keyframes"], saved["seconds"], saved["asym"]
    a.z_offset = saved.get("z_offset", 0.0)
    a.task, a.roll_max = saved.get("task", a.task), saved.get("roll_max", None)
    a.start_roll = saved.get("start_roll", None)
    a.goal = saved.get("goal", "stand")
    a.prefix, a.prefix_best = saved.get("prefix"), saved.get("prefix_best", False)
    if a.play_start_roll is not None:
        a.start_roll, a.roll_max = a.play_start_roll, None
n_env = a.n if saved else a.pop
prefix = torch.load(a.prefix, map_location=a.device) if a.prefix else None

cfg = load_env_cfg(a.task)
cfg.scene.num_envs = n_env
cfg.seed = a.seed
cfg.episode_length_s = a.seconds + (prefix["seconds"] if prefix else 0.0) + 2.0
cfg.curriculum.clear()
cfg.events.pop("push_robot", None)
if a.roll_max is not None:
    cfg.events["set_ground_state"].params["face_up_roll_max"] = float(np.radians(a.roll_max))


def _model_tensor(x):
    """A model field as a torch tensor sharing its memory (mjlab hands out either a tensor or a warp array)."""
    return x if isinstance(x, torch.Tensor) else x._tensor  # mjlab's TorchArray proxy keeps the torch view in _tensor


def _stress_friction(env, env_ids, lo: float, hi: float) -> None:
    """Startup event: the floor and every geom of the robot get one tangential friction coefficient per world, drawn from [lo, hi]."""
    gf = _model_tensor(env.sim.model.geom_friction)
    gf[:, :, 0] = torch.empty(gf.shape[0], 1, device=gf.device).uniform_(lo, hi)


def _stress_trunk(env, env_ids, grams_lo: float, grams_hi: float) -> None:
    """Startup event: more mass in the trunk body (density scaled up, so inertia follows), drawn per world from [grams_lo, grams_hi]."""
    import mujoco

    mj = env.sim.mj_model
    bid = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_BODY, "robot/trunk_base")
    mass = float(mj.body_mass[bid])
    lo, hi = (0.5 * float(np.log((mass + g / 1000.0) / mass)) for g in (grams_lo, grams_hi))
    asset_cfg = SceneEntityCfg("robot", body_names=("trunk_base",))
    asset_cfg.resolve(env.scene)
    dr.pseudo_inertia(env, env_ids, alpha_range=(lo, hi), asset_cfg=asset_cfg)
    print(f"stress: trunk_base {1000 * mass:.1f} g + {grams_lo:g} to {grams_hi:g} g", flush=True)


def _stress_com(env, env_ids, com_mm=None, range_mm=None) -> None:
    """Reset event, run after the task's own com randomization (which sets the centre of mass from the compile-time value on every
    reset and so undoes anything done at startup): the trunk's centre of mass is moved by com_mm millimetres along x, y, z, or by a
    random amount in [-range_mm, range_mm] along each axis."""
    asset_cfg = SceneEntityCfg("robot", body_names=("trunk_base",))
    asset_cfg.resolve(env.scene)
    if range_mm is not None:
        ranges = (-range_mm / 1000.0, range_mm / 1000.0)
    else:
        ranges = {i: (c / 1000.0, c / 1000.0) for i, c in enumerate(com_mm)}
    dr.body_ipos(env, env_ids, ranges=ranges, asset_cfg=asset_cfg, operation="add")
    import mujoco

    bid = mujoco.mj_name2id(env.sim.mj_model, mujoco.mjtObj.mjOBJ_BODY, "robot/trunk_base")
    print(f"stress: trunk_base body_ipos in world 0 after the reset events: {_model_tensor(env.sim.model.body_ipos)[0, bid].tolist()}", flush=True)


if a.stress_vin is not None or a.stress_drop is not None or a.stress_lag is not None:
    from copy import deepcopy

    robot_cfg = deepcopy(cfg.scene.entities["robot"])
    act_cfg = robot_cfg.articulation.actuators[0]
    if a.stress_vin is not None:
        act_cfg.vin_range = (a.stress_vin[0], a.stress_vin[-1])
    if a.stress_drop is not None:
        act_cfg.vin_drop_gain_range = (a.stress_drop[0], a.stress_drop[-1])
    if a.stress_lag is not None:
        act_cfg.delay_min_lag, act_cfg.delay_max_lag = a.stress_lag[0], a.stress_lag[-1]
    cfg.scene.entities = {**cfg.scene.entities, "robot": robot_cfg}
if a.stress_mu:
    cfg.events["stress_friction"] = EventTermCfg(func=_stress_friction, mode="startup", params={"lo": a.stress_mu[0], "hi": a.stress_mu[1]})
if a.stress_mass:
    cfg.events["stress_trunk"] = EventTermCfg(func=_stress_trunk, mode="startup",
                                              params={"grams_lo": a.stress_mass[0], "grams_hi": a.stress_mass[-1]})
if a.stress_com:
    cfg.events["stress_com"] = EventTermCfg(func=_stress_com, mode="reset", params={"com_mm": a.stress_com})
elif a.stress_com_range:
    cfg.events["stress_com"] = EventTermCfg(func=_stress_com, mode="reset", params={"range_mm": a.stress_com_range})
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


def expand(theta: torch.Tensor, asym: bool | None = None) -> torch.Tensor:
    """(N, K, free) -> (N, K, 10): mirrored legs unless --asym (or the given flag)."""
    return theta if (a.asym if asym is None else asym) else torch.cat([theta, -theta], dim=-1)


pfx_theta, pfx_steps = None, 0
if prefix:
    pfx_theta = expand(prefix["best" if a.prefix_best else "mean"].to(dev).unsqueeze(0), prefix["asym"])  # (1, K, 10)
    pfx_steps = round(prefix["seconds"] / env.step_dt)
settle_skip = 0 if pfx_steps else settle_steps  # the robots spawn up to 90 mm high and drop first, unless a prefix has already run


first_g: list = []  # projected gravity at the first step of the latest rollout, for the start-orientation breakdown
act_peak = {"tq": torch.zeros(n_env, 10, device=dev), "w": torch.zeros(n_env, 10, device=dev)}  # --log-actuators: per robot and joint


def log_actuators() -> None:
    if a.log_actuators:
        act_peak["tq"] = torch.maximum(act_peak["tq"], robot.data.qfrc_actuator.abs())
        act_peak["w"] = torch.maximum(act_peak["w"], robot.data.joint_vel.abs())


n_geom = env.sim.mj_model.ngeom
contact_seen = {"touched": np.zeros((n_env, n_geom, n_geom), dtype=bool), "steps": np.zeros((n_geom, n_geom)), "n": 0}


def log_contacts() -> None:
    """--log-contacts: which pairs of geoms touch, in the last physics substep of each policy step (a contact within 0.5 mm counts)."""
    if not a.log_contacts:
        return
    wd = env.sim.wp_data
    n = int(wd.nacon.numpy()[0])
    contact_seen["n"] += 1
    if n == 0:
        return
    geom, world, dist = wd.contact.geom.numpy()[:n], wd.contact.worldid.numpy()[:n], wd.contact.dist.numpy()[:n]
    keep = dist < 0.0005
    g0, g1 = np.minimum(geom[keep, 0], geom[keep, 1]), np.maximum(geom[keep, 0], geom[keep, 1])
    key = np.unique((world[keep].astype(np.int64) * n_geom + g0) * n_geom + g1)  # one entry per robot and pair
    w, rest = key // (n_geom * n_geom), key % (n_geom * n_geom)
    p0, p1 = rest // n_geom, rest % n_geom
    contact_seen["touched"][w, p0, p1] = True
    np.add.at(contact_seen["steps"], (p0, p1), 1)


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
    if a.start_roll is not None:  # every robot on its back rolled by exactly this far, to a random side, random heading
        n = n_env
        yaw = torch.rand(n, device=dev) * 2 * np.pi - np.pi
        cy, sy, s = torch.cos(yaw / 2), torch.sin(yaw / 2), 2.0 ** -0.5
        w, x, y, z = s * cy, s * sy, -s * cy, s * sy  # the task's face-up orientation
        theta = torch.where(torch.rand(n, device=dev) < 0.5, 1.0, -1.0) * float(np.radians(a.start_roll))
        ct, st = torch.cos(theta / 2), torch.sin(theta / 2)  # roll about the spine (body z), right-multiplied as in the task
        env.sim.data.qpos[:, 3:7] = torch.stack([w * ct - z * st, x * ct + y * st, y * ct - x * st, w * st + z * ct], dim=1)
        env.sim.data.qvel[:, :6] = 0.0
    for i in range(pfx_steps):  # the prefix motion, not scored
        pos = i / (pfx_steps - 1) * (pfx_theta.shape[1] - 1)
        k0 = min(int(pos), pfx_theta.shape[1] - 2)
        frac = pos - k0
        env.step((pfx_theta[:, k0] * (1 - frac) + pfx_theta[:, k0 + 1] * frac).expand(n_env, -1))
        log_actuators()
        log_contacts()
        if i == 0:
            first_g[:] = [robot.data.projected_gravity_b.clone()]
        if frames is not None:
            frames.append(env.render())
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
        log_actuators()
        log_contacts()
        if i == 0 and not pfx_steps:
            first_g[:] = [robot.data.projected_gravity_b.clone()]  # the start orientation, before the robot has moved much
        z_raw = robot.data.root_link_pos_w[:, 2]
        z = z_raw + a.z_offset
        gz = robot.data.projected_gravity_b[:, 2]
        # a blown-up simulation (trunk under the floor or flung up, or non-finite) must never score
        ok = torch.isfinite(z_raw) & torch.isfinite(gz) & (z_raw > 0.022) & (z_raw < 0.16)
        h = ((z - 0.04) / 0.07).clamp(0, 1)
        p = (-gz).clamp(0, 1)
        gx = robot.data.projected_gravity_b[:, 0]
        c = torch.where(ok, p * (0.3 + 0.7 * h), torch.zeros_like(p))
        if a.goal == "prone":  # credit for the screen facing the floor (gravity along +x in the body frame), not for height
            c = torch.where(ok, gx.clamp(0, 1), torch.zeros_like(p))
        if i >= settle_skip:  # the robots spawn up to 90 mm high and drop to rest first
            best = torch.maximum(best, c)
            peak_z = torch.maximum(peak_z, z)
            peak_p = torch.maximum(peak_p, p)
        if i >= steps - last_second:
            final += c / last_second
        held.append(ok & (gx > 0.9) if a.goal == "prone" else ok & (z > 0.10) & (gz < -0.94))
        if trace is not None:
            trace.append((z_raw.clone(), gz.clone(), ok.clone()))
        if frames is not None:
            frames.append(env.render())
    up = torch.stack(held[-last_second:]).all(0)
    return 0.3 * best + 0.7 * final + 0.5 * up.float(), up, peak_z, peak_p


if saved:
    theta = saved["mean" if a.use_mean else "best"].to(dev).unsqueeze(0).expand(n_env, -1, -1)
    frames = [] if a.video else None
    trace: list = []
    with torch.inference_mode():
        score, up, peak, prog = rollout(expand(theta), frames, trace)
    print(f"replay of {a.play}: {n_env} robots, {a.seconds:g} s: {'face-down' if a.goal == 'prone' else 'up'} for the last second {int(up.sum())}/{n_env} "
          f"({100 * up.float().mean():.0f}%), peak trunk height median {1000 * float(peak.median()):.0f} mm, "
          f"max {1000 * float(peak.max()):.0f} mm, closest to upright (tilt from vertical, 90 = lying) median "
          f"{tilt(prog.median()):.0f} deg, best {tilt(prog.max()):.0f} deg")
    if "FaceUp" in a.task:
        g0 = first_g[0]
        roll0 = torch.rad2deg(torch.atan2(g0[:, 1].abs(), -g0[:, 0]))  # 0 = flat on the back, 90 = on its side
        edges = [0, 15, 30, 45, 60, 75, 91]
        print("by start orientation (roll away from flat on the back, deg): " + ", ".join(
            f"{lo}-{hi - 1 if hi == 91 else hi}: {int((up & (roll0 >= lo) & (roll0 < hi)).sum())}/{int(((roll0 >= lo) & (roll0 < hi)).sum())}"
            for lo, hi in zip(edges[:-1], edges[1:])))
    if a.log_actuators:
        names = list(robot.joint_names)
        q = lambda x, p: torch.quantile(x, p, dim=0)  # noqa: E731
        print("peak joint torque over the replay, N*m, by joint (median robot / 95th percentile / worst): " + "; ".join(
            f"{n} {float(q(act_peak['tq'], .5)[j]):.2f}/{float(q(act_peak['tq'], .95)[j]):.2f}/{float(act_peak['tq'][:, j].max()):.2f}" for j, n in enumerate(names)))
        print("peak joint speed over the replay, rad/s, by joint (median robot / 95th percentile / worst): " + "; ".join(
            f"{n} {float(q(act_peak['w'], .5)[j]):.1f}/{float(q(act_peak['w'], .95)[j]):.1f}/{float(act_peak['w'][:, j].max()):.1f}" for j, n in enumerate(names)))
    if a.log_contacts:
        import mujoco

        mj = env.sim.mj_model
        nm = [(mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom{g}").replace("robot/", "").replace("_collision", "") for g in range(n_geom)]
        trunk = {"bezel", "comb", "tail", "shell", "wing_left", "wing_right"}
        kind = lambda g: "trunk" if nm[g] in trunk else "floor" if nm[g] == "terrain" else "leg"  # noqa: E731
        total = max(contact_seen["n"], 1)
        rows = []
        for p0, p1 in zip(*np.nonzero(contact_seen["steps"])):
            k = tuple(sorted((kind(p0), kind(p1))))
            rows.append((k, nm[p0], nm[p1], contact_seen["touched"][:, p0, p1].mean(), contact_seen["steps"][p0, p1] / (total * n_env)))
        for k, title in ((("leg", "trunk"), "trunk part against leg part"), (("leg", "leg"), "leg part against leg part (the two legs, or a leg's own parts)"),
                         (("floor", "trunk"), "trunk part against the floor"), (("floor", "leg"), "leg part against the floor")):
            top = sorted((r for r in rows if r[0] == k), key=lambda r: -r[3])[:8]
            print(f"contacts, {title}: pair | robots that touch it at least once | share of all robot-steps: " + "; ".join(
                f"{r[1]}-{r[2]} {100 * r[3]:.0f}% {100 * r[4]:.1f}%" for r in top))
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
m_rep = a.replicas
n_cand = n_env // m_rep
n_elite = max(2, int(a.elite * n_cand))
best_ever, best_theta = -1.0, mu.clone()
print(f"search: {n_cand} candidates x {m_rep} robots x {a.gens} generations, {K} keyframes over {a.seconds:g} s, "
      f"{'asymmetric' if a.asym else 'mirrored'} legs ({K * n_free} numbers), {steps} steps per rollout", flush=True)
t0 = time.time()
with torch.inference_mode():
    for g in range(a.gens):
        theta = (mu + sigma * torch.randn(n_cand, K, n_free, device=dev)).clamp(-a.limit, a.limit) * joint_mask
        score, up, peak, prog = rollout(expand(theta.repeat_interleave(m_rep, dim=0)))
        if m_rep > 1:  # a candidate's score is the mean over its robots, each with its own randomized physics
            score = score.view(n_cand, m_rep).mean(1)
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
                    "best_score": best_ever, "z_offset": a.z_offset, "task": a.task, "roll_max": a.roll_max, "start_roll": a.start_roll, "goal": a.goal,
                    "prefix": a.prefix, "prefix_best": a.prefix_best}, a.out)
print(f"saved {a.out}; best score {best_ever:.3f} (1.5 = up and upright for the last second)")
env.close()
