"""Physics randomization beyond the stock stand-up task's, for testing and training the MicroCluck recovery motions.

The stock task moves the feet's friction (0.7 to 1.3 against a floor of 1.0, and MuJoCo takes the larger of the two, so the
effective friction is never under 1.0), the trunk's mass by +-5% and its centre of mass by +-3 mm, the joints' friction and
armature, and the battery between 6.5 and 8.2 V with a sag of 0 to 0.2 V per N*m and a servo delay of 3 to 6 steps. The open-loop
get-up works only inside a narrow window of the trunk's fore-aft balance point (PRD section 7.5), so the recovery work needs
the ranges widened: friction of the floor and every part, the battery, the delay, more mass in the trunk, and the centre of mass.

  apply_stress(cfg, Stress(mu=(0.3, 0.5), vin=(6.4, 6.4), drop=(0.2, 0.2), mass=(60, 60), com=(-8, 0, 0)))

adds startup / reset events to an env config and overrides the actuator's battery and delay ranges. A field left None is untouched.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch

from mjlab.envs.mdp import dr
from mjlab.managers import EventTermCfg, SceneEntityCfg


@dataclass
class Stress:
    mu: tuple[float, float] | None = None  # friction coefficient of the floor and every robot geom, one per robot
    vin: tuple[float, float] | None = None  # battery voltage, V
    drop: tuple[float, float] | None = None  # voltage sag per N*m of summed joint torque, V
    lag: tuple[int, int] | None = None  # servo command delay, physics steps of 5 ms
    mass: tuple[float, float] | None = None  # grams added to the trunk body (density scaled up, centre of mass unchanged)
    com: tuple[float, float, float] | None = None  # trunk centre of mass moved by this many mm along x (forward), y (left), z (up)
    com_range: float | tuple[float, float, float] | None = None  # moved by a random amount in +-this many mm per axis, at every reset


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


_com_printed = False


def _stress_com(env, env_ids, com_mm=None, range_mm=None) -> None:
    """Reset event, run after the task's own com randomization (which sets the centre of mass from the compile-time value on every
    reset and so undoes anything done at startup): the trunk's centre of mass is moved by com_mm millimetres along x, y, z, or by a
    random amount in [-range_mm, range_mm] along each axis (one half-range, or one per axis)."""
    global _com_printed
    asset_cfg = SceneEntityCfg("robot", body_names=("trunk_base",))
    asset_cfg.resolve(env.scene)
    if range_mm is not None:
        r = (range_mm,) * 3 if isinstance(range_mm, (int, float)) else tuple(range_mm)
        ranges = {i: (-m / 1000.0, m / 1000.0) for i, m in enumerate(r)}
    else:
        ranges = {i: (c / 1000.0, c / 1000.0) for i, c in enumerate(com_mm)}
    dr.body_ipos(env, env_ids, ranges=ranges, asset_cfg=asset_cfg, operation="add")
    if not _com_printed:
        import mujoco

        bid = mujoco.mj_name2id(env.sim.mj_model, mujoco.mjtObj.mjOBJ_BODY, "robot/trunk_base")
        print(f"stress: trunk_base body_ipos in world 0 after the reset events: {_model_tensor(env.sim.model.body_ipos)[0, bid].tolist()}", flush=True)
        _com_printed = True


def apply_stress(cfg, s: Stress) -> None:
    """Add the events of s to an env config and override the actuator ranges; an event of the same name is replaced."""
    if s.vin is not None or s.drop is not None or s.lag is not None:
        robot_cfg = deepcopy(cfg.scene.entities["robot"])
        act_cfg = robot_cfg.articulation.actuators[0]
        if s.vin is not None:
            act_cfg.vin_range = tuple(s.vin)
        if s.drop is not None:
            act_cfg.vin_drop_gain_range = tuple(s.drop)
        if s.lag is not None:
            act_cfg.delay_min_lag, act_cfg.delay_max_lag = s.lag
        cfg.scene.entities = {**cfg.scene.entities, "robot": robot_cfg}
    if s.mu:
        cfg.events["stress_friction"] = EventTermCfg(func=_stress_friction, mode="startup", params={"lo": s.mu[0], "hi": s.mu[1]})
    if s.mass:
        cfg.events["stress_trunk"] = EventTermCfg(func=_stress_trunk, mode="startup", params={"grams_lo": s.mass[0], "grams_hi": s.mass[1]})
    if s.com:
        cfg.events["stress_com"] = EventTermCfg(func=_stress_com, mode="reset", params={"com_mm": s.com})
    elif s.com_range:
        cfg.events["stress_com"] = EventTermCfg(func=_stress_com, mode="reset", params={"range_mm": s.com_range})


def add_stress_args(ap) -> None:
    """The --stress-* command-line options of the replay and evaluation scripts."""
    ap.add_argument("--stress-mu", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                    help="the floor and every robot geom get one friction coefficient per robot, drawn from [LO, HI] "
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
    ap.add_argument("--stress-com-range", type=float, nargs="+", default=None, metavar="MM",
                    help="trunk centre of mass moved by a random amount in [-MM, MM] millimetres along each axis (one value, or X Y Z), redrawn at every reset")


def stress_from_args(a) -> Stress:
    ends = lambda v: None if v is None else (v[0], v[-1])  # noqa: E731  one value or LO HI
    com_range = None if not a.stress_com_range else (a.stress_com_range[0] if len(a.stress_com_range) == 1 else tuple(a.stress_com_range))
    return Stress(mu=ends(a.stress_mu), vin=ends(a.stress_vin), drop=ends(a.stress_drop), lag=ends(a.stress_lag),
                  mass=ends(a.stress_mass), com=tuple(a.stress_com) if a.stress_com else None, com_range=com_range)


def trunk_com_offset(env) -> torch.Tensor:
    """The trunk's centre of mass in this world less its compile-time value, metres (num_envs, 3): what the stress events moved."""
    import mujoco

    mj = env.sim.mj_model
    bid = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_BODY, "robot/trunk_base")
    ipos = _model_tensor(env.sim.model.body_ipos)[:, bid]
    return (ipos - torch.as_tensor(mj.body_ipos[bid], device=ipos.device, dtype=ipos.dtype)).expand(env.num_envs, 3)


def trunk_extra_mass(env) -> torch.Tensor:
    """The trunk's mass in this world less its compile-time value, kg (num_envs, 1)."""
    import mujoco

    mj = env.sim.mj_model
    bid = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_BODY, "robot/trunk_base")
    mass = _model_tensor(env.sim.model.body_mass)[:, bid]
    return (mass - float(mj.body_mass[bid])).reshape(-1, 1).expand(env.num_envs, 1)
