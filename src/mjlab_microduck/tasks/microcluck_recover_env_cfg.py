"""MicroCluck feedback recovery: a learned correction on top of the saved open-loop get-up motions.

The open-loop motions (motions/microcluck, found by scripts/standup_search.py) roll the robot from its side or its back onto its
front and then stand it up, but only inside a window of about 12 mm of the trunk's fore-aft balance point (PRD section 7.5): a motion
cannot react to a balance point it cannot see. Here the same motions are the reference and the policy is the feedback:

  joint target = home pose + the reference motion at the current time + RESIDUAL_SCALE * the policy's output

The policy sees the IMU (gravity, angular velocity), the joint positions and velocities, its last output, the reference offset for
this step, the time and which of the two motions is running. It is trained on starts on the side (70 to 90 degrees of roll from the
back) and on the back (up to 30 degrees), with the physics widened as in microcluck_stress.py (friction, battery, delay, extra
trunk mass, centre of mass), and rewarded with the search's own score: face-down while the roll runs, then upright and tall, plus a
bonus while the trunk is up. The critic also sees the centre-of-mass shift and the extra mass, which the policy has to infer.

Two tasks: Mjlab-Recover-MicroCluck (the widened physics, for training) and Mjlab-Recover-Nominal-MicroCluck (the stock stand-up
task's randomization only, the base for scripts/recover_eval.py, which adds whatever stress a test names).
"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import torch

from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.envs.mdp.actions.actions import JointPositionAction
from mjlab.managers import EventTermCfg, ObservationTermCfg, RewardTermCfg
from mjlab.tasks.velocity import mdp

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks import microduck_standup_env_cfg as _stock
from mjlab_microduck.tasks.microcluck_standup_env_cfg import make_microcluck_standup_env_cfg
from mjlab_microduck.tasks.microcluck_stress import Stress, apply_stress, trunk_com_offset, trunk_extra_mass

MOTIONS = Path(__file__).resolve().parents[3] / "motions" / "microcluck"
# (name, roll onto the front, stand-up behind it); the chain index is the position in this tuple.
CHAINS = (("side", "prone_side_r.pt", "chain_side_w.pt"), ("back", "prone_back_r.pt", "chain_back_w.pt"))
RESIDUAL_SCALE = 0.3  # rad of joint target per unit of policy output
EPISODE_S = 12.0  # the motions take 11 s (6 s roll, 5 s stand-up); the last pose is held for the rest
ACTION = "joint_pos"
# The widened physics of the training: friction of the floor and every part, battery with sag, servo delay (steps of 5 ms), extra trunk
# mass in grams, and the trunk's centre of mass by +-mm along x (fore-aft), y, z. Fore-aft is the one the open-loop motions are most
# sensitive to (about 4 mm back to 8 mm forward of the model's value), so it gets the widest range.
TRAIN_STRESS = Stress(mu=(0.3, 1.5), vin=(6.4, 8.2), drop=(0.0, 0.2), lag=(3, 10), mass=(0.0, 60.0), com_range=(16.0, 6.0, 8.0))


def _keyframe_table(m: dict, step_dt: float, device) -> torch.Tensor:
    """A saved search's mean motion as joint-target offsets for every policy step: (steps, 10), interpolated as standup_search.py plays it."""
    theta = m["mean"].to(device)
    theta = theta if m["asym"] else torch.cat([theta, -theta], dim=-1)  # the stand-ups mirror the legs
    steps, K = round(m["seconds"] / step_dt), theta.shape[0]
    pos = torch.arange(steps, device=device) / (steps - 1) * (K - 1)
    k0 = pos.floor().long().clamp(max=K - 2)
    frac = (pos - k0).unsqueeze(-1)
    return theta[k0] * (1 - frac) + theta[k0 + 1] * frac


@dataclass(kw_only=True)
class ResidualJointPositionActionCfg(JointPositionActionCfg):
    """Joint target = home pose + the reference motion at the current time + scale * action."""

    def build(self, env) -> ResidualJointPositionAction:
        return ResidualJointPositionAction(self, env)


class ResidualJointPositionAction(JointPositionAction):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        tables, rolls, z_offsets = [], [], set()
        for _, roll_file, stand_file in CHAINS:
            roll, stand = (torch.load(MOTIONS / f, map_location=self.device) for f in (roll_file, stand_file))
            assert roll["goal"] == "prone" and stand["goal"] == "stand", (roll_file, stand_file)
            z_offsets.add(stand["z_offset"])
            t_roll, t_stand = (_keyframe_table(m, env.step_dt, self.device) for m in (roll, stand))
            tables.append(torch.cat([t_roll, t_stand]))
            rolls.append(len(t_roll))
        assert len(set(rolls)) == 1 and len({len(t) for t in tables}) == 1 and len(z_offsets) == 1, (rolls, z_offsets)
        self.tables = torch.stack(tables)  # (chains, steps, 10)
        self.roll_steps, self.n_steps, self.z_offset = rolls[0], tables[0].shape[0], z_offsets.pop()
        self.chain = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def reference(self) -> torch.Tensor:
        """The reference offsets for the next action, (num_envs, 10); the last pose is held after the motion ends."""
        step = self._env.episode_length_buf.clamp(max=self.n_steps - 1)
        return self.tables[self.chain, step]

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self._processed_actions = self._offset + self.reference() + self._raw_actions * self._scale


def _term(env) -> ResidualJointPositionAction:
    return env.action_manager.get_term(ACTION)


# ── Observations ──────────────────────────────────────────────────────────────

def reference_offset(env) -> torch.Tensor:
    return _term(env).reference()


def motion_phase(env) -> torch.Tensor:
    t = _term(env)
    return (env.episode_length_buf.float() / t.n_steps).clamp(max=1.0).unsqueeze(-1)


def motion_chain(env) -> torch.Tensor:
    t = _term(env)
    return torch.nn.functional.one_hot(t.chain, len(CHAINS)).float()


def trunk_com_shift(env) -> torch.Tensor:
    return trunk_com_offset(env)


def trunk_added_mass(env) -> torch.Tensor:
    return trunk_extra_mass(env)


# ── Rewards ───────────────────────────────────────────────────────────────────

def _trunk_state(env):
    """Trunk height (offset as the search scored it), -gravity z (cosine of the tilt), gravity x, and whether the simulation is sane."""
    robot = env.scene["robot"]
    z_raw = robot.data.root_link_pos_w[:, 2]
    g = robot.data.projected_gravity_b
    ok = torch.isfinite(z_raw) & torch.isfinite(g).all(dim=-1) & (z_raw > 0.022) & (z_raw < 0.16)
    return z_raw + _term(env).z_offset, g, ok


def recovery_progress(env) -> torch.Tensor:
    """The search's per-step score: face-down (gravity along +x of the trunk) while the roll runs, then cos(tilt) * (0.3 + 0.7 * height)."""
    z, g, ok = _trunk_state(env)
    h = ((z - 0.04) / 0.07).clamp(0, 1)
    c_roll = g[:, 0].clamp(0, 1)
    c_stand = (-g[:, 2]).clamp(0, 1) * (0.3 + 0.7 * h)
    rolling = (env.episode_length_buf - 1) < _term(env).roll_steps  # the reward is computed after the step counter has advanced
    return torch.where(ok, torch.where(rolling, c_roll, c_stand), torch.zeros_like(c_roll))


def recovery_up(env) -> torch.Tensor:
    """1 while the trunk is up as the search counted it (higher than 0.10 m, tilted under 20 degrees), once the roll is done."""
    z, g, ok = _trunk_state(env)
    rolling = (env.episode_length_buf - 1) < _term(env).roll_steps
    return (ok & ~rolling & (z > 0.10) & (g[:, 2] < -0.94)).float()


def residual_l2(env) -> torch.Tensor:
    return torch.sum(torch.square(env.action_manager.action), dim=1)


# ── Start state ───────────────────────────────────────────────────────────────

def set_recovery_start(env, env_ids, side_prob: float = 0.5, side_roll: tuple[float, float] = (math.radians(70), math.radians(90)),
                       back_roll_max: float = math.radians(30)) -> None:
    """Reset event: lying on the back (rolled up to back_roll_max to a random side) for the back motion, or rolled side_roll away from the
    back, to a random side, for the side motion; random heading, dropped from 50 to 90 mm. Also picks which motion the robot plays."""
    if env_ids is None or len(env_ids) == 0:
        return
    microduck_mdp.set_random_ground_state(
        env, env_ids, face_down_prob=0.0, face_up_prob=1.0, sitting_prob=0.0, standing_prob=0.0,
        prone_z_min=0.05, prone_z_max=0.09, face_up_roll_max=back_roll_max)
    ids = env_ids.to(env.device).long()
    side = torch.rand(len(ids), device=env.device) < side_prob
    _term(env).chain[ids] = torch.where(side, 0, 1)  # CHAINS order: side, back
    s = ids[side]
    if len(s) == 0:
        return
    n, dev = len(s), env.device
    yaw = torch.rand(n, device=dev) * 2 * math.pi - math.pi
    cy, sy, c = torch.cos(yaw / 2), torch.sin(yaw / 2), 2.0 ** -0.5
    w, x, y, z = c * cy, c * sy, -c * cy, c * sy  # on the back, as the task starts a face-up robot
    roll = torch.empty(n, device=dev).uniform_(*side_roll) * torch.where(torch.rand(n, device=dev) < 0.5, 1.0, -1.0)
    ct, st = torch.cos(roll / 2), torch.sin(roll / 2)  # about the spine (body z), right-multiplied
    env.sim.data.qpos[s, 3:7] = torch.stack([w * ct - z * st, x * ct + y * st, y * ct - x * st, w * st + z * ct], dim=1)
    env.sim.data.qvel[s, :6] = 0.0


# ── Config ────────────────────────────────────────────────────────────────────

def make_microcluck_recover_env_cfg(play: bool = False, stress: Stress | None = TRAIN_STRESS):
    cfg = make_microcluck_standup_env_cfg(play=play, face_up_only=True)
    cfg.episode_length_s = EPISODE_S
    cfg.curriculum.clear()
    cfg.events.pop("push_robot", None)

    cfg.actions[ACTION] = ResidualJointPositionActionCfg(entity_name="robot", actuator_names=(".*",), scale=RESIDUAL_SCALE, use_default_offset=True)

    cfg.events["set_ground_state"] = EventTermCfg(func=set_recovery_start, mode="reset")  # keeps the stock event's place in the order

    actor, critic = cfg.observations["actor"].terms, cfg.observations["critic"].terms
    for group in (actor, critic):
        for name in ("command", "head_command", "body_command"):
            group.pop(name, None)
        group["reference"] = ObservationTermCfg(func=reference_offset)
        group["phase"] = ObservationTermCfg(func=motion_phase)
        group["chain"] = ObservationTermCfg(func=motion_chain)
    critic["com_shift"] = ObservationTermCfg(func=trunk_com_shift)
    critic["added_mass"] = ObservationTermCfg(func=trunk_added_mass)

    cfg.rewards = {
        "progress": RewardTermCfg(func=recovery_progress, weight=1.0),
        "up": RewardTermCfg(func=recovery_up, weight=1.0),
        "residual_l2": RewardTermCfg(func=residual_l2, weight=-0.01),
        "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.02),
        "dof_pos_limits": cfg.rewards["dof_pos_limits"],
    }
    if stress is not None:
        apply_stress(cfg, stress)  # after the stock events, so the centre-of-mass shift lands on top of the task's own randomization
    return cfg


def make_microcluck_recover_rl_cfg(run_name: str = "recover"):
    """The stock stand-up PPO recipe, with a small action noise to start from (the reference already works), no entropy bonus and outputs
    clipped to +-1. The first run (resid_v1: noise 0.3, bonus 0.005, centre of mass +-12 mm fore-aft) was best at 250 to 500 iterations and
    got worse after: with the bonus and a reward that barely moved, the action noise grew from 0.3 to 1.1 by iteration 700 and the
    in-distribution score fell from 96% to 88% by 750."""
    cfg = deepcopy(_stock.MicroduckStandUpRlCfg)
    cfg.experiment_name = "microcluck_recover"
    cfg.run_name = run_name
    cfg.wandb_project = "mjlab_microcluck"
    cfg.clip_actions = 1.0
    cfg.actor.distribution_cfg = {**cfg.actor.distribution_cfg, "init_std": 0.2}
    cfg.algorithm.entropy_coef = 0.0
    cfg.save_interval = 100
    cfg.max_iterations = 800
    return cfg


MicroCluckRecoverRlCfg = make_microcluck_recover_rl_cfg()
