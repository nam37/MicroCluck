"""MicroCluck velocity (walking) environment.

Pollen's stock Microduck velocity task with the robot swapped for MicroCluck (10 leg servos, screen
torso, no neck/head). Everything else (rewards, DR, curricula, BAM actuator model, 61-slot-style
observation layout) is inherited from the stock task so the first MicroCluck policy differs from the
stock duck only in its body.

What changes:
  - robot entity -> MICROCLUCK_WALK_ROBOT_CFG (10 actuated joints, trimmed home pose)
  - head rewards (head_pose_tracking, head_pose_bias), head CoM randomization and the head curricula
    are removed: there are no head joints left for them to act on
  - the head_pose (4) and body_pose (6) command slots STAY in the observation, sampled in tiny ranges
    (Pollen's rule: never delete a slot, keep the inputs alive). The firmware always sends zeros there.
    Observation layout: gyro 3, gravity 3, joint pos 10, joint vel 10, last action 10, command 13 = 49.

Not changed yet (S2 work): payload randomization on top of the torso, torso mass/CoM ranges
re-based on measured values, the smaller 256-128-64 actor from the PRD (the stock 512-256-128 is kept
for the first run so the comparison with the stock duck stays clean).
"""

from copy import deepcopy

from mjlab_microduck.robot.microcluck_constants import MICROCLUCK_WALK_ROBOT_CFG
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    MicroduckRlCfg,
    make_microduck_velocity_env_cfg,
)

# Sampling range of the unused command slots (rad / m). Tiny but non-zero, like Pollen's initial
# head/body ranges, so the observation normalizer sees a (near-)constant, well-defined input.
UNUSED_COMMAND_RANGE = 0.001

_HEAD_REWARDS = ("head_pose_tracking", "head_pose_bias")
_HEAD_EVENTS = ("randomize_head_com",)
_HEAD_CURRICULA = ("head_pose_range", "head_com_range", "head_pose_bias_weight", "body_pose_range")


def make_microcluck_velocity_env_cfg(play: bool = False, rough: bool = False):
    cfg = make_microduck_velocity_env_cfg(play=play, rough=rough)

    cfg.scene.entities = {**cfg.scene.entities, "robot": MICROCLUCK_WALK_ROBOT_CFG}

    for name in _HEAD_REWARDS:
        cfg.rewards.pop(name, None)
    for name in _HEAD_EVENTS:
        cfg.events.pop(name, None)
    for name in _HEAD_CURRICULA:
        cfg.curriculum.pop(name, None)

    # Keep the command slots alive but practically constant (see module docstring).
    r = UNUSED_COMMAND_RANGE
    cfg.commands["head_pose"].ranges = tuple((-r, r) for _ in range(4))
    cfg.commands["body_pose"].ranges = tuple((-r, r) for _ in range(6))
    return cfg


def make_microcluck_rl_cfg():
    """Stock PPO recipe (512-256-128 actor) with MicroCluck experiment names."""
    cfg = deepcopy(MicroduckRlCfg)
    cfg.experiment_name = "microcluck_velocity"
    cfg.run_name = "microcluck"
    cfg.wandb_project = "mjlab_microcluck"
    return cfg


MicroCluckRlCfg = make_microcluck_rl_cfg()
