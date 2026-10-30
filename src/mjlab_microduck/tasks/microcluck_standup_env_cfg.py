"""MicroCluck stand-up (fall recovery) environment.

Pollen's stock StandUp task with the robot swapped for MicroCluck's ground-contact model: 10 leg servos under a screen
torso whose bezel, body shell, comb, tail and wing discs all collide (robot/microcluck/build_standup_model.py). The
question this answers: with no head or neck to pivot on, can a policy learn to get up from lying on the floor?

Everything else is inherited from the stock task (rewards, curricula, domain randomization, BAM actuator model), so the
only differences from the stock duck's stand-up are its body and what is switched off because it has no head:

  - the head_pose command stays in the observation with a tiny sampling range (never delete a slot), its tracking and
    bias rewards, CoM randomization and curricula are removed
  - the body_pose slot is the stock task's zero padding (ENABLE_BODY_CONTROL off), like the firmware sends
  - the stock code is written for the 14-servo layout (0-4 left leg, 5-8 neck and head, 9-13 right leg): the joint
    indices become 0-4 left leg and 5-9 right leg, via module constants of the stock file that are read when the
    config is built
  - the standing height target is MicroCluck's trimmed home height instead of the stock duck's 0.115 m

Three tasks use it. "Prone" trains on face-down and standing starts only, which answers the first question quickly.
"FaceDown" trains on face-down starts only, with a stronger entropy bonus: in the first Prone run the easy standing starts
(30%) rewarded precision, so the one learned action noise per joint, shared by every start, fell from 1.0 to 0.10 and the
policy never explored a get-up motion from face-down (reward flat from iteration 300; it lay still, 8.8 against 8.3 for
zero actions). The full one keeps the stock ground-state curriculum (standing, sitting, face-down, then face-up, ending
at 35% face-up).
"""

from copy import deepcopy

from mjlab.managers import CurriculumTermCfg

from mjlab_microduck.robot.microcluck import build_walk_model as _build
from mjlab_microduck.robot.microcluck_constants import MICROCLUCK_STANDUP_ROBOT_CFG
from mjlab_microduck.tasks import microduck_standup_env_cfg as _stock
from mjlab_microduck.tasks.microcluck_velocity_env_cfg import UNUSED_COMMAND_RANGE

# 10-servo layout: 0-4 left leg (hip yaw, hip roll, hip pitch, knee, ankle), 5-9 right leg.
_LEG_JOINTS = list(range(10))
# The stock sitting keyframe (knees folded, hips leaning forward), re-indexed. It is the hand-off from the sit policy.
_SITTING_JOINT_OVERRIDES = {1: 0.0, 2: -0.4079, 3: 1.35, 4: 0.0, 6: 0.0, 7: 0.4079, 8: -1.35, 9: 0.0}

# Module constants of the stock file that the config builder reads; set while building, then restored.
_STOCK_OVERRIDES = {
    "_LEG_JOINTS": _LEG_JOINTS,
    "SITTING_JOINT_OVERRIDES": _SITTING_JOINT_OVERRIDES,
    "STAND_Z": _build.STAND_TRUNK_Z,
    "ENABLE_BODY_CONTROL": False,
    "ENABLE_HEAD_COM_RANDOMIZATION": False,
}

_HEAD_REWARDS = ("head_pose_tracking", "head_pose_bias")
_HEAD_CURRICULA = ("head_pose_range", "head_com_range", "head_pose_bias_weight")

# Prone mix: no sitting starts (the sit pose of the stock duck is not MicroCluck's), no face-up yet.
PRONE_MIX = {"standing_prob": 0.30, "sitting_prob": 0.0, "face_down_prob": 0.70, "face_up_prob": 0.0}
# Face-down only: nothing easy to exploit, so the shared action noise is only ever rewarded for finding a way up.
FACE_DOWN_MIX = {"standing_prob": 0.0, "sitting_prob": 0.0, "face_down_prob": 1.0, "face_up_prob": 0.0}
# Face-up only (lying on the back, with the stock task's partial roll toward the side unless a script sets it to 0).
FACE_UP_MIX = {"standing_prob": 0.0, "sitting_prob": 0.0, "face_down_prob": 0.0, "face_up_prob": 1.0}
# The stock bonus is 0.01; the FaceDown run triples it to keep the action noise from collapsing again.
FACE_DOWN_ENTROPY_COEF = 0.03


def make_microcluck_standup_env_cfg(
    play: bool = False, rough: bool = False, prone_only: bool = False, face_down_only: bool = False,
    face_up_only: bool = False,
):
    saved = {name: getattr(_stock, name) for name in _STOCK_OVERRIDES}
    for name, value in _STOCK_OVERRIDES.items():
        setattr(_stock, name, value)
    try:
        cfg = _stock.make_microduck_standup_env_cfg(play=play, rough=rough)
    finally:
        for name, value in saved.items():
            setattr(_stock, name, value)

    cfg.scene.entities = {**cfg.scene.entities, "robot": MICROCLUCK_STANDUP_ROBOT_CFG}

    for name in _HEAD_REWARDS:
        cfg.rewards.pop(name, None)
    for name in _HEAD_CURRICULA:
        cfg.curriculum.pop(name, None)
    r = UNUSED_COMMAND_RANGE
    cfg.commands["head_pose"].ranges = tuple((-r, r) for _ in range(4))

    mix = FACE_UP_MIX if face_up_only else FACE_DOWN_MIX if face_down_only else PRONE_MIX if prone_only else None
    if mix is not None:
        cfg.events["set_ground_state"].params.update(mix)
        cfg.curriculum["ground_state_mix"] = CurriculumTermCfg(
            func=cfg.curriculum["ground_state_mix"].func,
            params={"event_name": "set_ground_state", "param_stages": [{"step": 0, "params": dict(mix)}]},
        )
    return cfg


def make_microcluck_standup_rl_cfg(run_name: str, entropy_coef: float | None = None):
    """The stock stand-up PPO recipe with MicroCluck experiment names (and optionally another entropy bonus)."""
    cfg = deepcopy(_stock.MicroduckStandUpRlCfg)
    cfg.experiment_name = "microcluck_standup"
    cfg.run_name = run_name
    cfg.wandb_project = "mjlab_microcluck"
    if entropy_coef is not None:
        cfg.algorithm.entropy_coef = entropy_coef
    return cfg


MicroCluckStandUpRlCfg = make_microcluck_standup_rl_cfg("standup")
MicroCluckProneRlCfg = make_microcluck_standup_rl_cfg("prone")
MicroCluckFaceDownRlCfg = make_microcluck_standup_rl_cfg("facedown", entropy_coef=FACE_DOWN_ENTROPY_COEF)
