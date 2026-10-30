"""MicroCluck robot configs: Pollen's 10 leg servos under a screen torso (no neck/head).

Same BAM XL330 actuator model as Microduck (``microduck_constants.actuators``), so the sim2real
actuator story is unchanged. The model XML is generated from Pollen's walk model by
``robot/microcluck/build_walk_model.py`` (placeholder masses until the parts are weighed).
"""

import os
from pathlib import Path

import mujoco
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from mjlab_microduck.robot.microcluck import build_walk_model as _build
from mjlab_microduck.robot.microduck_constants import (
    FULL_COLLISION,
    actuators,
    backlash_actuators,
)

_MICROCLUCK_DIR: Path = Path(__file__).resolve().parent / "microcluck"

# MICROCLUCK_FOOT picks the foot at launch: "plain" (default: Pollen's sole) or "toes" (three chunky toes fused
# onto it, counted as collision; hardware/). Same task ids either way:
#   MICROCLUCK_FOOT=toes uv run train Mjlab-Velocity-Flat-MicroCluck ...
_FOOT = os.environ.get("MICROCLUCK_FOOT", "plain")
if _FOOT not in ("plain", "toes"):
    raise ValueError(f"MICROCLUCK_FOOT must be 'plain' or 'toes', got {_FOOT!r}")
_SUFFIX = "" if _FOOT == "plain" else "_toes"

# MICROCLUCK_WALK_XML points the walking tasks at an experiment variant instead (scripts/make_sidehip_variant.py --base walk;
# also set MICROCLUCK_LEG_DZ, see tasks/microcluck_velocity_env_cfg.py).
MICROCLUCK_WALK_XML: Path = (
    Path(os.environ["MICROCLUCK_WALK_XML"]) if "MICROCLUCK_WALK_XML" in os.environ else _MICROCLUCK_DIR / f"robot_walk{_SUFFIX}.xml"
)
MICROCLUCK_WALK_BACKLASH_XML: Path = _MICROCLUCK_DIR / f"robot_walk_backlash{_SUFFIX}.xml"
# Walking model plus ground-contact collision on the legs and torso, for fall recovery (build_standup_model.py).
# MICROCLUCK_STANDUP_XML points the stand-up tasks at an experiment variant instead (scripts/make_sidehip_variant.py).
MICROCLUCK_STANDUP_XML: Path = (
    Path(os.environ["MICROCLUCK_STANDUP_XML"]) if "MICROCLUCK_STANDUP_XML" in os.environ else _MICROCLUCK_DIR / "robot_standup.xml"
)

assert MICROCLUCK_WALK_XML.exists(), f"XML not found: {MICROCLUCK_WALK_XML} (run build_walk_model.py)"
assert MICROCLUCK_WALK_BACKLASH_XML.exists(), f"XML not found: {MICROCLUCK_WALK_BACKLASH_XML}"
assert MICROCLUCK_STANDUP_XML.exists(), f"XML not found: {MICROCLUCK_STANDUP_XML} (run build_standup_model.py)"


def get_microcluck_walk_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(MICROCLUCK_WALK_XML))


def get_microcluck_standup_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(MICROCLUCK_STANDUP_XML))


def get_microcluck_walk_backlash_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(MICROCLUCK_WALK_BACKLASH_XML))


# Home pose: Pollen's STAND pose with hip pitch and ankle trimmed together (foot stays flat) so the
# whole-body CoM sits over the ankle axis like the stock duck's. See robot/microcluck/trim_home.py.
_HIP_PITCH = _build.HIP_PITCH0 + _build.STAND_TRIM
_ANKLE = _build.ANKLE0 + _build.STAND_TRIM

MICROCLUCK_HOME_FRAME = EntityCfg.InitialStateCfg(
    joint_pos={
        r".*hip_yaw.*": 0.0,
        r".*left_hip_roll.*": -_build.HIP_ROLL0,
        r".*right_hip_roll.*": _build.HIP_ROLL0,
        r".*left_hip_pitch.*": -_HIP_PITCH,
        r".*right_hip_pitch.*": _HIP_PITCH,
        r".*left_knee.*": -_build.KNEE0,
        r".*right_knee.*": _build.KNEE0,
        r".*left_ankle.*": _ANKLE,
        r".*right_ankle.*": -_ANKLE,
    },
    joint_vel={".*": 0.0},
)

# Backlash model: the anchored rule pins every passive backlash hinge at 0 first (first match wins).
MICROCLUCK_BACKLASH_HOME_FRAME = EntityCfg.InitialStateCfg(
    joint_pos={r".*_backlash$": 0.0, **MICROCLUCK_HOME_FRAME.joint_pos},
    joint_vel={".*": 0.0},
)

MICROCLUCK_WALK_ROBOT_CFG = EntityCfg(
    spec_fn=get_microcluck_walk_spec,
    init_state=MICROCLUCK_HOME_FRAME,
    collisions=(FULL_COLLISION,),
    articulation=EntityArticulationInfoCfg(
        actuators=(actuators,),
        soft_joint_pos_limit_factor=0.9,
    ),
)

MICROCLUCK_WALK_BACKLASH_ROBOT_CFG = EntityCfg(
    spec_fn=get_microcluck_walk_backlash_spec,
    init_state=MICROCLUCK_BACKLASH_HOME_FRAME,
    collisions=(FULL_COLLISION,),
    articulation=EntityArticulationInfoCfg(
        actuators=(backlash_actuators,),
        soft_joint_pos_limit_factor=0.9,
    ),
)

MICROCLUCK_STANDUP_ROBOT_CFG = EntityCfg(
    spec_fn=get_microcluck_standup_spec,
    init_state=MICROCLUCK_HOME_FRAME,
    collisions=(FULL_COLLISION,),
    articulation=EntityArticulationInfoCfg(
        actuators=(actuators,),
        soft_joint_pos_limit_factor=0.9,
    ),
)
