"""MicroCluck model and task config: structure checks (CPU only, no env instantiation)."""

from pathlib import Path

import mujoco
import pytest

import mjlab_microduck.tasks  # noqa: F401  (registers the tasks)
from mjlab.tasks.registry import list_tasks, load_env_cfg
from mjlab_microduck.robot.microcluck import build_walk_model as build
from mjlab_microduck.robot.microcluck_constants import MICROCLUCK_HOME_FRAME

MODEL_DIR = Path(build.__file__).resolve().parent

LEG_JOINTS = [
    f"{side}_{joint}"
    for side in ("left", "right")
    for joint in ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle")
]


def _joint_names(model):
    return [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]


@pytest.fixture(scope="module")
def walk_model():
    return mujoco.MjModel.from_xml_path(str(MODEL_DIR / "scene_walk.xml"))


def test_ten_leg_servos_and_no_head(walk_model):
    names = [n for n in _joint_names(walk_model) if n != "trunk_base_freejoint"]
    assert sorted(names) == sorted(LEG_JOINTS)
    assert walk_model.nu == 10
    assert not any("neck" in n or "head" in n for n in _joint_names(walk_model))


def test_total_mass_matches_budget(walk_model):
    assert walk_model.body_subtreemass[1] == pytest.approx(build.TARGET_TOTAL_MASS, abs=1e-6)


def test_foot_collision_geoms_keep_their_names(walk_model):
    for name in ("left_foot_collision", "right_foot_collision"):
        assert mujoco.mj_name2id(walk_model, mujoco.mjtObj.mjOBJ_GEOM, name) >= 0
    for name in ("left_foot", "right_foot", "imu"):
        assert mujoco.mj_name2id(walk_model, mujoco.mjtObj.mjOBJ_SITE, name) >= 0


def test_backlash_model_pairs_every_servo_with_a_passive_joint():
    model = mujoco.MjModel.from_xml_path(str(MODEL_DIR / "scene_walk_backlash.xml"))
    names = _joint_names(model)
    assert model.nu == 10
    assert sum(n.startswith("passive_") and n.endswith("_backlash") for n in names) == 10
    assert model.nq == 7 + 20


def test_stand_keyframe_matches_home_frame(walk_model):
    key = mujoco.mj_name2id(walk_model, mujoco.mjtObj.mjOBJ_KEY, "STAND")
    qpos = walk_model.key_qpos[key]
    adr = lambda n: walk_model.jnt_qposadr[mujoco.mj_name2id(walk_model, mujoco.mjtObj.mjOBJ_JOINT, n)]  # noqa: E731
    home = MICROCLUCK_HOME_FRAME.joint_pos
    assert qpos[adr("left_hip_pitch")] == pytest.approx(home[r".*left_hip_pitch.*"])
    assert qpos[adr("right_ankle")] == pytest.approx(home[r".*right_ankle.*"])
    assert qpos[adr("left_hip_roll")] == pytest.approx(home[r".*left_hip_roll.*"])


def test_tasks_registered():
    ids = list_tasks()
    assert "Mjlab-Velocity-Flat-MicroCluck" in ids
    assert "Mjlab-Velocity-Flat-Backlash-MicroCluck" in ids


@pytest.mark.parametrize("task", ["Mjlab-Velocity-Flat-MicroCluck", "Mjlab-Velocity-Flat-Backlash-MicroCluck"])
def test_head_terms_removed_but_command_slots_kept(task):
    cfg = load_env_cfg(task)
    for name in ("head_pose_tracking", "head_pose_bias"):
        assert name not in cfg.rewards
    assert "randomize_head_com" not in cfg.events
    for name in ("head_pose_range", "head_com_range", "head_pose_bias_weight", "body_pose_range"):
        assert name not in cfg.curriculum
    # the observation keeps Pollen's 13-slot command block: twist 3 + head 4 + body 6
    assert "head_command" in cfg.observations["actor"].terms
    assert "body_command" in cfg.observations["actor"].terms
    assert len(cfg.commands["head_pose"].ranges) == 4
    assert len(cfg.commands["body_pose"].ranges) == 6
    assert all(hi <= 0.01 for _, hi in cfg.commands["head_pose"].ranges + cfg.commands["body_pose"].ranges)
