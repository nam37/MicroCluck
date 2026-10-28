"""MicroCluck model and task config: structure checks (CPU only, no env instantiation)."""

from pathlib import Path

import mujoco
import numpy as np  # noqa: F401
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


# --- torso layout: battery, electronics and the shell are drawn; geometry matches hardware/torso.scad ---

TORSO_SCAD = build.ASSETS / "torso.scad"  # copied from the MicroCluck repo by hardware/tools/build_shell.sh


def _scad_vec(text, name):
    import re

    m = re.search(rf"\b{name}\s*=\s*\[([^\]]*)\]", text)
    assert m, f"{name} not found in torso.scad"
    return [float(v) for v in m.group(1).split(",")]


def test_battery_and_electronics_are_drawn(walk_model):
    for name in ("battery_2s_visual", "electronics_visual", "bezel_and_bumpers_visual", "tail_visual"):
        gid = mujoco.mj_name2id(walk_model, mujoco.mjtObj.mjOBJ_GEOM, name)
        assert gid >= 0, name
        assert walk_model.geom_contype[gid] == 0 and walk_model.geom_conaffinity[gid] == 0  # visual only


@pytest.mark.skipif(not (build.ASSETS / "shell.stl").exists(), reason="shell STL not built (hardware/tools/build_shell.sh)")
def test_shell_is_a_visual_only_mesh(walk_model):
    gid = mujoco.mj_name2id(walk_model, mujoco.mjtObj.mjOBJ_GEOM, "shell_visual")
    assert gid >= 0
    assert walk_model.geom_contype[gid] == 0 and walk_model.geom_conaffinity[gid] == 0
    assert walk_model.body_subtreemass[1] == pytest.approx(build.TARGET_TOTAL_MASS, abs=1e-6)  # no mass added


@pytest.mark.skipif(not TORSO_SCAD.exists(), reason="torso.scad not copied (hardware/tools/build_shell.sh)")
def test_torso_scad_layout_matches_the_sim_parts_table():
    text = TORSO_SCAD.read_text()
    scad_name = {
        "chassis_block": "chassis", "hip_yaw_servos": "hipservos", "freenove_board": "board",
        "bezel_and_bumpers": "bezel", "battery_2s": "battery", "electronics": "electronics",
        "comb": "comb", "tail": "tail",
    }
    for part in build.PARTS:
        key = scad_name[part.name]
        pos = np.array(_scad_vec(text, f"{key}_pos")) / 1000.0
        size = np.array(_scad_vec(text, f"{key}_size")) / 1000.0
        assert pos == pytest.approx(np.array(part.pos), abs=1e-6), part.name
        assert size == pytest.approx(np.array(part.size), abs=1e-6), part.name


# --- toes variant: chunky toes fused onto Pollen's sole, counted as collision --------------------------

TOES_BUILT = all((build.ASSETS / n).exists() for n in build.TOE_STLS)


def _foot_extent_at_stand(scene, geom_name):
    """x extent (relative to the ankle axis) and lowest world z of a mesh geom in the STAND key."""
    model = mujoco.MjModel.from_xml_path(str(MODEL_DIR / scene))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "STAND"))
    mujoco.mj_forward(model, data)
    out = {}
    for side in ("left", "right"):
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name.format(side=side))
        assert gid >= 0
        mid = model.geom_dataid[gid]
        v0, n = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        world = model.mesh_vert[v0:v0 + n] @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid]
        ank = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"ankle_{side}")
        out[side] = (world - data.xpos[ank], float(world[:, 2].min()))
    return model, out


@pytest.mark.skipif(not TOES_BUILT, reason="toe STLs not built (hardware/tools/build_toes.sh)")
@pytest.mark.parametrize("scene", ["scene_walk_toes.xml", "scene_walk_backlash_toes.xml"])
def test_toes_variant_extends_the_contact_patch_forward_on_the_same_ground_plane(scene):
    _, plain = _foot_extent_at_stand("scene_walk.xml", "{side}_foot_collision")
    model, toes = _foot_extent_at_stand(scene, "{side}_foot_collision")
    assert model.nu == 10
    assert model.body_subtreemass[1] == pytest.approx(build.TARGET_TOTAL_MASS, abs=1e-6)
    for side in ("left", "right"):
        rel, zmin = toes[side]
        plain_rel, plain_zmin = plain[side]
        assert plain_rel[:, 0].max() == pytest.approx(0.034, abs=0.001)  # Pollen's front edge
        assert rel[:, 0].max() == pytest.approx(0.046, abs=0.0015)  # the middle toe tip, 12 mm further
        assert rel[:, 0].min() == pytest.approx(plain_rel[:, 0].min(), abs=0.0005)  # heel unchanged
        assert zmin == pytest.approx(plain_zmin, abs=1e-4)  # same ground plane
        # the toes also have a visual mesh with no contact
        visual = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_toes_visual")
        assert visual >= 0 and model.geom_contype[visual] == 0 and model.geom_conaffinity[visual] == 0


def test_plain_model_has_no_toes(walk_model):
    for side in ("left", "right"):
        assert mujoco.mj_name2id(walk_model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_toes_visual") < 0
