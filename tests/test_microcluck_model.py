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
    """The geom has no contact and no mass of its own; the shell's mass is in the trunk inertial (next test)."""
    gid = mujoco.mj_name2id(walk_model, mujoco.mjtObj.mjOBJ_GEOM, "shell_visual")
    assert gid >= 0
    assert walk_model.geom_contype[gid] == 0 and walk_model.geom_conaffinity[gid] == 0
    assert walk_model.body_subtreemass[1] == pytest.approx(build.TARGET_TOTAL_MASS, abs=1e-6)


@pytest.mark.skipif(not (build.ASSETS / "wings.stl").exists(), reason="no wing discs built (egg-style shell)")
def test_wing_discs_are_a_visual_only_mesh(walk_model):
    gid = mujoco.mj_name2id(walk_model, mujoco.mjtObj.mjOBJ_GEOM, "wings_visual")
    assert gid >= 0
    assert walk_model.geom_contype[gid] == 0 and walk_model.geom_conaffinity[gid] == 0
    assert walk_model.body_subtreemass[1] == pytest.approx(build.TARGET_TOTAL_MASS, abs=1e-6)


def _box_triangles(lo, hi):
    """The 12 outward-facing triangles of an axis-aligned box."""
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    v = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]])
    faces = [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4),
             (1, 2, 6), (1, 6, 5), (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7)]
    return v[np.array(faces)]


def test_mesh_mass_properties_match_a_solid_box():
    """A 20 x 30 x 40 mm box of 1000 kg/m^3 away from the origin: mass, CoM and inertia are known in closed form."""
    lo, hi = np.array([-0.005, -0.018, -0.018]), np.array([0.015, 0.012, 0.022])
    mass, com, inertia = build.mesh_mass_properties(_box_triangles(lo, hi), 1000.0)
    dx, dy, dz = hi - lo
    assert mass == pytest.approx(1000.0 * dx * dy * dz, rel=1e-9)
    assert com == pytest.approx((lo + hi) / 2, abs=1e-12)
    box = mass / 12.0 * np.array([dy * dy + dz * dz, dx * dx + dz * dz, dx * dx + dy * dy])
    assert np.diag(inertia) == pytest.approx(box, rel=1e-9)
    assert np.abs(inertia - np.diag(np.diag(inertia))).max() < 1e-15


@pytest.mark.skipif(not build.SHELL_PARTS, reason="no shell mass in this build (STLs missing or MICROCLUCK_SHELL_MASS=0)")
def test_shell_and_wing_mass_is_in_the_trunk_inertial(walk_model):
    """The shell and wing discs are parts of the combined trunk inertial, with mass and inertia from their meshes,
    and the total moves by what they weigh beyond the 37 g box shell the base budget already held."""
    parts = build.resolve_masses(build.leg_mass(build.SRC_XML))
    mass, com, _ = build.combined_inertial(parts)
    assert walk_model.body_mass[1] == pytest.approx(mass, abs=1e-6)
    assert walk_model.body_ipos[1] == pytest.approx(com, abs=1e-6)
    extra = sum(p.mass for p in build.SHELL_PARTS) - build.SHELL_IN_BASE_MASS
    assert build.TARGET_TOTAL_MASS == pytest.approx(build.BASE_TOTAL_MASS + extra, abs=1e-9)
    by_name = {p.name: p for p in build.SHELL_PARTS}
    shell = by_name["body_shell"]
    assert 0.02 < shell.mass < 0.1  # PETG shell, tens of grams
    assert shell.inertia is not None and np.all(np.linalg.eigvalsh(shell.inertia) > 0)


@pytest.mark.skipif(
    not ((build.ASSETS / "shell_cavity.stl").exists() and (build.ASSETS / "shell_envelope.stl").exists()),
    reason="shell STLs not built (hardware/tools/build_shell.sh)",
)
def test_battery_and_electronics_fit_in_the_shell_and_the_wall_holds():
    """The exported cavity holds both blocks, and the printed wall never thins far below the nominal 1.6 mm."""
    import importlib.util

    path = Path(__file__).resolve().parent.parent / "scripts/microcluck_shell_clearance.py"
    spec = importlib.util.spec_from_file_location("microcluck_shell_clearance", path)
    check = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(check)
    res = check.contents_fit(build.ASSETS / "shell_envelope.stl", TORSO_SCAD, 1.6, build.ASSETS / "shell_cavity.stl")
    assert res["battery"] >= 0 and res["electronics"] >= 0, res
    assert res.get("tail", 0) >= 0, res  # only measured when the body encloses the tail block
    assert res["wall"] >= 1.5, res


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
