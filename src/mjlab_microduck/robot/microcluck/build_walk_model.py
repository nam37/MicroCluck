#!/usr/bin/env python3
"""Build MicroCluck's walking model from Pollen's `microduck/robot_walk.xml`.

MicroCluck keeps Pollen's legs (meshes, joints, ranges, hip spacing) and replaces everything above
the hips: the neck/head chain and the duck's trunk shell are removed, and a placeholder torso
(Freenove screen board, bezel, battery, electronics, comb, tail) is added as ONE rigid trunk with
an explicit inertial. Pollen's CAD is not published, so this is a text edit of the exported XML,
not a re-export.

Everything marked PLACEHOLDER below is an estimate from the PRD mass budget and must be replaced
with weighed / CAD values (PRD section 6.5 and H3). Run from anywhere:

    uv run python src/mjlab_microduck/robot/microcluck/build_walk_model.py

It writes `robot_walk.xml` (+ `robot_walk_backlash.xml`) and the two scene files next to itself.
Meshes are read from `../microduck/assets` (CC BY-NC-SA 4.0, see LICENSE-MODELS.md).
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
SRC_DIR = HERE.parent / "microduck"
SRC_XML = SRC_DIR / "robot_walk.xml"
# The toes variant fuses three toes onto Pollen's sole. Its binary STLs (the toes, and the convex hull of
# sole + toes for collision) are written by hardware/tools/build_toes.sh <out> <assets> from the OpenSCAD
# sources in the MicroCluck repo (hardware/params.scad is the single parameter table).
ASSETS = HERE / "assets"
TOE_STLS = [f"{kind}_{side}.stl" for kind in ("toes", "collision_with_toes") for side in ("left", "right")]

# --- PLACEHOLDER budget (PRD 6.5 mass budget, middle of each range, v1 without the claw reserve) ---
# 10 x XL330 180 g + board 60 g + battery 85 g + frame/feet/bumpers 225 g + comb/tail 22 g + electronics 55 g
TARGET_TOTAL_MASS = 0.627  # kg

LEG_BODIES = {
    "yaw2roll", "hip_l", "upper_leg_left", "leg", "ankle_left",
    "bearing_roll", "hip_l_2", "upper_leg_right", "leg_2", "ankle_right",
}
HEAD_BODIES = {"neck", "neck_pitch", "yaw_roll_motion", "jaw_soft"}
HEAD_ACTUATORS = ("neck_pitch", "head_pitch", "head_yaw", "head_roll")
# Pollen trunk visuals that belong to the duck's shell/battery; the chassis, bearings, hip servos
# and power_support (which doubles as the self-collision geom) are kept.
DROP_TRUNK_MESHES = ("right_shell", "left_shell", "banana_pcb_locker", "np_f970")

# Home pose (radians). Hip pitch and ankle are trimmed together by STAND_TRIM so the foot stays flat
# while the whole-body CoM moves over the ankle axis. Pollen's values are HIP_PITCH0 / ANKLE0.
HIP_ROLL0 = 0.08726646259971647
HIP_PITCH0 = 0.457924
KNEE0 = 0.004940
ANKLE0 = 0.452984
# Radians added to |hip_pitch| and |ankle|, solved by trim_home.py (CoM over the ankle axis), and the
# trunk height solved with it so the soles just touch the floor. The env vars are experiment hooks
# only (e.g. MICROCLUCK_STAND_TRIM=0 MICROCLUCK_STAND_TRUNK_Z=0.12 for the untrimmed pose).
STAND_TRIM = float(os.environ.get("MICROCLUCK_STAND_TRIM", "0.082720"))
STAND_TRUNK_Z = float(os.environ.get("MICROCLUCK_STAND_TRUNK_Z", "0.11776"))


@dataclass
class Part:
    name: str
    mass: float | None  # kg; None = remainder so the total hits TARGET_TOTAL_MASS
    size: tuple[float, float, float]  # full box size (x, y, z), m
    pos: tuple[float, float, float]  # box center in the trunk_base frame (x forward, y left, z up), m
    note: str
    rgba: str | None = None  # draw a visual box when set


# Trunk frame: x forward, y left, z up; Pollen's hip axes sit at x=0.006, z=-0.005 .. 0.
PARTS = [
    Part("chassis_block", 0.050, (0.040, 0.060, 0.032), (0.004, 0.0, 0.0),
         "PLACEHOLDER mass: Pollen's chassis between the hip servos (their mesh stays as a visual)"),
    Part("hip_yaw_servos", 0.036, (0.024, 0.050, 0.022), (0.006, 0.0, 0.0115),
         "2 x XL330 at 18 g; drawn by Pollen's meshes"),
    Part("freenove_board", 0.060, (0.010, 0.056, 0.098), (0.030, 0.0, 0.075),
         "PLACEHOLDER: WEIGH IT and measure it (mass and size are guesses)"),
    Part("bezel_and_bumpers", None, (0.014, 0.066, 0.108), (0.029, 0.0, 0.075),
         "PLACEHOLDER: remainder of the budget, so the total matches TARGET_TOTAL_MASS", rgba="0.95 0.72 0.02 1"),
    Part("battery_2s", 0.085, (0.019, 0.038, 0.065), (0.0115, 0.0, 0.060),
         "PLACEHOLDER: 2S pack, 70-100 g in the PRD; position behind the screen is a guess", rgba="0.20 0.22 0.30 1"),
    Part("electronics", 0.055, (0.016, 0.046, 0.040), (-0.008, 0.0, 0.055),
         "PLACEHOLDER: INA226, logic buck, buffer board, wiring", rgba="0.12 0.55 0.30 1"),
    Part("comb", 0.008, (0.010, 0.012, 0.020), (0.025, 0.0, 0.139),
         "PLACEHOLDER: snap-on comb on the top boss", rgba="0.84 0.21 0.17 1"),
    Part("tail", 0.014, (0.030, 0.030, 0.070), (-0.045, 0.0, 0.060),
         "PLACEHOLDER: TPU tail; mass only, no collision geom for the walking task", rgba="0.95 0.72 0.02 1"),
]
# Thin dark face on the front of the bezel so the screen reads in renders (visual only, no mass).
SCREEN_FACE = ("0.0005 0.027 0.040", "0.0365 0.0 0.082", "0.02 0.02 0.03 1")


def leg_mass(xml_path: Path) -> float:
    root = ET.parse(xml_path).getroot()
    total = 0.0
    for body in root.iter("body"):
        if body.get("name") in LEG_BODIES:
            inertial = body.find("inertial")
            total += float(inertial.get("mass"))
    return total


def resolve_masses(legs: float) -> list[tuple[Part, float]]:
    fixed = sum(p.mass for p in PARTS if p.mass is not None)
    remainder = TARGET_TOTAL_MASS - legs - fixed
    if remainder <= 0:
        raise SystemExit(f"placeholder masses exceed the budget: remainder {remainder:.4f} kg")
    return [(p, p.mass if p.mass is not None else remainder) for p in PARTS]


def combined_inertial(parts: list[tuple[Part, float]]) -> tuple[float, np.ndarray, np.ndarray]:
    """Total mass, CoM and full inertia about the CoM (boxes + parallel axis theorem)."""
    mass = sum(m for _, m in parts)
    com = sum(m * np.array(p.pos) for p, m in parts) / mass
    inertia = np.zeros((3, 3))
    for p, m in parts:
        dx, dy, dz = p.size
        box = (m / 12.0) * np.diag([dy * dy + dz * dz, dx * dx + dz * dz, dx * dx + dy * dy])
        d = np.array(p.pos) - com
        inertia += box + m * ((d @ d) * np.eye(3) - np.outer(d, d))
    return mass, com, inertia


def fmt(values) -> str:
    return " ".join(f"{v:.8g}" for v in values)


def strip_neck(lines: list[str]) -> list[str]:
    start = next(i for i, l in enumerate(lines) if "<!-- Link neck -->" in l)
    body0 = start + 1
    assert '<body name="neck"' in lines[body0]
    depth = 0
    end = None
    for j in range(body0, len(lines)):
        depth += lines[j].count("<body ") - lines[j].count("</body>")
        if depth == 0:
            end = j
            break
    assert end is not None
    return lines[:start] + lines[end + 1:]


def build() -> None:
    legs = leg_mass(SRC_XML)
    parts = resolve_masses(legs)
    mass, com, inertia = combined_inertial(parts)
    ixx, iyy, izz = inertia[0, 0], inertia[1, 1], inertia[2, 2]
    ixy, ixz, iyz = inertia[0, 1], inertia[0, 2], inertia[1, 2]

    lines = SRC_XML.read_text().splitlines(keepends=True)

    # 1. header + mesh path
    out: list[str] = []
    for l in lines:
        if "Generated using onshape-to-robot" in l:
            out.append("<!-- MicroCluck walking model. Derived from Pollen Robotics' microduck robot_walk.xml\n"
                       "     (onshape-to-robot export; meshes CC BY-NC-SA 4.0). Generated by build_walk_model.py. -->\n")
        elif "cad.onshape.com/documents" in l:
            continue
        elif '<mujoco model="microduck">' in l:
            out.append('<mujoco model="microcluck">\n')
        elif "meshdir=" in l:
            out.append(l.replace('meshdir="assets"', 'meshdir="../microduck/assets"'))
        elif '<body name="trunk_base"' in l:
            # spawn just above the floor at the trimmed home pose (2 mm clearance)
            out.append(l.replace('pos="0 0 0.12"', f'pos="0 0 {STAND_TRUNK_Z + 0.002:.5g}"'))
        else:
            out.append(l)
    lines = out

    # 2. remove the neck / head chain and its four actuators
    lines = strip_neck(lines)
    lines = [l for l in lines
             if not any(f'name="{n}" joint="{n}"' in l for n in HEAD_ACTUATORS)]

    # 3. trunk: drop the duck's shell visuals, swap the inertial, add the torso visuals
    t0 = next(i for i, l in enumerate(lines) if '<body name="trunk_base"' in l)
    first_child = next(i for i in range(t0 + 1, len(lines)) if '<body name="yaw2roll"' in lines[i])
    trunk_block = lines[t0:first_child]
    kept: list[str] = []
    skip_comment = False
    for idx, l in enumerate(trunk_block):
        nxt = trunk_block[idx + 1] if idx + 1 < len(trunk_block) else ""
        if l.lstrip().startswith("<!-- Part") and any(f'mesh="{m}"' in nxt for m in DROP_TRUNK_MESHES):
            continue
        if any(f'mesh="{m}"' in l for m in DROP_TRUNK_MESHES):
            continue
        if "<inertial" in l:
            indent = l[: len(l) - len(l.lstrip())]
            kept.append(f"{indent}<!-- MicroCluck trunk: {mass:.4f} kg, placeholder budget (see build_walk_model.py PARTS) -->\n")
            kept.append(f'{indent}<inertial pos="{fmt(com)}" mass="{mass:.8g}" '
                        f'fullinertia="{fmt([ixx, iyy, izz, ixy, ixz, iyz])}"/>\n')
            continue
        kept.append(l)
    # torso visuals just before the first child link (the imu site line is the last kept line)
    indent = "      "
    kept.append(f"{indent}<!-- MicroCluck torso visuals (class visual: no contact). Masses live in the inertial above. -->\n")
    for p, _ in parts:
        if p.rgba:
            hx, hy, hz = (s / 2 for s in p.size)
            kept.append(f'{indent}<geom type="box" class="visual" name="{p.name}_visual" size="{hx:.5g} {hy:.5g} {hz:.5g}" '
                        f'pos="{fmt(p.pos)}" rgba="{p.rgba}"/>\n')
    kept.append(f'{indent}<geom type="box" class="visual" name="screen_face_visual" size="{SCREEN_FACE[0]}" '
                f'pos="{SCREEN_FACE[1]}" rgba="{SCREEN_FACE[2]}"/>\n')
    lines = lines[:t0] + kept + lines[first_child:]
    lines = add_shell_visual(lines)

    # 4. write the model; if the toe STLs exist, add them as visuals placed from the stand pose of the
    #    model just written, then write it again
    write_variant("", lines)  # plain: Pollen's sole, the default
    if all((ASSETS / n).exists() for n in TOE_STLS):
        write_variant("_toes", toe_lines(lines, foot_placements()))
    else:
        print("toe STLs not found in assets/: skipped the toes variant (run hardware/tools/build_toes.sh <out> <assets>)")

    print(f"legs {legs:.4f} kg (kept), trunk {mass:.4f} kg, total {legs + mass:.4f} kg")
    print(f"trunk CoM in trunk frame: {np.round(com, 4)}")
    for p, m in parts:
        print(f"  {p.name:18s} {m:.4f} kg  {p.note}")


def add_shell_visual(lines: list[str]) -> list[str]:
    """Draw the body shell (hardware/shell.scad) as a translucent visual-only mesh in the trunk frame, so the
    battery and electronics inside stay visible. No contact, no mass: the physics is unchanged."""
    if not (ASSETS / "shell.stl").exists():
        return lines
    out: list[str] = []
    for l in lines:
        if l.strip() == "</asset>":
            out.append('    <mesh name="mc_shell" file="../../microcluck/assets/shell.stl" scale="0.001 0.001 0.001"/>\n')
        out.append(l)
        if 'name="screen_face_visual"' in l:
            indent = l[: len(l) - len(l.lstrip())]
            out.append(f'{indent}<geom type="mesh" class="visual" name="shell_visual" mesh="mc_shell" pos="0 0 0" '
                       f'rgba="0.95 0.72 0.02 0.30"/>\n')
    return out


def write_variant(suffix: str, lines: list[str]) -> None:
    """Write robot_walk{suffix}.xml, its 2-degree backlash twin (Pollen's add_backlash.py) and the scenes."""
    robot = HERE / f"robot_walk{suffix}.xml"
    backlash = HERE / f"robot_walk_backlash{suffix}.xml"
    robot.write_text("".join(lines))
    backlash.write_text(robot.read_text())
    subprocess.run([sys.executable, str(SRC_DIR / "add_backlash.py"), str(backlash), "--backlash-deg", "2.0"], check=True)
    write_scene(HERE / f"scene_walk{suffix}.xml", robot.name, backlash=False, src_name="scene_walk.xml")
    write_scene(HERE / f"scene_walk_backlash{suffix}.xml", backlash.name, backlash=True, src_name="scene_walk_backlash.xml")


def lowest_sole_z(m, d) -> float:
    zmin = 1e9
    for side in ("left", "right"):
        gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_foot_collision")
        mid = m.geom_dataid[gid]
        v0, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
        world = m.mesh_vert[v0:v0 + n] @ d.geom_xmat[gid].reshape(3, 3).T + d.geom_xpos[gid]
        zmin = min(zmin, world[:, 2].min())
    return float(zmin)


def foot_placements() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Pose of the hardware/ foot frame in each ankle body frame.

    The foot is defined in the stand-pose foot frame (world-aligned axes, origin on the ground under the
    ankle axis), so the placement comes from forward kinematics at the home pose of the model just written.
    """
    m = mujoco.MjModel.from_xml_path(str(HERE / "scene_walk.xml"))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "STAND"))
    mujoco.mj_forward(m, d)
    ground = lowest_sole_z(m, d)  # the new sole sits where Pollen's lowest sole point is
    placements = {}
    for side in ("left", "right"):
        ank = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"ankle_{side}")
        rot, pos = d.xmat[ank].reshape(3, 3), d.xpos[ank].copy()
        origin = np.array([pos[0], pos[1], ground])
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, rot.T.flatten())
        placements[side] = (rot.T @ (origin - pos), quat)
    return placements


def toe_lines(lines: list[str], placements) -> list[str]:
    """The toes variant: three chunky toes fused onto Pollen's sole (hardware/toes.scad).

    - a visual mesh for the toes, next to Pollen's sole visual
    - Pollen's named sole collision geom is replaced by the convex hull of sole + toes, same name, so the
      contact sensors and friction rules find it exactly as before
    Both are placed from the stand pose (the toes are defined in the stand-pose foot frame).
    """
    out: list[str] = []
    seen_visual: set[str] = set()
    seen_collision: set[str] = set()
    for l in lines:
        indent = l[: len(l) - len(l.lstrip())]
        for side in ("left", "right"):
            pos, quat = placements[side]
            pose = f'pos="{fmt(pos)}" quat="{fmt(quat)}"'
            if f'name="{side}_foot_collision"' in l:
                seen_collision.add(side)
                out.append(f"{indent}<!-- MicroCluck foot ({side}): convex hull of Pollen's sole + toes, hardware/collision_with_toes.scad -->\n")
                out.append(f'{indent}<geom type="mesh" name="{side}_foot_collision" class="collision" '
                           f'mesh="mc_collision_{side}" {pose} rgba="0.956863 0.686275 0.137255 1"/>\n')
                break
        else:
            out.append(l)
            for side in ("left", "right"):
                if f'mesh="sole_{side}"' in l and 'class="visual"' in l and side not in seen_visual:
                    seen_visual.add(side)
                    pos, quat = placements[side]
                    out.append(f"{indent}<!-- MicroCluck toes ({side}): hardware/toes.scad -->\n")
                    out.append(f'{indent}<geom type="mesh" class="visual" name="{side}_toes_visual" mesh="mc_toes_{side}" '
                               f'pos="{fmt(pos)}" quat="{fmt(quat)}" rgba="0.956863 0.686275 0.137255 1"/>\n')
            if l.strip() == "</asset>":
                out.pop()  # asset lines go before the closing tag
                for s in ("left", "right"):
                    out.append(f'    <mesh name="mc_toes_{s}" file="../../microcluck/assets/toes_{s}.stl" scale="0.001 0.001 0.001"/>\n')
                    out.append(f'    <mesh name="mc_collision_{s}" file="../../microcluck/assets/collision_with_toes_{s}.stl" '
                               f'scale="0.001 0.001 0.001"/>\n')
                out.append(l)
    assert seen_visual == {"left", "right"} and seen_collision == {"left", "right"}, (seen_visual, seen_collision)
    return out


def stand_qpos(backlash: bool) -> str:
    h, k, a = HIP_PITCH0 + STAND_TRIM, KNEE0, ANKLE0 + STAND_TRIM
    left = [0.0, -HIP_ROLL0, -h, -k, a]
    right = [0.0, HIP_ROLL0, h, k, -a]
    if backlash:
        # backlash model: every servo joint is followed by its passive twin (qpos 0)
        left = [v for x in left for v in (x, 0.0)]
        right = [v for x in right for v in (x, 0.0)]
    return " ".join(f"{v:.8g}" for v in left + right)


def stand_ctrl() -> str:
    h, k, a = HIP_PITCH0 + STAND_TRIM, KNEE0, ANKLE0 + STAND_TRIM
    left = [0.0, -HIP_ROLL0, -h, -k, a]
    right = [0.0, HIP_ROLL0, h, k, -a]
    return " ".join(f"{v:.8g}" for v in left + right)


def write_scene(path: Path, robot_file: str, backlash: bool, src_name: str) -> None:
    src = (SRC_DIR / src_name).read_text()
    head = src.split("<keyframe>")[0].replace("robot_walk.xml", robot_file).replace("robot_walk_backlash.xml", robot_file)
    n_joints = 20 if backlash else 10
    zeros = " ".join("0" for _ in range(n_joints))
    key = (
        "    <keyframe>\n"
        f'        <key name="INIT" qpos="0 0 {STAND_TRUNK_Z:.5g} 1 0 0 0 {zeros}" ctrl="{" ".join("0" for _ in range(10))}"/>\n'
        f'        <key name="STAND" qpos="0 0 {STAND_TRUNK_Z:.5g} 1 0 0 0 {stand_qpos(backlash)}" ctrl="{stand_ctrl()}"/>\n'
        "    </keyframe>\n</mujoco>\n"
    )
    path.write_text(head + key)


if __name__ == "__main__":
    build()
