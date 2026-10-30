"""Write an experiment variant of a MicroCluck model with the legs mounted higher on the sides of the body.

In the real design the hip yaw axes sit just under the trunk's base frame (z = -5 mm, y = +-17.5 mm) while the body's
centre of mass is 54 mm above that frame, so lying on its front the legs can only reach behind the weight. A bird's legs
attach at the sides, level with the body. This moves each leg chain up by --dz and out by --dy (whole chain, rigidly).
Joint order, ranges, masses and the home pose are unchanged, so the same tasks and scripts run on it.

  --base standup (default): the stand-up model; the wing collision discs move up by --wing-dz so they do not sit on the
      hips. Use it with MICROCLUCK_STANDUP_XML (stand-up tasks, scripts/standup_search.py, standup_reach.py).
  --base walk: the walking model; the standing trunk is then dz lower, so trunk_base drops by dz and the wing visual
      moves up by --wing-dz. Use it with MICROCLUCK_WALK_XML and MICROCLUCK_LEG_DZ=<dz> (the walking task's spawn height
      is shifted by the same amount), then train Mjlab-Velocity-Flat-MicroCluck as usual.

    uv run python scripts/make_sidehip_variant.py --dz 0.07 --dy 0.025 --wing-dz 0.045 --name sidehip_b
    uv run python scripts/make_sidehip_variant.py --base walk --dz 0.07 --dy 0.025 --wing-dz 0.045 --name sidehip_b
    MICROCLUCK_WALK_XML=$PWD/src/mjlab_microduck/robot/microcluck/robot_walk_sidehip_b.xml MICROCLUCK_LEG_DZ=0.07 \
        uv run train Mjlab-Velocity-Flat-MicroCluck ...

  --clean: for the hips moved up into the body. The two hip-yaw servos (visual and their 36 g in the trunk inertial) move with
      the leg chain instead of staying at the old height, the power_support plate is removed, and the hip brackets may sit
      inside the shell (a shell opening is assumed; the thighs and shanks still collide with it).
  --thigh-collision: also collide the thigh (upper-leg mesh and its two servos) with the floor and the torso.

    uv run python scripts/make_sidehip_variant.py --dz 0.07 --dy 0 --wing-dz 0.045 --clean --thigh-collision --name centred_thigh

The variant is written next to its source XML (its asset paths are relative to that folder). Distances in metres.
"""

import argparse
import re
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--base", choices=["standup", "walk"], default="standup")
ap.add_argument("--dz", type=float, default=0.07, help="move the leg mounts up the body")
ap.add_argument("--dy", type=float, default=0.025, help="move the leg mounts outward")
ap.add_argument("--wing-dz", type=float, default=0.045, help="move the wings up so they do not sit on the hips")
ap.add_argument("--name", default="sidehip")
ap.add_argument("--clean", action="store_true",
                help="tidy the model for the moved hips: the two hip-yaw servos (visual and mass) move with the leg chain, "
                     "the power_support plate is removed, and the hip brackets may sit inside the shell (the shell is taken to "
                     "have an opening there; the thighs and shanks still collide with it)")
ap.add_argument("--tail-box", type=float, nargs=5, metavar=("HX", "HY", "HZ", "CX", "CZ"), default=None,
                help="replace the tail placeholder (visual and collider) by a box with these half sizes and this centre, metres, "
                     "trunk frame. The tail's 14 g stays in the trunk inertial where it was (a 10 mm move is 0.2 mm of centre of mass)")
ap.add_argument("--height-ref", action="store_true",
                help="add a massless body 'height_ref' near the top of the trunk (x 20 mm, z 120 mm). With the legs raised the trunk "
                     "frame is LOWER standing than lying on the back, so height rewards written for the trunk frame pull the wrong "
                     "way; the stand-up task measures height at this point instead (MICROCLUCK_HEIGHT_REF=height_ref)")
ap.add_argument("--move-chassis", action="store_true",
                help="with --clean: also move Pollen's chassis plate (visual and its 50 g placeholder mass) with the leg chain; "
                     "without it the plate and mass stay at the old low position, hanging below the body between the legs")
ap.add_argument("--dx", type=float, default=0.0,
                help="move the leg mounts along the body (negative = back, away from the screen); stand-up base only, the walking "
                     "home pose is not re-trimmed for it")
ap.add_argument("--leg-collision", action="store_true",
                help="collide every leg part in the simulation: the thigh (as --thigh-collision), the shank with its servo, and "
                     "the hip brackets. mjlab only collides geoms named *_collision and switches every other geom off, and the "
                     "stock stand-up model leaves the shank and hip-bracket colliders unnamed, so they never touch anything")
ap.add_argument("--thigh-collision", action="store_true",
                help="also collide the thigh (the upper-leg mesh and its two servos, as convex hulls) with the floor and the "
                     "torso; the stock stand-up model leaves them visual-only")
a = ap.parse_args()

folder = Path(__file__).resolve().parents[1] / "src/mjlab_microduck/robot/microcluck"
text = (folder / f"robot_{a.base}.xml").read_text()


def shift_body(text: str, body: str, dy: float, dz: float, dx: float = 0.0) -> str:
    pattern = re.compile(rf'(<body name="{body}" pos=")([-0-9.e]+) ([-0-9.e]+) ([-0-9.e]+)(")')
    m = pattern.search(text)
    assert m, body
    x, y, z = float(m[2]), float(m[3]), float(m[4])
    return pattern.sub(rf"\g<1>{x + dx:.6g} {y + dy:.6g} {z + dz:.6g}\g<5>", text, count=1)


YAW_SERVO_MASS = 0.018  # one XL330; the two hip-yaw servos are part of the trunk's mass
YAW_SERVO_XYZ = (-0.001, 0.018, 0.011)  # centre of the left one in the trunk frame (the right one mirrors y)


CHASSIS_MASS = 0.050  # build_walk_model.py's placeholder for Pollen's chassis between the hip servos
CHASSIS_XYZ = (0.004, 0.0, 0.0)  # its centre in the trunk frame


def move_yaw_servos(text: str, dy: float, dz: float, dx: float = 0.0, chassis: bool = False) -> str:
    """The stock trunk holds the two hip-yaw servos above the leg chain; moving the chain leaves them behind. Move their
    visuals with the chain and shift their mass in the trunk inertial (point masses, parallel-axis theorem). With
    ``chassis``, Pollen's chassis plate (the structure the hip servos bolt to, a 50 g placeholder mass in the inertial)
    moves with the chain too."""
    pattern = re.compile(r'(<geom type="mesh" class="visual" pos=")0\.006 (-?0\.0175) ([-0-9.e]+)'
                         r'(" quat="[^"]*" mesh="(?:xl330|seeed_bearing__configuration__22x16x4)")')
    n = 0

    def shift(m):
        nonlocal n
        n += 1
        y = float(m[2])
        return f"{m[1]}{0.006 + dx:.6g} {y + (dy if y > 0 else -dy):.6g} {float(m[3]) + dz:.6g}{m[4]}"

    text = pattern.sub(shift, text)
    assert n == 4, f"expected 2 yaw servos + 2 bearings in the trunk, found {n} geoms"
    if chassis:
        plate = re.compile(r'(<geom type="mesh" class="visual" pos=")([-0-9.e]+) ([-0-9.e]+) ([-0-9.e]+)'
                           r'(" quat="[^"]*" mesh="trunk_base" material=)')
        pm = plate.search(text)
        assert pm, "chassis plate visual"
        text = plate.sub(lambda m: f"{m[1]}{float(m[2]) + dx:.6g} {m[3]} {float(m[4]) + dz:.6g}{m[5]}", text, count=1)

    start = text.index('<body name="trunk_base"')  # the first inertial after it is the trunk's own
    m = re.compile(r'(<inertial pos=")([-0-9.e ]+)(" mass=")([-0-9.e]+)(" fullinertia=")([-0-9.e ]+)("/>)').search(text, start)
    assert m, "trunk inertial"
    c = np.array([float(v) for v in m[2].split()])
    mass = float(m[4])
    ixx, iyy, izz, ixy, ixz, iyz = (float(v) for v in m[6].split())
    inertia_c = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])
    eye = np.eye(3)
    steer = lambda p: np.dot(p, p) * eye - np.outer(p, p)  # parallel-axis term per unit mass
    inertia_o = inertia_c + mass * steer(c)  # about the trunk frame origin
    new_c = c.copy()
    for sy in (+1, -1):
        old = np.array([YAW_SERVO_XYZ[0], sy * YAW_SERVO_XYZ[1], YAW_SERVO_XYZ[2]])
        new = old + np.array([dx, sy * dy, dz])
        inertia_o += YAW_SERVO_MASS * (steer(new) - steer(old))
        new_c += YAW_SERVO_MASS * (new - old) / mass
    moved = 2 * YAW_SERVO_MASS
    if chassis:
        old = np.array(CHASSIS_XYZ)
        new = old + np.array([dx, 0.0, dz])
        inertia_o += CHASSIS_MASS * (steer(new) - steer(old))
        new_c += CHASSIS_MASS * (new - old) / mass
        moved += CHASSIS_MASS
    inertia_new = inertia_o - mass * steer(new_c)
    full = [inertia_new[0, 0], inertia_new[1, 1], inertia_new[2, 2], inertia_new[0, 1], inertia_new[0, 2], inertia_new[1, 2]]
    text = text.replace(m[0], f"{m[1]}{' '.join(f'{v:.10g}' for v in new_c)}{m[3]}{m[4]}{m[5]}"
                              f"{' '.join(f'{v:.10g}' for v in full)}{m[7]}", 1)
    print(f"{'yaw servos and chassis' if chassis else 'yaw servos'} moved ({1000 * moved:.0f} g): trunk centre of mass "
          f"{np.round(1000 * (new_c - c), 2)} mm, whole-robot shift about {1000 * moved * dz / 0.650:.1f} mm in z, "
          f"{1000 * moved * dx / 0.650:.1f} mm in x")
    return text


def drop_plate(text: str) -> str:
    """Remove the power_support plate (visual and self-collision geoms); its mass stays in the trunk inertial."""
    text, n = re.subn(r'[ \t]*<geom [^>]*mesh="power_support"[^>]*/>\n', "", text)
    assert n == 2, n
    return text.replace("      <!-- Part power_support -->\n", "", 1)


def hips_inside_shell(text: str) -> str:
    """Let the hip brackets sit inside the shell: they are the only leg parts at the shell's width. The thighs, shanks and
    soles keep colliding with it."""
    exclude = ('  <contact>\n    <exclude body1="trunk_base" body2="hip_l"/>\n    <exclude body1="trunk_base" body2="hip_l_2"/>\n'
               '  </contact>\n')
    assert "<exclude body1=\"trunk_base\"" not in text
    return text.replace("  <asset>", exclude + "  <asset>", 1)


def thigh_colliders(text: str) -> str:
    """Duplicate the thigh's visual meshes (upper-leg mesh and its servos) as collision geoms named *_collision."""
    out, side, count = [], None, {"left": 0, "right": 0}
    for line in text.splitlines(keepends=True):
        m = re.search(r'<body name="upper_leg_(left|right)"', line)
        if m:
            side = m[1]
        elif "<body name=" in line:
            side = None  # the shank starts: the thigh's own geoms are done
        out.append(line)
        if side and 'class="visual"' in line and re.search(r'mesh="(upper_leg_(left|right)|xl330)"', line):
            count[side] += 1
            name = f"thigh_{side}_{count[side]}_collision"
            out.append(line.replace('class="visual"', f'class="collision" name="{name}"'))
    assert count["left"] == count["right"] == 3, count  # the upper-leg mesh and two servos on each side
    return "".join(out)


def height_ref(text: str) -> str:
    """A static, massless (0.1 g) body fixed to the trunk, near the top of the body, for height rewards."""
    pattern = re.compile(r'(\n)([ \t]*)(<body name="yaw2roll")')
    m = pattern.search(text)
    assert m, "yaw2roll body"
    ind = m[2]
    block = (f'{ind}<body name="height_ref" pos="0.02 0 0.12">\n'
             f'{ind}  <inertial pos="0 0 0" mass="0.0001" diaginertia="1e-10 1e-10 1e-10"/>\n'
             f'{ind}</body>\n')
    return text[:m.start(2)] + block + text[m.start(2):]


def tail_box(text: str, hx: float, hy: float, hz: float, cx: float, cz: float) -> str:
    n = 0
    for name in ("tail_visual", "tail_collision"):
        pattern = re.compile(rf'(name="{name}" size=")[-0-9.e ]+(" pos=")[-0-9.e ]+(")')
        if pattern.search(text):
            text = pattern.sub(rf"\g<1>{hx:.6g} {hy:.6g} {hz:.6g}\g<2>{cx:.6g} 0 {cz:.6g}\g<3>", text, count=1)
            n += 1
    assert n >= 1, "no tail geom found"
    return text


def leg_colliders(text: str) -> str:
    """Make the shanks, their servos and the hip brackets real colliders. The stand-up model already has a collider on each
    shank mesh and each hip bracket, but unnamed, so mjlab's `.*_collision` rule switches them off; name them, and add a
    collider for the servo on each shank."""
    out, body, n = [], None, 0
    for line in text.splitlines(keepends=True):
        m = re.search(r'<body name="([^"]+)"', line)
        if m:
            body = m[1]
        if body in ("leg", "leg_2") and 'class="visual"' in line and 'mesh="xl330"' in line:
            out.append(line)
            n += 1
            out.append(line.replace('class="visual"', f'class="collision" name="{body}_servo_collision"'))
            continue
        if 'class="collision"' in line and "name=" not in line and re.search(r'mesh="(leg|hip_l)"', line):
            n += 1
            line = line.replace('class="collision"', f'class="collision" name="{body}_{re.search(r"mesh=.(leg|hip_l)", line)[1]}_collision"', 1)
        out.append(line)
    assert n == 6, f"expected 2 shank meshes, 2 shank servos and 2 hip brackets, got {n}"
    return "".join(out)


# the left leg chain starts at yaw2roll (y > 0), the right at bearing_roll (y < 0)
text = shift_body(text, "yaw2roll", +a.dy, a.dz, a.dx)
text = shift_body(text, "bearing_roll", -a.dy, a.dz, a.dx)
if a.clean:
    text = move_yaw_servos(text, a.dy, a.dz, a.dx, chassis=a.move_chassis)
    text = drop_plate(text)
    text = hips_inside_shell(text)
if a.tail_box:
    text = tail_box(text, *a.tail_box)
if a.height_ref:
    text = height_ref(text)
if a.thigh_collision or a.leg_collision:
    text = thigh_colliders(text)
if a.leg_collision:
    text = leg_colliders(text)

if a.base == "standup":
    for side in ("left", "right"):
        pattern = re.compile(rf'(name="wing_{side}_collision"[^>]*? pos=")([-0-9.e]+) ([-0-9.e]+) ([-0-9.e]+)(")')
        m = pattern.search(text)
        assert m, side
        text = pattern.sub(rf"\g<1>{float(m[2]):.6g} {float(m[3]):.6g} {float(m[4]) + a.wing_dz:.6g}\g<5>", text, count=1)
else:
    # the standing trunk is dz lower: drop trunk_base by dz (the body is placed so the soles just touch the floor)
    m = re.search(r'(<body name="trunk_base" pos="0 0 )([-0-9.e]+)(")', text)
    assert m, "trunk_base"
    text = text.replace(m[0], f"{m[1]}{float(m[2]) - a.dz:.6g}{m[3]}", 1)
    m = re.search(r'(name="wings_visual" mesh="mc_wings" pos="0 0 )([-0-9.e]+)(")', text)
    assert m, "wings_visual"
    text = text.replace(m[0], f"{m[1]}{float(m[2]) + a.wing_dz:.6g}{m[3]}", 1)

text = text.replace("<!-- MicroCluck ", f"<!-- SIDE-HIP VARIANT {a.name}: legs +{1000 * a.dz:.0f} mm up, +{1000 * a.dy:.0f} mm out, "
                    f"wings +{1000 * a.wing_dz:.0f} mm up. MicroCluck ", 1)
out = folder / f"robot_{a.base}_{a.name}.xml"
out.write_text(text)
print("wrote", out)
