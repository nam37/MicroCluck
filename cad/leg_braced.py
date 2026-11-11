"""MicroPup leg, MicroDuck-braced style: 12 x MG92B, 3 servos per leg (abduction, hip, knee).

After Pollen's MicroDuck leg: the hip and knee servos ride inside the thigh, one above the other along it, held by their
tabs in a printed shell and braced from the other side by a rigidity plate, so the servo cases are part of the
structure. The hip joint is a U-shaped bracket around the top of the thigh: the hip servo's horn is screwed to its outer
arm, and its inner arm carries a 693 bearing on a pin from the thigh shell, behind the hip servo. So the hip servo's
body turns with the thigh, and the joint is held on both sides. The knee is the same: the shank is a fork, its outer
plate on the knee servo's horn and its inner plate on a bearing on a pin from the shell behind the knee servo. The
abduction plates hang from the hip bracket as in the Orion-style leg (leg.py): the horn in front, a bearing behind on a
pin from a cap behind the abduction servo.

The whole leg is only as wide as one servo plus its plates, about 43 mm, against about 69 mm for the Orion-style leg.

Leg frame (front-left leg): x forward, y outward, z up; the hip axis is the y axis; y = 0 on the servos' case bottoms
(the shell's inner side). Angles in the x-z plane from +x toward +z, degrees: the sim's home pose has the thigh at -123.6
and the knee bent q = +67.2.

Usage: cad/.venv/Scripts/python cad/leg_braced.py   (STLs to cad/stl/braced/, assembly to cad/step/braced-home.step)
"""
import math
import sys
from pathlib import Path

from build123d import Compound, Cylinder, Location, Plane, Pos, Rot, Sphere, export_step, export_stl

sys.path.insert(0, str(Path(__file__).parent))
import servo as SV  # noqa: E402
from leg import (BRG_B, BRG_D, BRG_FIT, BRG_RACE_D, CAP_T, CLR, FILL, FOOT_R, L_SHANK, L_THIGH, M2_CLEAR, M2_HEAD_D,  # noqa: E402
                 M25_CLEAR, M25_PILOT, M3_CLEAR, M3_NUT_AF, M3_NUT_T, PLA, Q_HOME, SV_PILOT, T_HOME, WALL, bar,
                 bearing_pocket_x, bearing_pocket_y, bx, cx, cy, hex_x, hex_y, pt)

DRAFT = False
HERE = Path(__file__).parent

# --- thigh shell -----------------------------------------------------------------------------------------------
SHELL_HW = SV.W / 2 + CLR + WALL          # half width across the thigh, 8.7
SHELL_TOP, SHELL_BOT = 2.5, -(L_THIGH + 2.5)   # centres of the shell's rounded ends (radius SHELL_HW)
RIG_Y0 = SV.CASE_TOP + 0.2                # rigidity plate over the servos' case tops
RIG_T = 2.2
RIG_Y1 = RIG_Y0 + RIG_T                   # 27.4; the horns' hubs start at 28.0
BOSS_X = SHELL_HW - 0.2                   # rigidity plate screws: M2 into bosses on the shell's sides
BOSS_R = 2.4
BOSS_Z = (6.0, -L_THIGH / 2, -L_THIGH - 6.0)

# --- arms and forks: same planes as the Orion leg's plates ----------------------------------------------------
IN_Y1 = -CAP_T - 0.4                      # inner arm / inner fork plate, -8.6 .. -3.6, bearing inside
IN_Y0 = IN_Y1 - 5.0
OUT_Y0 = SV.HORN_TOP - SV.HORN_ARM_T - 0.2   # outer arm / outer fork plate on a horn, 30.2 .. 34.2
OUT_Y1 = OUT_Y0 + 4.0
Y_MID = (IN_Y0 + OUT_Y1) / 2              # 12.8, the shank bar and foot centre plane
HUB_R = 6.5
SWEEP_R = math.hypot(SHELL_HW, SHELL_TOP + SHELL_HW) + 0.2   # what the shell's top end sweeps about the hip (and its
                                                              # bottom end about the knee): 11.6 mm
BEAM_Z0 = SWEEP_R + 1.8                   # the bracket's beam over the thigh, 13.4 .. 16.4
BEAM_Z1 = BEAM_Z0 + 3.0
FORK_Z = (-31.0, -26.5)               # shank-local: the fork's bridge, far enough down that the knee folds to 150

# --- abduction (axis along x at (y, z) = (ABD_Y, 0); servo shifted forward so its mount clears the inner arm) -------
ABD_Y = -24.0      # 4 mm further in than the Orion leg: outward abduction then swings the inner arm over the servo
ABD_SERVO_X0 = -8.5
F_FACE = ABD_SERVO_X0 + SV.HORN_TOP - SV.HORN_ARM_T - 0.2    # 21.7
F_OUT = F_FACE + 4.0
ABD_CAP_X0 = ABD_SERVO_X0 - CAP_T                            # -11.7
R_FACE = ABD_CAP_X0 - 0.5                                    # -12.2
R_OUT = R_FACE - 0.5 - BRG_B - 1.0                           # -17.7
ABD_PLATE_Y0, ABD_LOW_Y1 = ABD_Y - 8.0, -12.0
BEAM_Y0 = -2.5      # the beam along x stays outboard of this, so outward abduction can't swing it into the mount
OUT_ARM_SCREWS = (-4.0, 4.0)


def abd_frame():
    """Servo frame onto the abduction servo: shaft along +x at (ABD_Y, 0), body above the shaft."""
    return Location(Plane(origin=(ABD_SERVO_X0, ABD_Y, 0), x_dir=(0, 0, -1), z_dir=(0, -1, 0)))


def hip_servo_loc():
    return Rot(0, -90, 0)                      # servo +x up the thigh: body from -16.9 to +5.9 about the hip


def knee_servo_loc():
    return Pos(0, 0, -L_THIGH) * Rot(0, 90, 0)    # servo +x down: body from the knee up to -28.1


def _stadium(y0, y1, hw=SHELL_HW, top=SHELL_TOP, bot=SHELL_BOT):
    return bar((0, top), (0, bot), 2 * hw, y0, y1)


# ===== parts ======================================================================================================
def shell():
    """Thigh shell: the servos drop in from outside, tabs on its end and middle blocks, M2 tab screws. Pins for the hip
    and knee bearings behind the servos. Thigh-local (pointing down). Print inner face down."""
    s = _stadium(-CAP_T, SV.TAB_UNDER)
    for z in BOSS_Z:
        for sx in (-1, 1):
            s += cy(BOSS_R, -CAP_T, RIG_Y0, sx * BOSS_X, z)
    for loc in (hip_servo_loc(), knee_servo_loc()):
        s -= loc * bx(SV.X0 - CLR, SV.X1 + CLR, 0.0, SV.TAB_UNDER + 1, -SV.W / 2 - CLR, SV.W / 2 + CLR)
        s -= loc * bx(SV.XC - SV.TAB_SPAN / 2 - 0.3, SV.XC + SV.TAB_SPAN / 2 + 0.3, SV.TAB_UNDER, RIG_Y0 + 1,
                      -SV.W / 2 - CLR, SV.W / 2 + CLR)
        for hx in SV.TAB_HOLES_X:
            s -= loc * cy(SV_PILOT / 2, SV.TAB_UNDER - 8, SV.TAB_UNDER + 1, hx, 0)
    for z in BOSS_Z:
        for sx in (-1, 1):
            s -= cy(0.8, RIG_Y0 - 8, RIG_Y0 + 1, sx * BOSS_X, z)
    for z in (0.0, -L_THIGH):     # pins: nut from the servo side, standoff outside for the bearing's inner race
        s -= hex_y(M3_NUT_AF, 0.01, -M3_NUT_T - 0.1, 0, z)
        s -= cy(M3_CLEAR / 2, -CAP_T - 2, 0.1, 0, z)
        s += cy(BRG_RACE_D / 2, -CAP_T + 0.01, IN_Y1 - 0.5, 0, z) - cy(M3_CLEAR / 2, -CAP_T + 1, IN_Y1 - 1, 0, z)
    s -= bx(-3.5, 3.5, -CAP_T - 1, 0.01, -24.0, -21.0)                    # small window between the pins
    # hollow the middle block below the top 7 mm that carries the tab screws (the rounded ends are small already)
    s -= bx(-SV.W / 2 - CLR, SV.W / 2 + CLR, 0.0, SV.TAB_UNDER - 7.0, -L_THIGH - SV.X0 + CLR, SV.X0 - CLR)
    return s


def rigidity_plate():
    """Over the servos' case tops, around their bosses; six M2 x 8 into the shell's bosses. Thigh-local."""
    p = _stadium(RIG_Y0, RIG_Y1)
    for z in BOSS_Z:
        for sx in (-1, 1):
            p += cy(BOSS_R, RIG_Y0, RIG_Y1, sx * BOSS_X, z)
            p -= cy(M2_CLEAR / 2, RIG_Y0 - 1, RIG_Y1 + 1, sx * BOSS_X, z)
    for z in (0.0, -L_THIGH):
        p -= cy(SV.BOSS_R + 0.6, RIG_Y0 - 1, RIG_Y1 + 1, 0, z)
    p -= bx(-4.0, 4.0, RIG_Y0 - 1, RIG_Y1 + 1, -L_THIGH + 14.0, -14.0)       # lighten between the servos
    return p


def bracket():
    """Hip bracket (one piece with the abduction plates): the inner arm with the hip bearing, a beam over the thigh,
    and the front and rear abduction plates down to the axis inside the body. Print the front plate's outer face down."""
    b = cy(HUB_R, IN_Y0, IN_Y1) + bar((0, 0), (0, BEAM_Z1 - 3.0), 2 * HUB_R, IN_Y0, IN_Y1)
    b -= bearing_pocket_y(IN_Y0, +1)
    b += bx(-HUB_R, HUB_R, IN_Y0, OUT_Y0 - 0.01, BEAM_Z0, BEAM_Z1)                   # beam across, to the outer arm
    b += bx(R_OUT, F_OUT, BEAM_Y0, OUT_Y0 - 6.0, BEAM_Z0, BEAM_Z1)                    # beam along, to the plates
    for xa, xb in ((F_FACE, F_OUT), (R_OUT, R_FACE)):
        b += bx(xa, xb, ABD_PLATE_Y0, OUT_Y0 - 6.0, BEAM_Z0, BEAM_Z1)                 # plates' tops, in to the axis
        b += bx(xa, xb, ABD_PLATE_Y0, ABD_LOW_Y1, -9.0, BEAM_Z0 + 0.01)               # plates down inside the body
    for x in OUT_ARM_SCREWS:
        b -= cy(M25_PILOT / 2, OUT_Y0 - 8.0, OUT_Y0 + 0.01, x, (BEAM_Z0 + BEAM_Z1) / 2)
    b -= abd_frame() * SV.horn_pocket(SV.HORN_ARM_L, 180.0, SV.HORN_TOP - SV.HORN_ARM_T - 0.2, SV.HORN_ARM_T + 0.21)
    b -= cx(M2_CLEAR / 2, F_FACE - 1, F_OUT + 1, ABD_Y, 0.0)
    b -= cx(M2_HEAD_D / 2 + 0.2, F_OUT - 1.6, F_OUT + 1, ABD_Y, 0.0)
    for r in SV.HORN_HOLES:
        b -= cx(M2_CLEAR / 2, F_FACE - 1, F_OUT + 1, ABD_Y, r)
    b -= bearing_pocket_x(R_FACE, -1, ABD_Y, 0.0)
    return b


def outer_arm():
    """The bracket's outer arm on the hip horn (its arm pointing up), two M2.5 x 10 into the beam."""
    a = cy(HUB_R, OUT_Y0, OUT_Y1) + bar((0, 0), (0, BEAM_Z1 - 3.0), 2 * HUB_R, OUT_Y0, OUT_Y1)
    a += bx(-HUB_R, HUB_R, OUT_Y0, OUT_Y1, BEAM_Z0, BEAM_Z1)
    a -= SV.horn_pocket(SV.HORN_ARM_L, 90.0, OUT_Y0, SV.HORN_ARM_T + 0.2)
    a -= cy(M2_CLEAR / 2, OUT_Y0 - 1, OUT_Y1 + 1)
    a -= cy(M2_HEAD_D / 2 + 0.2, OUT_Y1 - 1.6, OUT_Y1 + 1)
    for r in SV.HORN_HOLES:
        a -= cy(M2_CLEAR / 2, OUT_Y0 - 1, OUT_Y1 + 1, 0, r)
    for x in OUT_ARM_SCREWS:
        a -= cy(M25_CLEAR / 2, OUT_Y0 - 1, OUT_Y1 + 1, x, (BEAM_Z0 + BEAM_Z1) / 2)
    return a


def shank_outer():
    """Shank fork's outer plate on the knee horn (its arm along the shank). Shank-local. Inner face down."""
    p = cy(HUB_R, OUT_Y0, OUT_Y1) + bar((0, 0), (0, FORK_Z[0] + 2.5), 10.0, OUT_Y0, OUT_Y1)
    p -= SV.horn_pocket(SV.HORN_ARM_L, -90.0, OUT_Y0, SV.HORN_ARM_T + 0.2)
    p -= cy(M2_CLEAR / 2, OUT_Y0 - 1, OUT_Y1 + 1)
    p -= cy(M2_HEAD_D / 2 + 0.2, OUT_Y1 - 1.6, OUT_Y1 + 1)
    for r in SV.HORN_HOLES:
        p -= cy(M2_CLEAR / 2, OUT_Y0 - 1, OUT_Y1 + 1, 0, -r)
    return _fork_screw(p, OUT_Y1, -1)


def shank_inner():
    """Shank fork's inner plate on the bearing behind the knee servo. Shank-local. Pocket side down."""
    p = cy(HUB_R, IN_Y0, IN_Y1) + bar((0, 0), (0, FORK_Z[0] + 2.5), 10.0, IN_Y0, IN_Y1)
    p -= bearing_pocket_y(IN_Y0, +1)
    return _fork_screw(p, IN_Y0, +1)


def _fork_screw(p, y_out, into):
    return p - cy(M25_CLEAR / 2, y_out - into * 1, y_out + into * 6, 0, sum(FORK_Z) / 2)


def shank_body():
    """The fork's bridge, the bar down to the foot and the foot's stub; one M2.5 x 8 from each fork plate. Shank-local.
    Print on its side."""
    zc = sum(FORK_Z) / 2
    b = bx(-5.0, 5.0, IN_Y1, OUT_Y0, FORK_Z[0], FORK_Z[1])
    b += bar((0, zc), (0, -(L_SHANK - 6.0)), 6.0, Y_MID - 3.0, Y_MID + 3.0)
    b += Pos(0, Y_MID, -(L_SHANK - 3.0) - 0.01) * Cylinder(3.0, 6.0)
    b -= cy(M25_PILOT / 2, IN_Y1 - 1, IN_Y1 + 8, 0, zc)
    b -= cy(M25_PILOT / 2, OUT_Y0 - 8, OUT_Y0 + 1, 0, zc)
    b -= bx(-6, 6, IN_Y1 + 9, Y_MID - 4.0, FORK_Z[0] + 1.5, FORK_Z[1] - 1.5)          # lighten
    b -= bx(-6, 6, Y_MID + 4.0, OUT_Y0 - 9, FORK_Z[0] + 1.5, FORK_Z[1] - 1.5)
    return b


def foot():
    f = Pos(0, Y_MID, -L_SHANK) * Sphere(FOOT_R)
    return f - Pos(0, Y_MID, -(L_SHANK - 3.0) + 0.5) * Cylinder(2.9, 7.0)


MOUNT_Y0, MOUNT_Y1 = ABD_Y - SV.W / 2 - CLR - 2.0, ABD_Y + SV.W / 2 + CLR
MOUNT_TOP = -(SV.XC - SV.TAB_SPAN / 2) + 0.5


def abd_mount():
    """Body side, as in leg.py but moved with the abduction servo: tabs, cap with the rear pin's nut, top plate."""
    tab_x = ABD_SERVO_X0 + SV.TAB_UNDER
    z_lo = -(SV.XC + SV.TAB_SPAN / 2) - 0.5
    win = bx(tab_x - 4, tab_x + 1, ABD_Y - SV.W / 2 - CLR, ABD_Y + SV.W / 2 + CLR + 1, -SV.X1 - CLR, -SV.X0 + CLR)
    m = bx(tab_x - 3.0, tab_x, MOUNT_Y0, MOUNT_Y1, z_lo, MOUNT_TOP) - win
    for hx in SV.TAB_HOLES_X:
        m -= cx(SV_PILOT / 2, tab_x - 8, tab_x + 1, ABD_Y, -hx)
    m += bx(ABD_CAP_X0, ABD_SERVO_X0, MOUNT_Y0, MOUNT_Y1, -SV.X1 - CLR - 1.0, MOUNT_TOP)
    m += bx(ABD_CAP_X0, tab_x, MOUNT_Y0, ABD_Y + 3.0, MOUNT_TOP - 0.01, MOUNT_TOP + 3.0)
    m += bx(ABD_CAP_X0, tab_x, MOUNT_Y0, MOUNT_Y0 + 2.0, -SV.X1 - CLR - 1.0, MOUNT_TOP)
    m -= hex_x(M3_NUT_AF, ABD_SERVO_X0 + 0.01, ABD_SERVO_X0 - M3_NUT_T - 0.1, ABD_Y, 0.0)
    m -= cx(M3_CLEAR / 2, ABD_CAP_X0 - 2, ABD_SERVO_X0 + 0.1, ABD_Y, 0.0)
    m += cx(BRG_RACE_D / 2, R_FACE - 0.5, ABD_CAP_X0 + 0.01, ABD_Y, 0.0) - cx(M3_CLEAR / 2, R_FACE - 1, ABD_CAP_X0 + 1, ABD_Y, 0.0)
    for x in (ABD_CAP_X0 + 3.5, tab_x - 6.5):
        m -= Pos(x, ABD_Y - 2.0, MOUNT_TOP) * Cylinder(M25_CLEAR / 2, 8.0)
    return m


# ===== assembly ====================================================================================================
_PARTS = {}
BUILD = {"shell": shell, "rigidity_plate": rigidity_plate, "bracket": bracket, "outer_arm": outer_arm,
         "shank_outer": shank_outer, "shank_inner": shank_inner, "shank_body": shank_body, "foot": foot,
         "abd_mount": abd_mount}
HOLDER_SIDE = ("bracket", "outer_arm", "hip_horn", "abd_servo", "abd_horn", "abd_mount")


def part(name):
    if name not in _PARTS:
        _PARTS[name] = BUILD[name]()
    return _PARTS[name]


_LOCAL = {}


def local_parts():
    """{name: shape} in the frames posed() moves them in (animation export)."""
    if _LOCAL:
        return _LOCAL
    p = _LOCAL
    p.update({n: part(n) for n in BUILD})
    p["hip_servo"] = hip_servo_loc() * SV.body()
    p["knee_servo"] = knee_servo_loc() * SV.body()
    p["abd_servo"] = abd_frame() * SV.body()
    p["hip_horn"] = SV.horn(SV.HORN_ARM_L, 90.0)
    p["knee_horn"] = SV.horn(SV.HORN_ARM_L, -90.0)
    p["abd_horn"] = abd_frame() * SV.horn(SV.HORN_ARM_L, 180.0)
    return p


THIGH_PARTS = ("shell", "rigidity_plate", "hip_servo", "knee_servo")
SHANK_PARTS = ("shank_outer", "shank_inner", "shank_body", "foot", "knee_horn")


def frames(t, q, a):
    """{part: Location} for thigh direction t, knee bend q, abduction a (front-left leg frame)."""
    s = t + q
    kx, kz = pt(L_THIGH, t)
    thigh = Rot(0, -(t + 90), 0)
    shank = Pos(kx, 0, kz) * Rot(0, -(s + 90), 0)
    abd = Pos(0, ABD_Y, 0) * Rot(a, 0, 0) * Pos(0, -ABD_Y, 0)
    out = {}
    for n in local_parts():
        loc = thigh if n in THIGH_PARTS else shank if n in SHANK_PARTS else Location()
        out[n] = loc if n in ("abd_servo", "abd_mount") else abd * loc
    return out


def posed(t=T_HOME, q=Q_HOME, a=0.0, rear=False):
    """Every part placed. rear: the holder side mirrored front to back (the abduction rotation commutes with that)."""
    lp = local_parts()
    return {n: loc * (lp[n].mirror(Plane.YZ) if rear and n in HOLDER_SIDE else lp[n])
            for n, loc in frames(t, q, a).items()}


PRINT = {"shell": Rot(90, 0, 0), "rigidity_plate": Rot(90, 0, 0), "bracket": Rot(0, 90, 0),
         "outer_arm": Rot(-90, 0, 0), "shank_outer": Rot(-90, 0, 0), "shank_inner": Rot(90, 0, 0),
         "shank_body": Rot(0, 90, 0), "foot": Rot(0, 0, 0), "abd_mount": Rot(180, 0, 0)}


def on_bed(name):
    s = PRINT[name] * part(name)
    bb = s.bounding_box()
    return Pos(-(bb.min.X + bb.max.X) / 2, -(bb.min.Y + bb.max.Y) / 2, -bb.min.Z) * s


def main():
    out = HERE / "stl" / ("draft" if DRAFT else "braced")
    out.mkdir(parents=True, exist_ok=True)
    total = 0.0
    for name in PRINT:
        s = on_bed(name)
        export_stl(s, str(out / f"{name.replace('_', '-')}.stl"))
        bb = s.bounding_box()
        g = part(name).volume / 1000 * PLA * FILL
        total += g if name != "abd_mount" else 0.0
        print(f"{name:15s} {bb.size.X:6.1f} x {bb.size.Y:5.1f} x {bb.size.Z:5.1f} mm  {g:5.1f} g")
    print(f"moving printed parts per leg: {total:.1f} g (PLA, {FILL:.0%} fill); leg spans y {IN_Y0 - 2.1:.1f} .. "
          f"{OUT_Y1 + 1.6:.1f} mm from the servos' case bottoms")
    step = HERE / "step"
    step.mkdir(exist_ok=True)
    export_step(Compound(children=list(posed().values())), str(step / "braced-home.step"))
    print("wrote", out, "and", step / "braced-home.step")


if __name__ == "__main__":
    main()
