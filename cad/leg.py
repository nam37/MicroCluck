"""MicroPup leg, Orion style: 12 x MG92B, 3 servos per leg (abduction, hip, knee).

The hip and knee servos sit one above the other in a printed holder, held by their tabs. The holder hangs from the
abduction servo in the body: its horn on the holder's front plate, and a 693 bearing in the holder's rear plate riding on
a pin from a cap behind that servo. The thigh is two plates: the outer one is screwed to the hip servo's horn, the inner
one runs on a 693 bearing on a pin behind the hip servo, so the hip is held on both sides. A bridge joins the plates near
the knee, outside the circle the holder and the servos sweep. The knee is driven Orion-style: a crank on the knee servo,
a short link to a bellcrank on the hip axis, and a long link from the bellcrank down to a lever on the shank. Both links
form parallelograms, so the shank's angle follows the knee servo's angle directly, whatever the hip does. A shoulder
cover holds the bellcrank's axle and a bearing for the knee crank's outer end, so that crank is held on both sides too.

Sizes follow the simulation (study §8, PRD §7): thigh 45 mm, shank 45 mm hip-to-knee and knee-to-foot-centre, 14 mm
foot ball. Placeholder servo and horn sizes are in servo.py; the fit coupon checks them first.

Leg frame (front-left leg, the others mirror it): x forward, y outward, z up. The hip axis is the y axis; y = 0 on the
hip servo's case bottom. Angles in the x-z plane are measured from +x toward +z, in degrees: a thigh pointing straight
down is at -90, and the simulation's home pose has the thigh at -123.6 (leaning back) and the shank at -56.4, a knee bend
q = shank - thigh of +67.2.

Usage: cad/.venv/Scripts/python cad/leg.py
Writes each part in its print orientation to cad/stl/leg/ (cad/stl/draft/ while DRAFT is set) and the assembled leg at
the home pose to cad/step/leg-home.step. The right legs are mirror prints of the left ones; the rear legs use the same
parts with the holder mirrored front to back (see README.md).
"""
import math
import sys
from pathlib import Path

from build123d import (Box, Compound, Cylinder, Location, Plane, Pos, RegularPolygon, Rot, Sphere, export_step,
                       export_stl, extrude)

sys.path.insert(0, str(Path(__file__).parent))
import servo as SV  # noqa: E402

DRAFT = False
HERE = Path(__file__).parent

# --- leg geometry (from the simulation) ----------------------------------------------------------------------------
L_THIGH, L_SHANK, FOOT_R = 45.0, 45.0, 7.0
T_HOME, Q_HOME = -123.6, 67.2      # thigh direction and knee bend at the sim's home pose
Q_RANGE = (20.0, 150.0)            # knee bend the linkage is laid out for (the sim allows 0..149)
ABD_RANGE = (-29.0, 29.0)          # sim: +-0.5 rad

# --- hardware ------------------------------------------------------------------------------------------------------
BRG_D, BRG_B, BRG_RACE_D = 8.0, 4.0, 4.6   # 693ZZ: 3 x 8 x 4 mm; inner race contact diameter
BRG_FIT = 0.0                               # pocket = BRG_D + BRG_FIT; the fit coupon tries -0.1 .. +0.15
M3_CLEAR, M3_PILOT, M3_NUT_AF, M3_NUT_T = 3.3, 2.5, 5.7, 2.5
M2_CLEAR, M2_PILOT, M2_HEAD_D = 2.3, 1.6, 4.2
M25_CLEAR, M25_PILOT = 2.9, 2.4    # 2.4: the DeskSprite screw test's best M2.5 grip (8 Nov 2026)
SV_PILOT = 1.6                      # the servo bag's M2 self-tapping screws, through the tabs

# --- holder (cradle for the hip and knee servos) -----------------------------------------------------------------
CLR, WALL = 0.3, 2.4
KS_Z = 15.0                         # knee servo shaft above the hip servo's (a 2.4 mm divider between them)
CAP_T = 3.2                         # plate behind the servos' bottoms, carries the hip pin's nut
CRADLE_X0 = SV.XC - SV.TAB_SPAN / 2 - 0.5          # rear tab wall's back face, -22.2
CRADLE_X1 = SV.XC + SV.TAB_SPAN / 2 + 0.5          # front tab wall's front face, 11.2
CRADLE_Z0 = -SV.W / 2 - CLR - WALL                 # -8.7
CRADLE_Z_TOP = 25.0                                 # top plate's top (the cover sits on it)
CRADLE_Y1 = SV.TAB_UNDER                            # the servos' tabs rest here
COVER_SCREWS_X = (CRADLE_X0 + 2.5, CRADLE_X1 - 2.5)
COVER_SCREW_Y = 8.0

# --- abduction --------------------------------------------------------------------------------------------------
ABD_Y = -20.0                       # abduction axis: along x, at (y, z) = (ABD_Y, 0), inside the body
ABD_PLATE_Y0 = -28.0                # the holder's front and rear plates reach in to here
ABD_LOW_Y1 = -12.0                  # below z = ABD_BLOCK_Z0 the plates stay inside this (clear of the inner thigh plate)
ABD_BLOCK_Z0 = 8.5
F_FACE = 11.7                       # front plate's inner face where the abduction horn sits (lower part)
F_OUT = 15.7
ABD_SERVO_X0 = F_FACE + SV.HORN_ARM_T + 0.2 - SV.HORN_TOP   # abduction servo's case bottom, -18.5
ABD_CAP_X0 = ABD_SERVO_X0 - CAP_T                            # -21.7 (body-side cap behind the abduction servo)
R_FACE = ABD_CAP_X0 - 0.5                                    # rear plate's inner face, -22.2
R_OUT = R_FACE - 0.5 - BRG_B - 1.0                           # -27.7

# --- thigh -----------------------------------------------------------------------------------------------------
IP_Y1 = -CAP_T - 0.4                # inner plate's face toward the servo, -3.6
IP_T = 5.0
IP_Y0 = IP_Y1 - IP_T                # -8.6
OP_Y0 = SV.HORN_TOP - SV.HORN_ARM_T - 0.2           # outer plate's inner face, 30.2
OP_T = 4.0
OP_Y1 = OP_Y0 + OP_T                # 34.2
THIGH_W, HUB_R = 10.0, 6.5
BRIDGE_Z = (-39.5, -35.0)           # thigh-local; inside r = 35, outside everything the holder sweeps (r <= 33.4)

# --- outboard planes (y) ---------------------------------------------------------------------------------------
GAP = 0.4
C0 = OP_Y1 + GAP; C1 = C0 + 3.0                          # long link              34.6 .. 37.6
D0 = C1 + GAP; D1 = D0 + 2 * BRG_B + 0.4                 # hubs with two 693s     38.0 .. 46.4
F0 = D1 + GAP; F1 = F0 + 3.0                             # short link             46.8 .. 49.8
G0 = F1 + GAP; G1 = G0 + BRG_B + 1.0                     # shoulder cover plate   50.2 .. 55.2

# --- linkage ---------------------------------------------------------------------------------------------------
RB = 10.0           # bellcrank long arm = shank lever (parallelogram with the thigh)
RS = 9.0            # bellcrank short arm = knee crank (parallelogram with the hip-to-knee-servo line)
ALPHA = 10.0        # shank lever's angle from the shank: the long parallelogram's dead points fall at q = -10 and 170
KNEE_MID = 180.0    # knee crank direction at the knee servo's centre position (pointing back)
KNEE_WINDOW = 55.0  # usable +- from there: past about 57 the short link meets the crank's and the axle's spacers (dead point at 90)
_phi_mid = T_HOME + sum(Q_RANGE) / 2 + ALPHA
BETA = (KNEE_MID - _phi_mid + 180.0) % 360.0 - 180.0     # short arm's angle from the long arm

# --- feet --------------------------------------------------------------------------------------------------------
FOOT_STUB_D, FOOT_STUB_L = 6.0, 6.0
Y_MID = (D0 + D1) / 2               # shank and foot centre plane

PLA, FILL = 1.24, 0.8               # g/cm^3; small parts print nearly solid (walls + 20 % infill)


# ===== helpers =====================================================================================================
def bx(x0, x1, y0, y1, z0, z1):
    return Pos((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2) * Box(abs(x1 - x0), abs(y1 - y0), abs(z1 - z0))


def cy(r, y0, y1, x=0.0, z=0.0):
    """Cylinder along y."""
    return Pos(x, (y0 + y1) / 2, z) * Rot(90, 0, 0) * Cylinder(r, abs(y1 - y0))


def cx(r, x0, x1, y=0.0, z=0.0):
    """Cylinder along x."""
    return Pos((x0 + x1) / 2, y, z) * Rot(0, 90, 0) * Cylinder(r, abs(x1 - x0))


def hex_y(af, y0, y1, x=0.0, z=0.0):
    return Pos(x, min(y0, y1), z) * Rot(-90, 0, 0) * extrude(RegularPolygon(af / math.sqrt(3), 6), abs(y1 - y0))


def hex_x(af, x0, x1, y=0.0, z=0.0):
    return Pos(min(x0, x1), y, z) * Rot(0, 90, 0) * extrude(RegularPolygon(af / math.sqrt(3), 6), abs(x1 - x0))


def d(a):
    return math.cos(math.radians(a)), math.sin(math.radians(a))


def pt(r, a, base=(0.0, 0.0)):
    return base[0] + r * math.cos(math.radians(a)), base[1] + r * math.sin(math.radians(a))


def bar(p, q, w, y0, y1):
    """A flat bar in the x-z plane from point p to q (x, z), width w, rounded ends, between y0 and y1."""
    (x0, z0), (x1, z1) = p, q
    length = math.hypot(x1 - x0, z1 - z0)
    ang = math.degrees(math.atan2(z1 - z0, x1 - x0))
    s = Pos((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2) * Rot(0, -ang, 0) * Box(length, abs(y1 - y0), w)
    return s + cy(w / 2, y0, y1, x0, z0) + cy(w / 2, y0, y1, x1, z1)


def ry(angle):
    """Turns something modelled pointing along +x to point along `angle` (x-z plane, from +x toward +z)."""
    return Rot(0, -angle, 0)


def bearing_pocket_y(y_open, toward, x=0.0, z=0.0, recess=0.5, lip_hole=6.2):
    """Pocket for a 693 opening at y_open and going `toward` (+1/-1) into the part, the bearing recessed by `recess`,
    and a hole through whatever lies behind it."""
    depth = recess + BRG_B
    p = cy((BRG_D + BRG_FIT) / 2, y_open, y_open + toward * depth, x, z)
    return p + cy(lip_hole / 2, y_open, y_open + toward * (depth + 20), x, z)


def bearing_pocket_x(x_open, toward, y, z, recess=0.5, lip_hole=6.2):
    depth = recess + BRG_B
    p = cx((BRG_D + BRG_FIT) / 2, x_open, x_open + toward * depth, y, z)
    return p + cx(lip_hole / 2, x_open, x_open + toward * (depth + 20), y, z)


def abd_frame():
    """Maps the servo frame onto the abduction servo: shaft along +x at (ABD_Y, 0), case bottom at ABD_SERVO_X0, and the
    body above the shaft (servo +x is world -z), so nothing of it or its mount sits where the leg swings inward."""
    return Location(Plane(origin=(ABD_SERVO_X0, ABD_Y, 0), x_dir=(0, 0, -1), z_dir=(0, -1, 0)))


# ===== parts, each in the leg frame at its own zero (thigh and shank pointing straight down, cranks along +x) =========
def holder():
    """Cradle for the hip and knee servos, with the abduction plates. Print with the front plate's outer face down."""
    h = bx(CRADLE_X0, CRADLE_X1, -CAP_T, CRADLE_Y1, CRADLE_Z0, CRADLE_Z_TOP)
    for zc in (0.0, KS_Z):   # the servos go in from +y, tabs resting on the end walls
        h -= bx(SV.X0 - CLR, SV.X1 + CLR, 0.0, CRADLE_Y1 + 1, zc - SV.W / 2 - CLR, zc + SV.W / 2 + CLR)
        for hx in SV.TAB_HOLES_X:
            h -= cy(SV_PILOT / 2, CRADLE_Y1 - 8, CRADLE_Y1 + 1, hx, zc)
    # hip pin: M3 nut dropped in from the servo side before the servo; standoff for the bearing's inner race outside
    h -= hex_y(M3_NUT_AF, 0.01, -M3_NUT_T - 0.1)
    h -= cy(M3_CLEAR / 2, -CAP_T - 2, 0.1)
    h += cy(BRG_RACE_D / 2, -CAP_T + 0.01, IP_Y1 - 0.5) - cy(M3_CLEAR / 2, -CAP_T + 1, IP_Y1 - 1)
    for x in COVER_SCREWS_X:
        h -= Pos(x, COVER_SCREW_Y, CRADLE_Z_TOP - 4) * Cylinder(M25_PILOT / 2, 8.02)
    # abduction plates: a block above the hip (clear of the inner thigh plate) and a lower part inside the body
    h += bx(CRADLE_X1 - 0.01, F_OUT, ABD_PLATE_Y0, 2.0, ABD_BLOCK_Z0, CRADLE_Z_TOP)
    h += bx(F_FACE, F_OUT, ABD_PLATE_Y0, ABD_LOW_Y1, -9.0, ABD_BLOCK_Z0 + 0.01)
    h += bx(R_OUT, CRADLE_X0 + 0.01, ABD_PLATE_Y0, 2.0, ABD_BLOCK_Z0, CRADLE_Z_TOP)
    for xa, xb in ((SV.X1 + CLR - 0.01, CRADLE_X1 - WALL), (CRADLE_X0 + WALL, SV.X0 - CLR + 0.01)):   # thin end walls
        h -= bx(xa, xb, 0.0, CRADLE_Y1 - 7.0, CRADLE_Z0 + WALL, CRADLE_Z_TOP - 3.7)
    # windows in the cap behind the knee servo and beside the hip pin (the frame around them keeps the box stiff)
    h -= bx(SV.X0 + 3.0, SV.X1 - 3.0, -CAP_T - 1, 0.01, KS_Z - 4.0, KS_Z + 4.0)
    h -= bx(SV.X0 + 3.0, -6.0, -CAP_T - 1, 0.01, -3.5, 3.5)
    h += bx(R_OUT, R_FACE, ABD_PLATE_Y0, ABD_LOW_Y1, -9.0, ABD_BLOCK_Z0 + 0.01)
    # abduction horn in the front plate (its arm pointing up), screwed from outside
    h -= abd_frame() * SV.horn_pocket(SV.HORN_ARM_L, 180.0, SV.HORN_TOP - SV.HORN_ARM_T - 0.2, SV.HORN_ARM_T + 0.21)
    h -= cx(M2_CLEAR / 2, F_FACE - 1, F_OUT + 1, ABD_Y, 0.0)
    h -= cx(M2_HEAD_D / 2 + 0.2, F_OUT - 1.6, F_OUT + 1, ABD_Y, 0.0)
    for r in SV.HORN_HOLES:
        h -= cx(M2_CLEAR / 2, F_FACE - 1, F_OUT + 1, ABD_Y, r)
    # abduction bearing in the rear plate, opening toward the cap
    h -= bearing_pocket_x(R_FACE, -1, ABD_Y, 0.0)
    return h


MOUNT_Y0, MOUNT_Y1 = ABD_Y - SV.W / 2 - CLR - 2.0, ABD_Y + SV.W / 2 + CLR   # inner rail; nothing past the servo outward
MOUNT_TOP = -(SV.XC - SV.TAB_SPAN / 2) + 0.5                                     # 22.2


def abd_mount():
    """Body side: holds the abduction servo by its tabs, and carries the cap with the rear pin's nut. Bolts to the body
    (or to a test stand) with two M2.5 screws from above, through the plate on top. Everything sits above the leg's
    inward swing and inboard of the servo. Print the top plate down."""
    tab_x = ABD_SERVO_X0 + SV.TAB_UNDER
    z_lo = -(SV.XC + SV.TAB_SPAN / 2) - 0.5                                      # servo x maps to world -z
    win = bx(tab_x - 4, tab_x + 1, ABD_Y - SV.W / 2 - CLR, ABD_Y + SV.W / 2 + CLR + 1, -SV.X1 - CLR, -SV.X0 + CLR)
    m = bx(tab_x - 3.0, tab_x, MOUNT_Y0, MOUNT_Y1, z_lo, MOUNT_TOP) - win        # tab plate: two pads and the inner rail
    for hx in SV.TAB_HOLES_X:
        m -= cx(SV_PILOT / 2, tab_x - 8, tab_x + 1, ABD_Y, -hx)
    m += bx(ABD_CAP_X0, ABD_SERVO_X0, MOUNT_Y0, MOUNT_Y1, -SV.X1 - CLR - 1.0, MOUNT_TOP)   # cap behind the servo
    m += bx(ABD_CAP_X0, tab_x, MOUNT_Y0, ABD_Y + 3.0, MOUNT_TOP - 0.01, MOUNT_TOP + 3.0)   # top plate
    m += bx(ABD_CAP_X0, tab_x, MOUNT_Y0, MOUNT_Y0 + 2.0, -SV.X1 - CLR - 1.0, MOUNT_TOP)    # inner rail along the servo
    m -= hex_x(M3_NUT_AF, ABD_SERVO_X0 + 0.01, ABD_SERVO_X0 - M3_NUT_T - 0.1, ABD_Y, 0.0)
    m -= cx(M3_CLEAR / 2, ABD_CAP_X0 - 2, ABD_SERVO_X0 + 0.1, ABD_Y, 0.0)
    m += cx(BRG_RACE_D / 2, R_FACE - 0.5, ABD_CAP_X0 + 0.01, ABD_Y, 0.0) - cx(M3_CLEAR / 2, R_FACE - 1, ABD_CAP_X0 + 1, ABD_Y, 0.0)
    for x in (ABD_CAP_X0 + 3.5, tab_x - 6.5):
        m -= Pos(x, ABD_Y - 2.0, MOUNT_TOP) * Cylinder(M25_CLEAR / 2, 8.0)
    return m


def outer_plate():
    """Thigh plate on the hip horn (its arm along the thigh). Print the inner face (horn pocket, nut trap) down."""
    p = bar((0, 0), (0, -L_THIGH), THIGH_W, OP_Y0, OP_Y1) + cy(HUB_R, OP_Y0, OP_Y1)
    p -= SV.horn_pocket(SV.HORN_ARM_L, -90.0, OP_Y0, SV.HORN_ARM_T + 0.2)
    p -= cy(M2_CLEAR / 2, OP_Y0 - 1, OP_Y1 + 1)
    p -= cy(M2_HEAD_D / 2 + 0.2, OP_Y1 - 1.6, OP_Y1 + 1)                     # horn's centre screw
    for r in SV.HORN_HOLES:
        p -= cy(M2_CLEAR / 2, OP_Y0 - 1, OP_Y1 + 1, 0, -r)
    p += cy(BRG_RACE_D / 2, OP_Y1 - 0.01, D0, 0, -L_THIGH)                   # knee spacer boss
    p -= cy(M3_CLEAR / 2, OP_Y0 - 1, D0 + 1, 0, -L_THIGH)
    p -= hex_y(M3_NUT_AF, OP_Y0 - 0.01, OP_Y0 + M3_NUT_T + 0.1, 0, -L_THIGH)
    return _bridge_seat(p, OP_Y0, +1)


def inner_plate():
    """Thigh plate on the bearing behind the hip servo. Print the bearing pocket's open face down."""
    p = bar((0, 0), (0, -40.0), THIGH_W, IP_Y0, IP_Y1) + cy(HUB_R, IP_Y0, IP_Y1)
    p -= bearing_pocket_y(IP_Y0, +1)
    return _bridge_seat(p, IP_Y1, -1)


def _bridge_seat(p, y_face, into):
    """A 0.8 mm seat that keys the bridge, and the screw hole through the plate. `into`: +1/-1, into the plate."""
    zc = sum(BRIDGE_Z) / 2
    p -= bx(-3.0, 3.0, y_face - into * 0.01, y_face + into * 0.8, BRIDGE_Z[0] + 0.6, BRIDGE_Z[1] - 0.6)
    p -= cy(M25_CLEAR / 2, y_face - into * 1, y_face + into * 20, 0, zc)
    return p


def bridge():
    """Joins the thigh plates near the knee: one M2.5 x 8 from each plate into 2.4 mm pilots, keyed into the seats."""
    zc = sum(BRIDGE_Z) / 2
    b = bx(-THIGH_W / 2, THIGH_W / 2, IP_Y1, OP_Y0, BRIDGE_Z[0], BRIDGE_Z[1])
    b += bx(-2.9, 2.9, IP_Y1 - 0.7, IP_Y1 + 0.01, BRIDGE_Z[0] + 0.7, BRIDGE_Z[1] - 0.7)
    b += bx(-2.9, 2.9, OP_Y0 - 0.01, OP_Y0 + 0.7, BRIDGE_Z[0] + 0.7, BRIDGE_Z[1] - 0.7)
    b -= cy(M25_PILOT / 2, IP_Y1 - 1, IP_Y1 + 8, 0, zc)
    b -= cy(M25_PILOT / 2, OP_Y0 - 8, OP_Y0 + 1, 0, zc)
    b -= bx(-THIGH_W / 2 - 1, THIGH_W / 2 + 1, IP_Y1 + 9, OP_Y0 - 9, BRIDGE_Z[0] + 1.6, BRIDGE_Z[1] - 1.6)  # lighten
    return b


def shank():
    """Knee hub with two 693s, the bar to the foot stub, and the lever pin for the long link. Shank-local: knee at the
    origin, foot straight down. Print the inner face down."""
    s = cy(BRG_D / 2 + 1.4, D0, D1)
    s += bar((0, 0), (0, -(L_SHANK - FOOT_STUB_L)), 6.0, D0 + 0.5, D1 - 0.5)
    s -= bearing_pocket_y(D0, +1, recess=0.0, lip_hole=5.6)
    s -= bearing_pocket_y(D1, -1, recess=0.0, lip_hole=5.6)
    lx, lz = pt(RB, -90 + ALPHA)
    s += cy(3.0, D0, D1 - 0.5, lx, lz)
    s -= cy(M2_CLEAR / 2, D0 - 1, D1 + 1, lx, lz)
    s -= cy(M2_HEAD_D / 2 + 0.2, C0 + 8.0, D1 + 1, lx, lz)        # M2 x 8 from outside, its tip at the link's far face
    s += Pos(0, Y_MID, -(L_SHANK - FOOT_STUB_L / 2) - 0.01) * Cylinder(FOOT_STUB_D / 2, FOOT_STUB_L)
    return s


def foot():
    """Foot ball pushed onto the shank's stub: TPU, or silicone cast in a mould. Shank-local."""
    f = Pos(0, Y_MID, -L_SHANK) * Sphere(FOOT_R)
    f -= Pos(0, Y_MID, -(L_SHANK - FOOT_STUB_L / 2) + 0.5) * Cylinder(FOOT_STUB_D / 2 - 0.1, FOOT_STUB_L + 1)
    return f


def bellcrank():
    """On the hip axis, on two 693s and an M3 x 20 axle held by the cover. Long arm along +x, short arm at BETA."""
    sx, sz = pt(RS, BETA)
    b = cy(BRG_D / 2 + 1.2, D0, D1) + bar((0, 0), (RB, 0), 6.0, D0, D1) + bar((0, 0), (sx, sz), 6.0, D0, D1)
    b -= bearing_pocket_y(D0, +1, recess=0.0, lip_hole=5.6)
    b -= bearing_pocket_y(D1, -1, recess=0.0, lip_hole=5.6)
    b -= cy(M2_CLEAR / 2, D0 - 1, D1 + 1, RB, 0)                  # long link pin: M2 x 8 from outside
    b -= cy(M2_HEAD_D / 2 + 0.2, C0 + 8.0, D1 + 1, RB, 0)
    b -= cy(M2_CLEAR / 2, D0 - 1, D1 + 1, sx, sz)                 # short link pin: M2 x 8 from inside
    b -= cy(M2_HEAD_D / 2 + 0.2, D0 - 1, F1 - 8.0, sx, sz)
    return b


def knee_crank():
    """On the knee servo's horn, crank arm along +x. The horn's arm is cut to 10 mm and sits at -90 to the crank, so it
    points up when the crank points back. The crank's outer end turns in a bearing in the cover (M3 x 12 from outside).
    Knee-servo-local: shaft at the origin. Print the base down."""
    base_tip = (0, -(SV.HORN_ARM_L_CUT + 1.0))
    k = cy(5.0, OP_Y0, OP_Y1) + bar((0, 0), base_tip, 7.0, OP_Y0, OP_Y1)
    k += cy(4.0, OP_Y1 - 0.01, D1) + bar((0, 0), (RS, 0), 6.0, OP_Y1 - 0.01, D1)
    k += cy(BRG_RACE_D / 2, D1 - 0.01, G0 + 0.5)
    k -= SV.horn_pocket(SV.HORN_ARM_L_CUT, -90.0, OP_Y0, SV.HORN_ARM_T + 0.2)
    k -= cy(2.3, OP_Y0 - 0.01, OP_Y0 + SV.HORN_ARM_T + 0.2 + 1.8)          # the horn's centre screw head
    k -= cy(M2_CLEAR / 2, OP_Y0 - 1, OP_Y1 + 1, 0, -SV.HORN_HOLES[0])
    k -= cy(M2_HEAD_D / 2 + 0.2, OP_Y1 - 1.5, OP_Y1 + 0.01, 0, -SV.HORN_HOLES[0])
    k -= cy(M3_PILOT / 2, G0 + 1, D1 - 6, 0, 0)
    k -= cy(M2_CLEAR / 2, OP_Y1 - 1, D1 + 1, RS, 0)                         # short link pin: M2 x 8 from inside
    k -= cy(M2_HEAD_D / 2 + 0.2, OP_Y1 - 1, F1 - 8.0, RS, 0)
    return k


def long_link():
    """Bellcrank long arm to the shank lever; the M2 pins thread into it (1.6 mm pilots). Along +x."""
    return bar((0, 0), (L_THIGH, 0), 5.0, C0, C1) - cy(M2_PILOT / 2, C0 - 1, C1 + 1) - cy(M2_PILOT / 2, C0 - 1, C1 + 1, L_THIGH)


def short_link():
    """Bellcrank short arm to the knee crank, along +z (parallel to the hip-to-knee-servo line)."""
    return bar((0, 0), (0, KS_Z), 5.0, F0, F1) - cy(M2_PILOT / 2, F0 - 1, F1 + 1) - cy(M2_PILOT / 2, F0 - 1, F1 + 1, 0, KS_Z)


HOOD_Z0 = KS_Z + SV.HORN_ARM_L_CUT + 1.0 + 3.5 + 0.5      # above the knee crank's base arm at its highest


def cover():
    """Shoulder cover: a hood over the holder and a plate outside the linkage that carries the bellcrank's axle (M3 x 20
    from inside, nut outside) and the knee crank's outer bearing. Two M2.5 x 10 into the holder's top. Print the outer
    face down."""
    x0, x1 = CRADLE_X0, CRADLE_X1
    z1 = HOOD_Z0 + WALL
    hy0 = COVER_SCREW_Y - 4.0
    c = bx(x0, x1, hy0, G1, HOOD_Z0, z1)
    for xa, xb in ((x0, x0 + 5.0), (x1 - 5.0, x1)):                           # feet on the holder's top
        c += bx(xa, xb, hy0, CRADLE_Y1, CRADLE_Z_TOP, HOOD_Z0 + 0.01)
    c += bx(-9.0, 9.0, G0, G1, 0.0, HOOD_Z0 + 0.01) + cy(9.0, G0, G1) + cy(8.0, G0, G1, 0, KS_Z)
    c += bx(-13.0, 13.0, G0, G1, KS_Z, HOOD_Z0 + 0.01)
    c -= bx(-6.0, 6.0, G0 - 1, G1 + 1, KS_Z + 7.5, HOOD_Z0 - 2.0)               # window (lighter; shows the crank)
    c += cy(BRG_RACE_D / 2, D1, G0 + 0.01)                                     # bellcrank axle spacer
    c -= cy(M3_CLEAR / 2, D1 - 1, G1 + 1)
    c -= hex_y(M3_NUT_AF, G1 + 0.01, G1 - M3_NUT_T - 0.1)
    c -= bearing_pocket_y(G1, -1, 0, KS_Z)
    for x in COVER_SCREWS_X:
        c -= Pos(x, COVER_SCREW_Y, (CRADLE_Z_TOP + z1) / 2) * Cylinder(M25_CLEAR / 2, z1 - CRADLE_Z_TOP + 1)
    return c


# ===== linkage and assembly =========================================================================================
def wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def linkage(t, q):
    """Joint directions for thigh direction t and knee bend q (degrees). Returns a dict with the bellcrank angle phi,
    the knee crank angle psi, the knee servo's angle from centre, and the pin points."""
    s = t + q
    phi = s + ALPHA
    psi = phi + BETA
    knee = (0.0, 0.0)
    kx, kz = pt(L_THIGH, t)
    return dict(s=s, phi=phi, psi=psi, servo=wrap(psi - KNEE_MID), K=(kx, kz),
                B=pt(RB, phi), L=pt(RB, s + ALPHA, (kx, kz)), P1=pt(RS, psi), P2=pt(RS, psi, (0.0, KS_Z)), H=knee)


def abd_rot(a):
    return Pos(0, ABD_Y, 0) * Rot(a, 0, 0) * Pos(0, -ABD_Y, 0)


_PARTS = {}


def part(name):
    """Each part built once."""
    if name not in _PARTS:
        _PARTS[name] = {"holder": holder, "abd_mount": abd_mount, "outer_plate": outer_plate, "inner_plate": inner_plate,
                        "bridge": bridge, "shank": shank, "foot": foot, "bellcrank": bellcrank,
                        "knee_crank": knee_crank, "long_link": long_link, "short_link": short_link,
                        "cover": cover}[name]()
    return _PARTS[name]


HOLDER_SIDE = ("holder", "cover", "hip_servo", "knee_servo", "abd_servo", "abd_horn", "abd_mount")


def posed(t=T_HOME, q=Q_HOME, a=0.0, with_servos=True, rear=False):
    """Every part placed for thigh direction t, knee bend q and abduction a. Returns {name: shape}. rear: a rear leg,
    whose holder, cover, abduction servo and mount are the front ones mirrored front to back (the servos turned 180
    degrees about their shafts); the thigh, shank and linkage are the same parts."""
    lk = linkage(t, q)
    thigh = Rot(0, -(t + 90), 0)             # parts modelled pointing down (-90)
    shank_loc = Pos(lk["K"][0], 0, lk["K"][1]) * Rot(0, -(lk["s"] + 90), 0)
    (bx_, bz_), (lx_, lz_) = lk["B"], lk["L"]
    link_ang = math.degrees(math.atan2(lz_ - bz_, lx_ - bx_))
    ks = Pos(0, 0, KS_Z)
    out = {
        "holder": part("holder"),
        "cover": part("cover"),
        "outer_plate": thigh * part("outer_plate"),
        "inner_plate": thigh * part("inner_plate"),
        "bridge": thigh * part("bridge"),
        "shank": shank_loc * part("shank"),
        "foot": shank_loc * part("foot"),
        "bellcrank": ry(lk["phi"]) * part("bellcrank"),
        "knee_crank": ks * ry(lk["psi"]) * part("knee_crank"),
        "long_link": Pos(bx_, 0, bz_) * ry(link_ang) * part("long_link"),
        "short_link": Pos(lk["P1"][0], 0, lk["P1"][1]) * part("short_link"),   # always parallel to the hip-knee-servo line
    }
    if with_servos:
        out["hip_servo"] = SV.body()
        out["hip_horn"] = thigh * SV.horn(SV.HORN_ARM_L, -90.0)
        out["knee_servo"] = ks * SV.body()
        out["knee_horn"] = ks * ry(lk["psi"]) * SV.horn(SV.HORN_ARM_L_CUT, -90.0)
        out["abd_servo"] = abd_frame() * SV.body()
        out["abd_horn"] = abd_frame() * SV.horn(SV.HORN_ARM_L, 180.0)
    out["abd_mount"] = part("abd_mount")
    if rear:
        for k in HOLDER_SIDE:
            if k in out:
                out[k] = out[k].mirror(Plane.YZ)
    moving_with_holder = [k for k in out if k not in ("abd_servo", "abd_mount")]
    for k in moving_with_holder:
        out[k] = abd_rot(a) * out[k]
    return out


# ===== export ======================================================================================================
PRINT = {   # how each part lies on the bed: a rotation, then dropped to z = 0
    "holder": Rot(0, 90, 0),       # front plate's outer face down
    "abd_mount": Rot(180, 0, 0),   # top plate down
    "outer_plate": Rot(90, 0, 0),  # inner face down
    "inner_plate": Rot(90, 0, 0),
    "bridge": Rot(90, 0, 0),
    "shank": Rot(90, 0, 0),
    "foot": Rot(0, 0, 0),
    "bellcrank": Rot(90, 0, 0),
    "knee_crank": Rot(90, 0, 0),
    "long_link": Rot(90, 0, 0),
    "short_link": Rot(90, 0, 0),
    "cover": Rot(-90, 0, 0),       # outer face down
}


def on_bed(name):
    s = PRINT[name] * part(name)
    bb = s.bounding_box()
    return Pos(-(bb.min.X + bb.max.X) / 2, -(bb.min.Y + bb.max.Y) / 2, -bb.min.Z) * s


def grams(name):
    return part(name).volume / 1000 * PLA * FILL


def main():
    out = HERE / "stl" / ("draft" if DRAFT else "leg")
    out.mkdir(parents=True, exist_ok=True)
    total = 0.0
    for name in PRINT:
        s = on_bed(name)
        export_stl(s, str(out / f"{name.replace('_', '-')}.stl"))
        bb = s.bounding_box()
        g = grams(name)
        total += g
        print(f"{name:12s} {bb.size.X:6.1f} x {bb.size.Y:5.1f} x {bb.size.Z:5.1f} mm  {g:5.1f} g")
    print(f"printed parts per leg: {total:.1f} g (PLA, {FILL:.0%} fill)")
    step = HERE / "step"
    step.mkdir(exist_ok=True)
    shapes = posed()
    export_step(Compound(children=list(shapes.values())), str(step / "leg-home.step"))
    print(f"wrote {out} and {step / 'leg-home.step'}; BETA = {BETA:.1f} deg")


if __name__ == "__main__":
    main()
