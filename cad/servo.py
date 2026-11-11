"""TowerPro MG92B stand-in and its horn, for fitting the leg parts around.

PLACEHOLDER sizes from shop listings (10 Nov 2026), not measured: 22.8 x 12.0 mm body, 31 mm to the top of the spline,
13.8 g, 23-tooth spline. The listings disagree on the tab span and none gives the tab hole spacing, so the values below
are the common 9 g servo pattern (MG90S / SG90). The fit coupon (fit_coupon.py) checks them before anything else is printed:
measure a real servo and update the MEASURE values here.

Servo frame: the output shaft is the y axis, pointing up out of the servo (+y), with y = 0 on the case bottom. The long
side runs along x, the shaft nearer the +x end. The short side (12 mm) runs along z.
"""
from build123d import Align, Box, Cylinder, Pos, Rot

# MEASURE: body
L = 22.8               # long side (x)
W = 12.0               # short side (z)
SHAFT_FROM_END = 5.9   # shaft axis from the +x end of the body
CASE_TOP = 25.0        # case bottom to the flat top of the case
BOSS_R, BOSS_TOP = 5.9, 28.5    # raised gear housing around the shaft
SPLINE_R, SPLINE_TOP = 2.4, 31.0
# MEASURE: mounting tabs (plates perpendicular to the shaft, sticking out of both ends of the long side)
TAB_UNDER = 18.4       # case bottom to the underside of the tabs
TAB_T = 2.5
TAB_SPAN = 32.4        # tip to tip along x
TAB_HOLE_PITCH = 27.8  # between the two tab holes, centred on the body
TAB_HOLE_D = 2.0
X0, X1 = SHAFT_FROM_END - L, SHAFT_FROM_END      # body x range: -16.9 .. 5.9
XC = (X0 + X1) / 2                                # body centre, -5.5
TAB_HOLES_X = (XC - TAB_HOLE_PITCH / 2, XC + TAB_HOLE_PITCH / 2)

# MEASURE: single-arm horn from the servo's bag
HORN_HUB_D = 7.0
HORN_TOP = SPLINE_TOP + 1.0          # the horn's top face (arm and hub flush)
HORN_HUB_BOTTOM = SPLINE_TOP - 3.0
HORN_ARM_T = 1.6
HORN_ARM_L = 15.0                    # axis to tip; the knee servo's horn is cut to HORN_ARM_L_CUT
HORN_ARM_L_CUT = 10.0
HORN_ARM_W0, HORN_ARM_W1 = 6.0, 4.0  # width at the hub and at the tip
HORN_HOLES = (9.0, 13.0)             # the arm's screw holes, from the axis (only the inner one on a cut horn)
HORN_SCREW = "M2 x 6 self-tapping (in the servo's bag)"

MASS = 13.8


def _bx(x0, x1, y0, y1, z0, z1):
    return Pos((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2) * Box(x1 - x0, y1 - y0, z1 - z0)


def _cy(r, y0, y1, x=0.0, z=0.0):
    return Pos(x, (y0 + y1) / 2, z) * Rot(90, 0, 0) * Cylinder(r, y1 - y0)


def body():
    """The servo with its tabs, boss and spline (no horn)."""
    s = _bx(X0, X1, 0, CASE_TOP, -W / 2, W / 2)
    s += _bx(XC - TAB_SPAN / 2, XC + TAB_SPAN / 2, TAB_UNDER, TAB_UNDER + TAB_T, -W / 2, W / 2)
    s += _cy(BOSS_R, CASE_TOP, BOSS_TOP)
    s += _cy(SPLINE_R, BOSS_TOP, SPLINE_TOP)
    for hx in TAB_HOLES_X:
        s -= _cy(TAB_HOLE_D / 2, TAB_UNDER - 1, TAB_UNDER + TAB_T + 1, x=hx)
    return s


def horn(arm_len=HORN_ARM_L, arm_angle=0.0):
    """Single-arm horn on the spline; the arm points along direction arm_angle (degrees, in the x-z plane from +x toward
    +z)."""
    h = _cy(HORN_HUB_D / 2, HORN_HUB_BOTTOM, HORN_TOP)
    arm = Pos(arm_len / 2, (2 * HORN_TOP - HORN_ARM_T) / 2, 0) * Box(arm_len, HORN_ARM_T, HORN_ARM_W0)
    arm += _cy(HORN_ARM_W1 / 2, HORN_TOP - HORN_ARM_T, HORN_TOP, x=arm_len)
    h += Rot(0, -arm_angle, 0) * arm
    return h


def horn_pocket(arm_len, arm_angle, y_face, depth, clear=0.2):
    """The space a horn's arm and hub need inside a part whose face lies at y_face (the horn's top face side), as a solid
    to subtract: the arm sinks `depth` into the part, the hub too."""
    w0, w1 = HORN_ARM_W0 + 2 * clear, HORN_ARM_W1 + 2 * clear
    p = _cy(HORN_HUB_D / 2 + clear, y_face - 0.01, y_face + depth)
    arm = Pos(arm_len / 2, y_face + depth / 2, 0) * Box(arm_len, depth + 0.02, w0)
    arm += _cy(w1 / 2, y_face - 0.01, y_face + depth, x=arm_len + clear)
    return p + Rot(0, -arm_angle, 0) * arm
