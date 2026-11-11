"""The leg servo and its horn, for fitting the leg parts around.

Two servos, picked with the environment variable MICROPUP_SERVO:

- scs0009 (default since 11 Nov 2026): MangDang's TTL serial bus servo for Mini Pupper 2, in the case of Feetech's
  SCS0009. Sizes taken from the SCS0009 solid and its horn ("Rudder_Arm") in MangDang's Mini Pupper STEP (2021,
  github.com/gotree94/Mini-Pupper mirror), checked against MangDang's servo drawing (MiniPupper-Servo-Spec-20211026.pdf,
  the PWM version in the same case): 23.2 x 12.0 mm body, a gear housing at the shaft end, 28.7 mm to the top of the
  spline, 13.5 g. The bus version is sold as a drop-in upgrade for Mini Pupper, so the case should be the same, but
  nothing here has been measured on a real one yet: MEASURE values are checked with calipers on arrival (H0).
- mg92b: the TowerPro MG92B stand-in used until then (shop-listing sizes, the common 9 g tab pattern), the fallback.

Servo frame: the output shaft is the y axis, pointing up out of the servo (+y), with y = 0 on the case bottom. The long
side runs along x, the shaft nearer the +x end. The short side (12 mm) runs along z.
"""
import os

from build123d import Box, Cylinder, Pos, Rot

SERVO = os.environ.get("MICROPUP_SERVO", "scs0009")

if SERVO == "scs0009":
    # MEASURE: body
    L = 23.2               # long side (x)
    W = 12.0               # short side (z)
    SHAFT_FROM_END = 6.2   # shaft axis from the +x end of the body
    MAIN_TOP = 21.5        # case bottom to the top of the full-length case
    CASE_TOP = 25.5        # case bottom to the top of the gear housing (the highest flat face; the rigidity plate sits here)
    HOUSING_X = (-8.5, 6.0)   # the gear housing's x range about the shaft (full width in z)
    BOSS_R, BOSS_TOP = 2.4, 25.7    # spline root
    SPLINE_R, SPLINE_TOP = 2.4, 28.7
    # MEASURE: mounting tabs (plates perpendicular to the shaft, out of both ends of the long side, a slot to each hole)
    TAB_UNDER = 16.9       # case bottom to the underside of the tabs
    TAB_T = 1.6
    TAB_SPAN = 32.4        # tip to tip along x
    TAB_HOLE_PITCH = 28.0  # between the two tab holes, centred on the body
    TAB_HOLE_D = 2.0       # the model's holes are about 3 mm with a slot to the tip; the shell's M2 pilots line up with them
    # MEASURE: single-arm horn ("Rudder_Arm"): 6 mm hub 4 mm long, the arm 1.9 mm thick and 0.35 mm in from the hub's outer
    # end, tapering from 6 to 3.2 mm to a tip 17.75 mm from the axis, screw holes at 7 and 15 mm
    HORN_HUB_D = 6.0
    HORN_HUB_L = 4.0
    HORN_ARM_T = 2.25                    # the arm (1.9) plus the hub standing 0.35 proud of it: what a plate on the horn sees
    HORN_ARM_L = 16.15                   # axis to the tip's centre (tip radius 1.6)
    HORN_ARM_W0, HORN_ARM_W1 = 6.0, 3.2  # width at the hub and at the tip
    HORN_HOLES = (7.0, 15.0)
    HORN_SCREW = "M2 x 4 (the horn screw), M2 self-tapping into the arm (to be checked)"
    MASS = 13.5
elif SERVO == "mg92b":
    # PLACEHOLDER sizes from shop listings (10 Nov 2026), not measured
    L = 22.8
    W = 12.0
    SHAFT_FROM_END = 5.9
    MAIN_TOP = 25.0
    CASE_TOP = 25.0
    HOUSING_X = None
    BOSS_R, BOSS_TOP = 5.9, 28.5    # raised gear housing around the shaft
    SPLINE_R, SPLINE_TOP = 2.4, 31.0
    TAB_UNDER = 18.4
    TAB_T = 2.5
    TAB_SPAN = 32.4
    TAB_HOLE_PITCH = 27.8
    TAB_HOLE_D = 2.0
    HORN_HUB_D = 7.0
    HORN_HUB_L = 4.0
    HORN_ARM_T = 1.6
    HORN_ARM_L = 15.0
    HORN_ARM_W0, HORN_ARM_W1 = 6.0, 4.0
    HORN_HOLES = (9.0, 13.0)
    HORN_SCREW = "M2 x 6 self-tapping (in the servo's bag)"
    MASS = 13.8
else:
    raise ValueError(f"MICROPUP_SERVO={SERVO!r}: use 'scs0009' or 'mg92b'")

X0, X1 = SHAFT_FROM_END - L, SHAFT_FROM_END      # body x range
XC = (X0 + X1) / 2                                # body centre
TAB_HOLES_X = (XC - TAB_HOLE_PITCH / 2, XC + TAB_HOLE_PITCH / 2)
HORN_TOP = SPLINE_TOP + 1.0                       # the horn's outer face
HORN_HUB_BOTTOM = HORN_TOP - HORN_HUB_L
HORN_ARM_L_CUT = 10.0                             # the knee servo's horn is cut to this (only the inner hole kept)
HUB_CLEAR_R = max(BOSS_R, HORN_HUB_D / 2) + 0.6   # a hole that clears the boss and the horn's hub


def _bx(x0, x1, y0, y1, z0, z1):
    return Pos((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2) * Box(x1 - x0, y1 - y0, z1 - z0)


def _cy(r, y0, y1, x=0.0, z=0.0):
    return Pos(x, (y0 + y1) / 2, z) * Rot(90, 0, 0) * Cylinder(r, y1 - y0)


def body():
    """The servo with its tabs, housing or boss, and spline (no horn)."""
    s = _bx(X0, X1, 0, MAIN_TOP, -W / 2, W / 2)
    if HOUSING_X is not None:
        s += _bx(HOUSING_X[0], HOUSING_X[1], MAIN_TOP - 0.01, CASE_TOP, -W / 2, W / 2)
    s += _bx(XC - TAB_SPAN / 2, XC + TAB_SPAN / 2, TAB_UNDER, TAB_UNDER + TAB_T, -W / 2, W / 2)
    s += _cy(BOSS_R, CASE_TOP - 0.01, BOSS_TOP)
    s += _cy(SPLINE_R, BOSS_TOP - 0.01, SPLINE_TOP)
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
