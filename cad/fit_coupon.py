"""Fit coupons for the leg: print these before any leg part, with a real MG92B, its horn and a 693ZZ in hand.

  fit-cradle.stl    a slice of the holder: one servo pocket with its end walls and tab screw pilots (checks the body
                    size, the tab height and the tab hole spacing, which are placeholders in servo.py)
  fit-bearings.stl  three 693 pockets, 7.9 / 8.0 / 8.1 mm (one, two and three dots): the bearing should press in by
                    thumb and not fall out
  fit-horn.stl      the outer thigh plate's hub: the horn pocket, its centre screw and arm screw holes
  fit-pilots.stl    M2 pilots 1.5 / 1.6 / 1.7 mm (link pins and servo tab screws), an M2.5 2.4 mm pilot, an M3 nut trap

Usage: cad/.venv/Scripts/python cad/fit_coupon.py   (writes to cad/stl/leg-fit/)
"""
import sys
from pathlib import Path

from build123d import Cylinder, Pos, Rot, export_stl

sys.path.insert(0, str(Path(__file__).parent))
import leg as LG  # noqa: E402
import servo as SV  # noqa: E402
from leg import bx, cy, hex_y  # noqa: E402

HERE = Path(__file__).parent


def dots(n, x, y, z):
    s = None
    for i in range(n):
        d = Pos(x + i * 2.2, y, z) * Cylinder(0.6, 0.8)
        s = d if s is None else s + d
    return s


def cradle():
    """One servo's pocket from the holder, cap down."""
    z0, z1 = -SV.W / 2 - LG.CLR - LG.WALL, SV.W / 2 + LG.CLR + LG.WALL
    c = bx(LG.CRADLE_X0, LG.CRADLE_X1, -LG.CAP_T, LG.CRADLE_Y1, z0, z1)
    c -= bx(SV.X0 - LG.CLR, SV.X1 + LG.CLR, 0.0, LG.CRADLE_Y1 + 1, -SV.W / 2 - LG.CLR, SV.W / 2 + LG.CLR)
    for hx in SV.TAB_HOLES_X:
        c -= cy(LG.SV_PILOT / 2, LG.CRADLE_Y1 - 8, LG.CRADLE_Y1 + 1, hx, 0)
    c -= hex_y(LG.M3_NUT_AF, 0.01, -LG.M3_NUT_T - 0.1)
    c -= cy(LG.M3_CLEAR / 2, -LG.CAP_T - 2, 0.1)
    return Rot(90, 0, 0) * c


def bearings():
    p = bx(-15, 15, 0, 5.0, -8, 8)
    for i, fit in enumerate((-0.1, 0.0, 0.1)):
        x = -9.5 + i * 9.5
        p -= cy((LG.BRG_D + fit) / 2, 5.01, 5.0 - LG.BRG_B - 0.5, x, 0)
        p -= cy(3.0, -1, 6, x, 0)
    p = Rot(90, 0, 0) * p
    for i in range(3):
        p += dots(i + 1, -9.5 + i * 9.5 - 1.1 * i, -6.5, 5.4)
    return p


def horn():
    """The outer plate's hub and the first 20 mm of the thigh, inner face down."""
    p = LG.bar((0, 0), (0, -20), LG.THIGH_W, LG.OP_Y0, LG.OP_Y1) + cy(LG.HUB_R, LG.OP_Y0, LG.OP_Y1)
    p += LG.bar((0, -15), (0, -20), 20, LG.OP_Y0, LG.OP_Y1)
    p -= SV.horn_pocket(SV.HORN_ARM_L, -90.0, LG.OP_Y0, SV.HORN_ARM_T + 0.2)
    p -= cy(LG.M2_CLEAR / 2, LG.OP_Y0 - 1, LG.OP_Y1 + 1)
    p -= cy(LG.M2_HEAD_D / 2 + 0.2, LG.OP_Y1 - 1.6, LG.OP_Y1 + 1)
    for r in SV.HORN_HOLES:
        p -= cy(LG.M2_CLEAR / 2, LG.OP_Y0 - 1, LG.OP_Y1 + 1, 0, -r)
    return Rot(90, 0, 0) * Pos(0, -LG.OP_Y0, 0) * p


def pilots():
    p = bx(-17, 17, 0, 8.0, -8, 8)
    for i, dia in enumerate((1.5, 1.6, 1.7)):
        p -= cy(dia / 2, 8.01, 0.5, -13 + i * 6.0, 0)
    p -= cy(LG.M25_PILOT / 2, 8.01, 0.5, 5.0, 0)
    p -= hex_y(LG.M3_NUT_AF, 8.01, 8.0 - LG.M3_NUT_T - 0.1, 12.0, 0) + cy(LG.M3_CLEAR / 2, 9, -1, 12.0, 0)
    p = Rot(90, 0, 0) * p
    for i in range(3):
        p += dots(i + 1, -13 + i * 6.0 - 1.1 * i, -6.0, 8.4)
    return p


def on_bed(s):
    bb = s.bounding_box()
    return Pos(-(bb.min.X + bb.max.X) / 2, -(bb.min.Y + bb.max.Y) / 2, -bb.min.Z) * s


def main():
    out = HERE / "stl" / "leg-fit"
    out.mkdir(parents=True, exist_ok=True)
    for name, f in (("fit-cradle", cradle), ("fit-bearings", bearings), ("fit-horn", horn), ("fit-pilots", pilots)):
        s = on_bed(f())
        export_stl(s, str(out / f"{name}.stl"))
        bb = s.bounding_box()
        print(f"{name:13s} {bb.size.X:5.1f} x {bb.size.Y:5.1f} x {bb.size.Z:5.1f} mm  {s.volume / 1000 * LG.PLA * LG.FILL:4.1f} g")


if __name__ == "__main__":
    main()
