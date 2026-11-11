"""MicroPup body shell in the simulation's style (the user's pick, 11 Nov 2026): rounded, the top sloping down to the rear,
the face leaning back 10 deg in a yellow bezel, the paddle tail. Built around the braced legs and the parts inside, and
checked: body_fit() below tests the parts inside the body against each other and the legs' body-side parts, every leg
over its range against the shell, and the left and right legs' abduction plates against each other.

Body frame: x forward, y left, z up, z = 0 on the hip axes (the simulation's trunk frame), mm.

Usage: cad/.venv/Scripts/python cad/body.py [--fit]   (writes stl/body/ and step/body.step; --fit runs the checks)
"""
import itertools
import math
import sys
from pathlib import Path

from build123d import (Box, Compound, Cylinder, Plane, Polygon, Pos, Rot, Sphere, export_step, export_stl, extrude,
                       fillet, scale)

sys.path.insert(0, str(Path(__file__).parent))
import leg_braced as LB  # noqa: E402
from leg import bx  # noqa: E402

HERE = Path(__file__).parent

# --- layout (the simulation's 12-servo micro dog: build_quad_model.py BODY12, HIP_DX, ABD_Y) --------------------------
HIP_X = 55.0                 # hip axes at x = +-55
LEG_Y = 36.0                 # hip servos' case bottoms from the middle: abduction axes at 36 - 24 = 12, as in the sim
FACE_TILT = 10.0             # the face leans back by this much; so does the body's front, and the bezel sits flush on it
FRONT_CLEAR = (83.0, 19.2)   # the slanted front passes outside this point: the front legs' abduction plates (to x 80.7,
                             # their deck to z 19.2) and the wall
X0 = -90.0                   # rear end
X1 = FRONT_CLEAR[0] + (FRONT_CLEAR[1] - -27.0) * math.tan(math.radians(FACE_TILT))   # front end at the bottom, 91.1
HW = 23.0                    # half width
Z0, Z1 = -27.0, 36.0         # bottom and top
WALL = 1.2                   # 3 lines of 0.4 mm (was 1.6: 71 g of shell)
EDGE_R = 5.0                 # rounding of every edge
REAR_DROP = 10.0             # the top slopes down from x = 0 to the rear by this much (sim: 16; the rear legs' abduction
                             # mounts stand 25 mm up, so 10 at most)
BELLY_HW, BELLY_Z = 16.0, -13.0   # the lower sides taper in below BELLY_Z to BELLY_HW at the bottom: thighs swinging inward
                                  # (abduction) pass under the shoulders

# --- face: Waveshare 1.69 in LCD module (33 x 40 x 5) in a bezel with a forehead band (camera) and a chin (ToF) ---------
BOARD = (5.0, 33.0, 40.0)
BEZEL_MARGIN, BEZEL_CHIN, BEZEL_TOP = 4.0, 14.0, 14.0
BEZEL_EXP = 0.35
SCREEN_HALF = (14.0, 16.3)

# --- tail: the sim's paddle (QUAD_TAIL=paddle, resting pose) -----------------------------------------------------------
TAIL_X = X0 + 16.0
TAIL_ANGLE = -45.0
TAIL_LEN, TAIL_BEND, TAIL_R = 66.0, 50.0, (4.4, 6.0)
TAIL_BULB = (9.2, 7.8, 17.0)
TAIL_BOOT = (5, 6.6, 2.8)

# --- parts inside (PLACEHOLDER sizes from the simulation's micro budget) -------------------------------------------------
INSIDE = {
    "battery": (Pos(0, 0, Z0 + WALL + 2.8 + 6.5) * Box(50, 30, 13)),                   # 2S ~450 mAh, in the belly
    "electronics": (Pos(0, 0, -2.0) * Box(60, 40, 12)),                                # ESP32-P4 board, PCA9685, IMU, 6 V regulator
    "tail_servo": (Pos(-21.0, 0, 16.0) * (Box(23, 12.2, 22.5) + Pos(0, 0, 4.0) * Box(32, 12.2, 2.5))),   # 9 g micro servo,
                                                                                       # pushrod to the tail's hinge
}


def top_z(x):
    return Z1 if x >= 0 else Z1 - REAR_DROP * (-x) / (-X0)


def _yz_prism(pts, x0, x1):
    s = extrude(Plane.YZ * Polygon(*pts, align=None), amount=x1 - x0)
    return Pos(x0 - s.bounding_box().min.X, 0, 0) * s


def _xz_prism(pts, y0, y1):
    s = extrude(Plane.XZ * Polygon(*pts, align=None), amount=y1 - y0)
    return Pos(0, y0 - s.bounding_box().min.Y, 0) * s


def outline(d=0.0):
    """The body's solid, shrunk by d all round (d = WALL gives the inside)."""
    b = bx(X0 + d, X1 - d, -HW + d, HW - d, Z0 + d, Z1 - d)
    for s in (1, -1):   # belly taper
        y0, z0 = HW - d, BELLY_Z
        y1, z1 = BELLY_HW - d, Z0 + d
        k = (y0 - y1) / (z0 - z1)
        pts = [(s * y0, z0 + 0.0), (s * (HW + 10), z0), (s * (HW + 10), Z0 - 10), (s * (y1 - k * 10), Z0 - 10 + d)]
        b -= _yz_prism(pts, X0 - 5, X1 + 5)
    tn = math.tan(math.radians(FACE_TILT))   # the front leans back: x = X1 - d / cos - (z - Z0) tan
    xb = X1 - d / math.cos(math.radians(FACE_TILT))
    b -= _xz_prism([(xb + 10 * tn, Z0 - 10), (xb + 40, Z0 - 10), (xb + 40, Z1 + 10),
                    (xb - (Z1 + 10 - Z0) * tn, Z1 + 10)], -HW - 5, HW + 5)
    slope = REAR_DROP / -X0
    b -= _xz_prism([(0.0, Z1 - d), (X0 - 10, Z1 - d - slope * (-(X0 - 10))), (X0 - 10, Z1 + 20), (0.0, Z1 + 20)], -HW - 5, HW + 5)
    try:
        b = fillet(b.edges(), max(EDGE_R - d, 0.8))
    except Exception:  # noqa: BLE001  (OpenCascade refuses some fillets; keep the sharp shape)
        pass
    return b


def leg_parts(front, left, t=LB.T_HOME, q=LB.Q_HOME, a=0.0, names=None):
    """One leg's parts posed in the body frame. front, left: +1 or -1."""
    p = LB.posed(t, q, a, rear=front < 0)
    out = {}
    for n, s in p.items():
        if names and n not in names:
            continue
        if left < 0:
            s = s.mirror(Plane.XZ)
        out[n] = Pos(front * HIP_X, left * LEG_Y, 0) * s
    return out


# parts that stay with the body or move only with abduction (they pass through the side wall)
HOLDER = ("bracket", "outer_arm", "hip_horn", "shoulder_cover", "abd_servo", "abd_horn", "abd_mount")
SWING = ("bracket", "outer_arm", "hip_horn", "shoulder_cover")
SWING_ANGLES = (-30.5, -29.0, -24.0, -18.0, -12.0, -6.0, 0.0, 6.0, 12.0, 18.0, 24.0, 29.0, 30.5)


_COVER_ENV = []


def swept_parts():
    """The shoulders' swinging parts at abduction angles over +-31 deg (2 deg past the range), every leg; the cover as its
    filled outline."""
    if not _COVER_ENV:
        _COVER_ENV.append(LB.shoulder_cover(envelope=True))
    for f, l in itertools.product((1, -1), (1, -1)):
        for a in SWING_ANGLES:
            yield from leg_parts(f, l, a=a, names=[n for n in SWING if n != "shoulder_cover"]).values()
            env = Pos(0, LB.ABD_Y, 0) * Rot(a, 0, 0) * Pos(0, -LB.ABD_Y, 0) * _COVER_ENV[0]
            if l < 0:
                env = env.mirror(Plane.XZ)
            yield Pos(f * HIP_X, l * LEG_Y, 0) * env


def shell():
    """The body shell with openings where the shoulders' parts pass through and swing (cut one at a time: OpenCascade
    fails to fuse them all first)."""
    s = outline() - outline(WALL)
    failed = 0
    for part in swept_parts():
        try:
            r = s - part
            if 0 < s.volume - r.volume <= part.volume + 1.0 or abs(s.volume - r.volume) < 1e-6:
                s = r
            else:
                failed += 1     # OpenCascade returned a fragment; keep the shell as it was
        except Exception:  # noqa: BLE001
            failed += 1
    if failed:
        print(f"  ({failed} opening cuts failed)")
    return s


def face():
    bt = BOARD[0]
    bh = BOARD[2]
    zc = Z0 + 4.0 + BEZEL_CHIN + bh / 2          # the bezel's centre height, as in the sim
    tr = math.radians(FACE_TILT)
    xf = X1 - (zc - Z0) * math.tan(tr)            # the front face at that height
    c = (xf + (bt + 4.0) / 2 * math.cos(tr), 0.0, zc + (bt + 4.0) / 2 * math.sin(tr))   # back face flush on the front
    hw, hh = BOARD[1] / 2 + BEZEL_MARGIN, (BOARD[2] + BEZEL_TOP + BEZEL_CHIN) / 2

    def slab(w, h, t, n=64):
        pts = []
        for k in range(n):
            th = 2 * math.pi * k / n
            cs, sn = math.cos(th), math.sin(th)
            pts.append((w * math.copysign(abs(cs) ** BEZEL_EXP, cs), h * math.copysign(abs(sn) ** BEZEL_EXP, sn)))
        return _yz_prism(pts, -t / 2, t / 2)

    tilt = Pos(*c) * Rot(0, -FACE_TILT, 0)
    bezel = tilt * slab(hw, hh, bt + 4.0)
    screen = tilt * Pos(bt / 2 + 2.0, 0, 0) * slab(*SCREEN_HALF, 0.6)
    eyes = None
    for y in (6.5, -6.5):
        e = tilt * Pos(bt / 2 + 2.4, y, 4.0) * Rot(0, 90, 0) * Cylinder(3.2, 0.4)
        eyes = e if eyes is None else eyes + e
    return dict(bezel=bezel, screen=screen, eyes=eyes)


def tail():
    piv = (TAIL_X, 0.0, top_z(TAIL_X) - 2.0)
    bend = math.radians(TAIL_BEND)
    rad = TAIL_LEN / bend

    def point(s):
        phi = s / rad
        return rad * (1 - math.cos(phi)), rad * math.sin(phi), phi

    place = Pos(*piv) * Rot(0, TAIL_ANGLE, 0)
    s1 = TAIL_LEN - 0.6 * TAIL_BULB[2]
    n = 8
    t = None
    for i in range(n):
        x0, z0, _ = point(s1 * i / n)
        x1, z1, _ = point(s1 * (i + 1) / n)
        r = TAIL_R[0] + (TAIL_R[1] - TAIL_R[0]) * i / (n - 1)
        length = math.hypot(x1 - x0, z1 - z0)
        ang = math.degrees(math.atan2(z1 - z0, x1 - x0))
        seg = Pos((x0 + x1) / 2, 0, (z0 + z1) / 2) * Rot(0, 90 - ang, 0) * Cylinder(r, length) + Pos(x1, 0, z1) * Sphere(r)
        t = seg if t is None else t + seg
    xb, zb, phi = point(TAIL_LEN)
    t += Pos(xb, 0, zb) * Rot(0, math.degrees(phi), 0) * scale(Sphere(1.0), TAIL_BULB)
    boot = None
    for k in range(TAIL_BOOT[0]):
        x, z, phi = point(3.0 + k * TAIL_BOOT[2])
        ring = Pos(x, 0, z) * Rot(0, math.degrees(phi), 0) * Cylinder(TAIL_BOOT[1], 1.8)
        boot = ring if boot is None else boot + ring
    return dict(tail=place * t, tail_boot=place * boot)


def parts():
    p = dict(body=shell())
    p.update(face())
    p.update(tail())
    return p


# ===== fit checks ===================================================================================================
def _hit(a, b):
    ba, bb = a.bounding_box(), b.bounding_box()
    if (ba.min.X > bb.max.X or bb.min.X > ba.max.X or ba.min.Y > bb.max.Y or bb.min.Y > ba.max.Y
            or ba.min.Z > bb.max.Z or bb.min.Z > ba.max.Z):
        return 0.0
    i = a & b
    return i.volume if i else 0.0


def body_fit():
    body = shell()
    inner = outline(WALL)
    problems = 0

    def report(what, v):
        nonlocal problems
        if v > 0.5:
            problems += 1
            print(f"  PROBLEM {what}: {v:.1f} mm^3")

    print("parts inside: within the shell, apart from each other and from the legs' body-side parts (abduction 0, +-29)")
    for n, s in INSIDE.items():
        report(f"{n} outside the body", s.volume - (s & inner).volume)
    for (n1, s1), (n2, s2) in itertools.combinations(INSIDE.items(), 2):
        report(f"{n1} x {n2}", _hit(s1, s2))
    for f, l in itertools.product((1, -1), (1, -1)):
        for a in (-29.0, 0.0, 29.0):
            for pn, ps in leg_parts(f, l, a=a, names=HOLDER).items():
                for n, s in INSIDE.items():
                    report(f"{n} x leg({f:+d},{l:+d}) {pn} at abduction {a:+.0f}", _hit(s, ps))
    print("left and right legs' abduction parts, both swung in or out by 29 deg")
    for f in (1, -1):
        for al, ar in itertools.product((-29.0, 0.0, 29.0), repeat=2):
            L = leg_parts(f, 1, a=al, names=HOLDER)
            R = leg_parts(f, -1, a=ar, names=HOLDER)
            for (n1, s1), (n2, s2) in itertools.product(L.items(), R.items()):
                report(f"{'front' if f > 0 else 'rear'} left {n1} ({al:+.0f}) x right {n2} ({ar:+.0f})", _hit(s1, s2))
    print("legs against the shell (front-left and rear-left; the right side is the mirror image)")
    poses = [(LB.T_HOME + dt, q, a) for dt, q, a in itertools.product((-70.0, 0.0, 80.0), (0.0, LB.Q_HOME, 150.0),
                                                                        (-29.0, 0.0, 29.0))]
    for f in (1, -1):
        for t, q, a in poses:
            for pn, ps in leg_parts(f, 1, t, q, a).items():
                if pn in ("abd_servo", "abd_mount", "abd_horn"):
                    continue
                report(f"{'front' if f > 0 else 'rear'} leg {pn} at hip {t - LB.T_HOME:+.0f} knee {q:.0f} abd {a:+.0f} x body",
                       _hit(ps, body))
    print("parts inside x shell")
    for n, s in INSIDE.items():
        report(f"{n} x shell", _hit(s, body))
    print("OK" if problems == 0 else f"{problems} problems")


def main():
    out = HERE / "stl" / "body"
    out.mkdir(parents=True, exist_ok=True)
    p = parts()
    for n, s in p.items():
        export_stl(s, str(out / f"{n}.stl"), tolerance=0.05, angular_tolerance=0.2)
    export_step(Compound(list(p.values()) + list(INSIDE.values())), str(HERE / "step" / "body.step"))
    b = p["body"]
    print(f"body shell {b.volume * 1.24 / 1000:.1f} g solid PLA; wrote {out} and step/body.step")
    if "--fit" in sys.argv:
        body_fit()


if __name__ == "__main__":
    main()
