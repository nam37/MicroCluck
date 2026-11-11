"""Checks the leg for collisions over its range of motion, and the knee linkage against the knee servo's travel.

Every pair of parts (servos and horns included) is intersected at each pose; any overlap above 0.5 mm^3 is reported.
The pose sets: home; the hip swept with the knee at home; the knee swept with the hip at home; abduction swept; and a
coarse hip x knee grid that maps where the linkage runs out of servo travel.

Usage: cad/.venv/Scripts/python cad/check_fit.py [home|hip|knee|abd|combos|grid|all] [--rear]
"""
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import leg as LG  # noqa: E402

MIN_OVERLAP = 0.5   # mm^3
# The parts at a pin joint touch along the pin's washer faces only; these pairs never share volume when the model is right.
SKIP = {("hip_servo", "hip_horn"), ("knee_servo", "knee_horn"), ("abd_servo", "abd_horn"), ("shank", "foot")}  # horn on spline; foot pressed on its stub


def overlaps(shapes):
    names = list(shapes)
    boxes = {n: shapes[n].bounding_box() for n in names}
    hits = []
    for a, b in itertools.combinations(names, 2):
        if (a, b) in SKIP or (b, a) in SKIP:
            continue
        ba, bb = boxes[a], boxes[b]
        if (ba.min.X > bb.max.X or bb.min.X > ba.max.X or ba.min.Y > bb.max.Y or bb.min.Y > ba.max.Y
                or ba.min.Z > bb.max.Z or bb.min.Z > ba.max.Z):
            continue
        v = (shapes[a] & shapes[b]).volume
        if v > MIN_OVERLAP:
            hits.append((a, b, v))
    return hits


REAR = False


def check(label, t, q, a):
    lk = LG.linkage(t, q)
    servo_ok = abs(lk["servo"]) <= LG.KNEE_WINDOW
    hits = overlaps(LG.posed(t, q, a, rear=REAR))
    flag = "ok" if not hits and servo_ok else "PROBLEM"
    print(f"{label:28s} t {t:7.1f} q {q:6.1f} a {a:5.1f} | knee servo {lk['servo']:+6.1f}{'' if servo_ok else ' (past travel)'}"
          f" | {flag}")
    for x, y, v in hits:
        print(f"      {x} x {y}: {v:.1f} mm^3")
    return not hits and servo_ok


def main(which):
    t0, q0 = LG.T_HOME, LG.Q_HOME
    if which in ("home", "all"):
        check("home", t0, q0, 0.0)
    if which in ("hip", "all"):
        for dt in (-35, -20, 20, 40, 60, 70):
            check(f"hip {dt:+d}", t0 + dt, q0, 0.0)
    if which in ("knee", "all"):
        for q in (30, 45, 100, 125, 140):
            check(f"knee {q}", t0, q, 0.0)
    if which in ("abd", "all"):
        for a in (-29, -15, 15, 29):
            check(f"abduction {a:+d}", t0, q0, a)
    if which in ("combos", "all"):
        for dt, q in ((-35, 140), (-35, 100), (60, 30), (70, 45), (0, 140), (0, 30)):
            check(f"hip {dt:+d}, knee {q}", t0 + dt, q, 0.0)
        for dt, q, a in ((-35, 140, -29), (60, 30, 29), (0, 140, -29), (0, 30, 29)):
            check(f"hip {dt:+d}, knee {q}, abd {a:+d}", t0 + dt, q, a)
    if which in ("grid", "all"):
        print("knee servo angle from centre over hip x knee (x: past its +-%.0f deg travel)" % LG.KNEE_WINDOW)
        qs = list(range(20, 151, 10))
        print("   t \\ q " + "".join(f"{q:6d}" for q in qs))
        for dt in range(-60, 61, 15):
            row = []
            for q in qs:
                s = LG.linkage(t0 + dt, q)["servo"]
                row.append(f"{s:+6.0f}" if abs(s) <= LG.KNEE_WINDOW else "     x")
            print(f"{t0 + dt:8.1f} " + "".join(row))


if __name__ == "__main__":
    REAR = "--rear" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--rear"]
    print("rear leg (holder mirrored front to back)" if REAR else "front leg")
    main(args[0] if args else "all")
