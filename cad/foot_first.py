"""Foot-first angles: how far a braced leg can lean sideways (body roll plus abduction) before some other part of it
touches a flat floor ahead of the foot. For each part, the smallest angle atan(height above the foot's bottom / sideways
distance from the foot's centre) over its vertices, on the outer and inner side, at several leg poses.

Usage: cad/.venv/Scripts/python cad/foot_first.py
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import leg_braced as LB  # noqa: E402

POSES = {"home": (LB.T_HOME, LB.Q_HOME), "crouch (h 48)": (-90 - math.degrees(math.acos(48 / 90)), 2 * math.degrees(math.acos(48 / 90))),
         "straight": (-90.0, 0.0), "folded 150": (LB.T_HOME - 30, 150.0)}


def angles(t, q):
    parts = LB.posed(t, q, 0.0)
    foot = parts["foot"].bounding_box()
    fy, fz = (foot.min.Y + foot.max.Y) / 2, foot.min.Z
    out = {}
    for name, s in parts.items():
        if name in ("foot", "abd_servo", "abd_mount", "abd_horn"):
            continue
        v = np.array([[p.X, p.Y, p.Z] for p in s.vertices()] +
                     [list(p) for p in s.tessellate(0.3)[0]])
        dy, h = v[:, 1] - fy, v[:, 2] - fz
        res = []
        for side in (1, -1):
            m = side * dy > 1.0
            res.append(min(math.degrees(math.atan2(hh, abs(d))) for hh, d in zip(h[m], dy[m])) if m.any() else 90.0)
        out[name] = res
    return out


def main():
    for pose, (t, q) in POSES.items():
        a = angles(t, q)
        worst = sorted(a.items(), key=lambda kv: min(kv[1]))[:4]
        print(f"{pose:14s} " + "  ".join(f"{n} out {o:.0f} in {i:.0f}" for n, (o, i) in worst))


if __name__ == "__main__":
    main()
