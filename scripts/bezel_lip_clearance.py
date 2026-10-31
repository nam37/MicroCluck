"""Do the bezel's bumper lips clear the legs? Signed clearance between each lip and every leg part, on the hips-back layout.

    uv run python scripts/bezel_lip_clearance.py --poses ~/microcluck_gait.npz --proud 4 8 12 16

The lips (hardware/bezel.scad: a capsule across the bezel's width, flush with the frame's top or bottom edge, standing `proud` mm ahead of the
frame face) are convex, so each is a set of planes, and the clearance is computed as microcluck_shell_clearance.py does for the shell: positive
outside, negative inside (penetration). Poses are the recorded gait (a stand-in: it was recorded on the layout before the hips moved up and
back, with the same legs and joint ranges) and a random sweep over 70% of each joint's range.
"""

import argparse
import re
from pathlib import Path

import mujoco
import numpy as np

from microcluck_shell_clearance import ROOT, Sweep, _planes


def lip_shapes(torso_scad: Path, proud_mm: float, radius_mm: float):
    """Convex plane sets of the top and bottom lip in the trunk frame, metres: the part of each that stands ahead of the frame face."""
    text = torso_scad.read_text()
    vec = lambda name: np.array([float(v) for v in re.search(rf"\b{name}\s*=\s*\[([^\]]*)\]", text).group(1).split(",")])  # noqa: E731
    pos, size = vec("bezel_pos") / 1000.0, vec("bezel_size") / 1000.0
    r, proud = radius_mm / 1000.0, proud_mm / 1000.0
    hw, x1 = size[1] / 2, pos[0] + size[0] / 2
    k = np.arange(200) + 0.5
    phi, theta = np.arccos(1 - 2 * k / len(k)), np.pi * (1 + 5 ** 0.5) * k  # a sphere's worth of directions
    dirs = np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], axis=1)
    xc = x1 + proud - r
    shapes = []
    for zc in (pos[2] + size[2] / 2 - r, pos[2] - size[2] / 2 + r):
        pts = [np.array([xc, s * (hw - r), zc]) + r * dirs for s in (-1, 1)]
        pts.append(np.array([[x1, sy * hw, zc + sz * r] for sy in (-1, 1) for sz in (-1, 1)]))
        pts = np.concatenate(pts)
        shapes.append(_planes(pts[pts[:, 0] >= x1 - 1e-9]))  # only what stands ahead of the frame face: the frame itself is there without lips
    return shapes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--torso", default=str(ROOT / "assets/torso.scad"))
    ap.add_argument("--scene", default=str(ROOT / "scene_walk_back29_fix.xml"), help="a scene including the hips-back walking model")
    ap.add_argument("--envelope", default=str(ROOT / "assets/shell_envelope.stl"))
    ap.add_argument("--poses", help=".npz of recorded gait joint positions (torque_baseline.py --dump-joints)")
    ap.add_argument("--proud", type=float, nargs="+", default=[4, 8, 12, 16], help="mm ahead of the frame face")
    ap.add_argument("--radius", type=float, default=5.0, help="mm")
    ap.add_argument("--samples", type=int, default=6000)
    ap.add_argument("--stress-fraction", type=float, default=0.7)
    a = ap.parse_args()

    s = Sweep(Path(a.scene), Path(a.envelope))
    rng = np.random.default_rng(0)
    sets = {}
    if a.poses:
        z = np.load(a.poses, allow_pickle=True)
        cols = [[str(n) for n in z["joint_names"]].index(n) for n in s.joint_names]
        pos = z["joint_pos"]
        sets["real gait"] = pos[rng.choice(len(pos), size=min(a.samples, len(pos)), replace=False)][:, cols]
    lo = np.array([s.m.jnt_range[s.jid[n]][0] for n in s.joint_names])
    hi = np.array([s.m.jnt_range[s.jid[n]][1] for n in s.joint_names])
    sets["stress"] = (lo + hi) / 2 + rng.uniform(-1, 1, size=(a.samples, len(s.joint_names))) * (hi - lo) / 2 * a.stress_fraction
    print(f"{len(s.geoms)} leg meshes; lips radius {a.radius:g} mm; clearance in mm, negative = the leg is inside the lip")
    for proud in a.proud:
        shapes = lip_shapes(Path(a.torso), proud, a.radius)
        for label, poses in sets.items():
            top, bottom, who = [], [], []
            for q in poses:
                cache = _cache(s, q)
                per = [s._closest([shape], s.geoms, *cache) for shape in shapes]
                top.append(per[0][0])
                bottom.append(per[1][0])
                who.append(per[0][1] if per[0][0] < per[1][0] else per[1][1])
            top, bottom = np.array(top), np.array(bottom)
            m = np.minimum(top, bottom)
            w = int(np.argmin(m))
            print(f"lip {proud:4.0f} mm | {label:9s} {len(poses):5d} poses | closest {1000 * m.min():7.1f} mm ({who[w]}, "
                  f"{'top' if top[w] < bottom[w] else 'bottom'} lip) | 1st pct {1000 * np.percentile(m, 1):6.1f} mm | "
                  f"intersecting: top {int((top < 0).sum())}, bottom {int((bottom < 0).sum())} ({100 * (m < 0).mean():.1f}% of poses)", flush=True)


def _cache(s, q):
    """Put the legs in pose q; return the leg parts' hull centres and local-vertex function, as Sweep.clearance builds them."""
    m, d = s.m, s.data
    for adr, v in zip(s.adr, q):
        d.qpos[adr] = v
    mujoco.mj_kinematics(m, d)
    origin, rot = d.xpos[s.trunk], d.xmat[s.trunk].reshape(3, 3)
    rots = [d.geom_xmat[g].reshape(3, 3) for g, *_ in s.geoms]
    poss = [d.geom_xpos[g] for g, *_ in s.geoms]
    cents = [((r @ c + p) - origin) @ rot for r, p, (_, _, _, c, _) in zip(rots, poss, s.geoms)]
    return cents, lambda i: ((s.geoms[i][2] @ rots[i].T + poss[i]) - origin) @ rot


if __name__ == "__main__":
    main()
