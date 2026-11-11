"""Shaded pictures of the leg assembly (matplotlib, no extra software).

Usage: cad/.venv/Scripts/python cad/render.py OUT.png [t q a]   (thigh direction, knee bend, abduction in degrees;
       default the home pose). Draws four views: front-left, side (from outside), front, and from above-behind.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
if "--braced" in sys.argv:          # the MicroDuck-braced leg (leg_braced.py)
    sys.argv.remove("--braced")
    import leg_braced as LG  # noqa: E402
else:
    import leg as LG  # noqa: E402

COLORS = {
    "shell": "#e9e6df", "rigidity_plate": "#f4f2ec", "bracket": "#c9ccd2", "outer_arm": "#aeb2ba",
    "shank_outer": "#c9ccd2", "shank_inner": "#c9ccd2", "shank_body": "#aeb2ba", "shoulder_cover": "#f4f2ec",
    "holder": "#e9e6df", "cover": "#f4f2ec", "abd_mount": "#b9b4a8",
    "outer_plate": "#c9ccd2", "inner_plate": "#c9ccd2", "bridge": "#aeb2ba",
    "shank": "#c9ccd2", "foot": "#e8b21c",
    "bellcrank": "#5b8fd6", "knee_crank": "#5b8fd6", "long_link": "#d9822b", "short_link": "#d9822b",
    "hip_servo": "#2b2b2b", "knee_servo": "#2b2b2b", "abd_servo": "#2b2b2b",
    "hip_horn": "#f0f0f0", "knee_horn": "#f0f0f0", "abd_horn": "#f0f0f0",
}
VIEWS = [(20, -55, "front-left"), (0, 90, "side, from outside"), (0, 0, "front"), (35, -150, "above, behind")]


def mesh(shape, tol=0.15):
    verts, tris = shape.tessellate(tol, 0.3)
    v = np.array([(p.X, p.Y, p.Z) for p in verts])
    return v[np.array(tris)]


def shade(tris, rgb):
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    light = np.array([0.4, 0.8, 0.6]); light /= np.linalg.norm(light)
    k = 0.45 + 0.55 * np.abs(n @ light)
    return np.clip(np.array(rgb)[None, :] * k[:, None], 0, 1)


def render(out, t=LG.T_HOME, q=LG.Q_HOME, a=0.0, hide=()):
    shapes = {k: v for k, v in LG.posed(t, q, a).items() if k not in hide}
    meshes = {k: mesh(v) for k, v in shapes.items()}
    allv = np.concatenate([m.reshape(-1, 3) for m in meshes.values()])
    lo, hi = allv.min(0), allv.max(0)
    c, r = (lo + hi) / 2, (hi - lo).max() / 2
    fig = plt.figure(figsize=(12, 12), dpi=110)
    for i, (elev, azim, title) in enumerate(VIEWS):
        ax = fig.add_subplot(2, 2, i + 1, projection="3d")
        for k, m in meshes.items():
            rgb = matplotlib.colors.to_rgb(COLORS.get(k, "#888888"))
            ax.add_collection3d(Poly3DCollection(m, facecolors=shade(m, rgb), edgecolors="none"))
        ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r); ax.set_zlim(c[2] - r, c[2] + r)
        ax.set_box_aspect((1, 1, 1)); ax.view_init(elev, azim); ax.set_axis_off(); ax.set_title(title)
    fig.suptitle(f"MicroPup leg (front left): thigh {t:.0f}, knee {q:.0f}, abduction {a:.0f} deg")
    fig.tight_layout()
    fig.savefig(out)
    print("wrote", out)


if __name__ == "__main__":
    args = [float(x) for x in sys.argv[2:5]]
    render(sys.argv[1] if len(sys.argv) > 1 else "leg.png", *args)
