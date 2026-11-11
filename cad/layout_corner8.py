"""MicroPup layout study (11 Nov 2026): 8 servos, the hip and knee servo pairs as the corners of the body.

After Mini Pupper 4, whose motor drums are the body's corners with only flat leg parts outside them. Here each corner is
the Orion-style leg's holder (leg.py) fixed to the body: no abduction servo, plates or mount. The hip and knee servos sit
one above the other in it, and the knee is driven through the same two parallelogram links, so the shank's angle still
follows the knee servo directly. A layout study to measure widths and clearances, not a set of printable parts: the
deck, the parts inside and the face are stand-in boxes.

Two variants:
  A  the Orion leg's parts as they are. The thigh keeps its inner plate and bridge, so the left and right legs' inner
     plates swing in the middle of the body and everything between the front and rear pairs has to sit above them.
  B  no inner plate and no bridge: the thigh is held on the outside only, by the hip horn and by the shoulder cover,
     which is now a fixed part of the body. NOT DESIGNED: the cover carries the bellcrank's axle today, and the thigh
     plate would need a bearing on that axle (or a hollow boss around the horn's centre screw). With nothing swinging
     inside the body, the left and right holders sit back to back and the parts inside go between the pairs at hip height.

Body frame as body.py: x forward, y left, z up, z = 0 on the hip axes, mm.

Usage: cad/.venv/Scripts/python cad/layout_corner8.py [--no-fit] [--no-png] [--step]
       writes step/corner8-a.png and step/corner8-b.png; --step also writes the assemblies, step/corner8-a.step and
       step/corner8-b.step (about 4 MB each)
"""
import sys
from pathlib import Path

from build123d import Box, Compound, Plane, Pos, export_step

sys.path.insert(0, str(Path(__file__).parent))
import leg as LG  # noqa: E402
import servo as SV  # noqa: E402
from leg import bx  # noqa: E402

HERE = Path(__file__).parent
HIP_X = 55.0                       # hip axes at x = +-55, as the simulation and body.py
MIN_OVERLAP = 0.5                  # mm^3, as check_fit.py
MOVING = ("outer_plate", "inner_plate", "bridge", "shank", "foot", "bellcrank", "knee_crank", "long_link",
          "short_link", "hip_horn", "knee_horn")
DROP = ("abd_servo", "abd_horn", "abd_mount")
# Poses check_fit.py already clears for the leg on its own (hip from home, knee bend), so any hit here is the layout's.
POSES = ((0, LG.Q_HOME), (-35, LG.Q_HOME), (-35, 140), (-35, 100), (40, LG.Q_HOME), (70, LG.Q_HOME), (70, 45), (60, 30),
         (0, 140), (0, 30))
DECK_Z = (LG.HOOD_Z0 + LG.WALL, LG.HOOD_Z0 + LG.WALL + 3.0)   # a plate over the covers' hoods, 32.4 .. 35.4
FACE = Box(5.0, 33.0, 40.0)        # the 1.69 in LCD module (body.py BOARD)
ELECTRONICS, BATTERY = Box(60, 40, 12), Box(50, 30, 13)        # body.py INSIDE (placeholders from the sim's budget)

VARIANTS = {
    # leg_y: the servos' case bottoms from the middle of the body
    "a": dict(title="A: Orion leg parts as they are",
              leg_y=1.0 - LG.IP_Y0,            # 2 mm between the left and right inner plates
              without=(), keep_pin=True,
              inside={"electronics": Pos(0, 17.0, 26.2) * ELECTRONICS,    # side by side under the deck, above the thighs
                      "battery": Pos(0, -22.0, 25.7) * BATTERY}),
    "b": dict(title="B: thigh held on the outside only (to design)",
              leg_y=1.5 + LG.CAP_T,            # a 3 mm spine between the left and right holders' caps
              without=("inner_plate", "bridge"), keep_pin=False,
              inside={"electronics": Pos(0, 0, -2.0) * ELECTRONICS,       # stacked between the pairs at hip height
                      "battery": Pos(0, 0, 13.0) * BATTERY}),
}


def cradle(keep_pin):
    """The Orion holder without its abduction plates (they lie outside the cradle's length), and for variant B without
    the hip pin's standoff behind the cap."""
    y0 = LG.IP_Y1 - 0.6 if keep_pin else -LG.CAP_T
    return LG.part("holder") & bx(LG.CRADLE_X0, LG.CRADLE_X1, y0, LG.CRADLE_Y1 + 0.01, LG.CRADLE_Z0, LG.CRADLE_Z_TOP)


_CRADLES = {}


def leg_parts(v, front, left, dt=0.0, q=LG.Q_HOME):
    """One leg's parts in the body frame, hip moved dt degrees from home. front, left: +1 or -1."""
    cfg = VARIANTS[v]
    p = LG.posed(LG.T_HOME + dt, q, 0.0, rear=front < 0)
    if v not in _CRADLES:
        _CRADLES[v] = cradle(cfg["keep_pin"])
    p["holder"] = _CRADLES[v].mirror(Plane.YZ) if front < 0 else _CRADLES[v]
    out = {}
    for n, s in p.items():
        if n in DROP or n in cfg["without"]:
            continue
        if left < 0:
            s = s.mirror(Plane.XZ)
        out[n] = Pos(front * HIP_X, left * cfg["leg_y"], 0) * s
    return out


def body_parts(v):
    cfg = VARIANTS[v]
    half_w = cfg["leg_y"] + LG.G1
    x_half = HIP_X + max(-LG.CRADLE_X0, LG.CRADLE_X1) + 2.0
    out = {"deck": bx(-x_half, x_half, -half_w, half_w, *DECK_Z),
           "face": Pos(x_half + 6.0, 0, 10.0) * FACE}
    out.update(cfg["inside"])
    return out


def robot(v, dt=0.0, q=LG.Q_HOME):
    """{name: shape} for the whole robot, every leg at the same pose."""
    out = dict(body_parts(v))
    for front, fl in ((1, "f"), (-1, "r")):
        for left, ll in ((1, "l"), (-1, "r")):
            for n, s in leg_parts(v, front, left, dt, q).items():
                out[f"{fl}{ll}_{n}"] = s
    return out


def _hits(moving, fixed):
    """Overlaps above MIN_OVERLAP between each moving shape and each fixed one."""
    hits = []
    boxes = {n: s.bounding_box() for n, s in fixed.items()}
    for mn, ms in moving.items():
        mb = ms.bounding_box()
        for fn, fs in fixed.items():
            fb = boxes[fn]
            if (mb.min.X > fb.max.X or fb.min.X > mb.max.X or mb.min.Y > fb.max.Y or fb.min.Y > mb.max.Y
                    or mb.min.Z > fb.max.Z or fb.min.Z > mb.max.Z):
                continue
            vol = (ms & fs).volume
            if vol > MIN_OVERLAP:
                hits.append((mn, fn, vol))
    return hits


def fit(v):
    """Each left leg's moving parts, over POSES, against the body's stand-ins, the other left leg's fixed parts and the
    right leg opposite. Returns the number of poses with a hit."""
    bad = 0
    body = body_parts(v)
    for front, name in ((1, "front"), (-1, "rear")):
        other = {f"other_{n}": s for n, s in leg_parts(v, -front, 1).items() if n not in MOVING}
        for dt, q in POSES:
            mine = leg_parts(v, front, 1, dt, q)
            moving = {n: s for n, s in mine.items() if n in MOVING}
            opposite = {f"opposite_{n}": s for n, s in leg_parts(v, front, -1, dt, q).items()}
            hits = _hits(moving, {**body, **other, **opposite})
            bad += bool(hits)
            print(f"  {name:5s} hip {dt:+4.0f} knee {q:5.1f} | {'ok' if not hits else 'HIT'}")
            for a, b, vol in hits:
                print(f"        {a} x {b}: {vol:.1f} mm^3")
    return bad


def measure(v):
    cfg = VARIANTS[v]
    shapes = robot(v)
    legs = Compound(children=[s for n, s in shapes.items() if n[:2] in ("fl", "fr", "rl", "rr")])
    bb = legs.bounding_box()
    y = cfg["leg_y"]
    return dict(overall=bb.max.Y - bb.min.Y, plates=2 * (y + LG.OP_Y1), feet=2 * (y + LG.Y_MID),
                length=bb.max.X - bb.min.X, between_pairs=2 * (HIP_X - max(-LG.CRADLE_X0, LG.CRADLE_X1)),
                foot_z=bb.min.Z, shapes=shapes)


def render(v, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    import render as RD

    colors = dict(RD.COLORS, deck="#f2c230", face="#f2c230", electronics="#3f8f5a", battery="#4a6fb5")
    shapes = robot(v)
    meshes = {k: RD.mesh(s, 0.3) for k, s in shapes.items()}
    allv = np.concatenate([m.reshape(-1, 3) for m in meshes.values()])
    lo, hi = allv.min(0), allv.max(0)
    c, r = (lo + hi) / 2, 0.62 * (hi - lo).max()
    fig = plt.figure(figsize=(18, 6.5), dpi=100)
    for i, (elev, azim, title) in enumerate(((22, -60, "front-left"), (0, 0, "front"), (0, -90, "left side"))):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        for k, m in meshes.items():
            base = k.split("_", 1)[1] if k[:2] in ("fl", "fr", "rl", "rr") else k
            rgb = matplotlib.colors.to_rgb(colors.get(base, "#888888"))
            ax.add_collection3d(Poly3DCollection(m, facecolors=RD.shade(m, rgb), edgecolors="none"))
        ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r); ax.set_zlim(c[2] - r, c[2] + r)
        ax.set_box_aspect((1, 1, 1)); ax.view_init(elev, azim); ax.set_axis_off(); ax.set_title(title)
    fig.suptitle(f"MicroPup, 8 servos as the body's corners. {VARIANTS[v]['title']}")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


def main():
    step = HERE / "step"
    step.mkdir(exist_ok=True)
    print("reference (README): braced 12-servo legs 141 mm across the shoulders, feet 95 apart; Orion 12-servo 184 and 153")
    for v, cfg in VARIANTS.items():
        m = measure(v)
        print(f"\n{cfg['title']}")
        print(f"  overall width {m['overall']:.1f} mm (over the shoulder covers); over the outer thigh plates "
              f"{m['plates']:.1f}; feet {m['feet']:.1f} apart")
        print(f"  legs and holders {m['length']:.1f} mm long; {m['between_pairs']:.1f} mm between the front and rear "
              f"holders; feet {-m['foot_z']:.1f} mm below the hip axes at the home pose")
        if "--step" in sys.argv:
            export_step(Compound(children=list(m["shapes"].values())), str(step / f"corner8-{v}.step"))
        if "--no-fit" not in sys.argv:
            bad = fit(v)
            print(f"  {2 * len(POSES) - bad}/{2 * len(POSES)} poses clear")
        if "--no-png" not in sys.argv:
            render(v, str(step / f"corner8-{v}.png"))


if __name__ == "__main__":
    main()
