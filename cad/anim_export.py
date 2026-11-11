"""Exports the meshes and constants for animate.py: every leg part in its own joint frame, in the four mirror variants
(fl: as modelled; fr: mirrored left-right; rl: the holder side mirrored front-back; rr: both), plus a body shell.

Usage: cad/.venv/Scripts/python cad/anim_export.py      (writes cad/anim/; git-ignored)
"""
import json
import sys
from pathlib import Path

from build123d import Plane, Pos, Rot, Box, Cylinder, Sphere, export_stl, fillet

sys.path.insert(0, str(Path(__file__).parent))
import leg as LG  # noqa: E402
import servo as SV  # noqa: E402

OUT = Path(__file__).parent / "anim"

# Robot layout for the picture (PLACEHOLDER body: the simulation's micro trunk, narrowed to 44 mm so the legs' pins
# clear its side walls; the abduction axes end up 14.5 mm either side of the middle, the sim has 12).
HIP_X = 55.0
BODY_HALF_W = 22.0
LEG_Y = BODY_HALF_W + 12.5        # hip servo case bottom from the middle
BODY = dict(x0=-85.0, x1=85.0, z0=-27.0, z1=36.0)


def local_parts():
    """{name: shape} in the frames animate.py poses them in."""
    p = {n: LG.part(n) for n in ("holder", "cover", "outer_plate", "inner_plate", "bridge", "shank", "foot",
                                 "bellcrank", "knee_crank", "long_link", "short_link", "abd_mount")}
    p["hip_servo"] = SV.body()
    p["knee_servo"] = SV.body()
    p["abd_servo"] = LG.abd_frame() * SV.body()
    p["hip_horn"] = SV.horn(SV.HORN_ARM_L, -90.0)
    p["knee_horn"] = SV.horn(SV.HORN_ARM_L_CUT, -90.0)
    p["abd_horn"] = LG.abd_frame() * SV.horn(SV.HORN_ARM_L, 180.0)
    return p


def body_parts():
    b = Pos((BODY["x0"] + BODY["x1"]) / 2, 0, (BODY["z0"] + BODY["z1"]) / 2) * Box(
        BODY["x1"] - BODY["x0"], 2 * BODY_HALF_W, BODY["z1"] - BODY["z0"])
    b = fillet(b.edges(), 8.0)
    bezel = fillet((Pos(BODY["x1"] + 2.0, 0, 6.0) * Box(5.0, 40.0, 54.0)).edges(), 2.0)
    screen = Pos(BODY["x1"] + 4.6, 0, 8.0) * Box(0.6, 28.0, 33.0)
    eyes = Pos(BODY["x1"] + 5.0, 6.5, 12.0) * Rot(0, 90, 0) * Cylinder(3.2, 0.4)
    eyes += Pos(BODY["x1"] + 5.0, -6.5, 12.0) * Rot(0, 90, 0) * Cylinder(3.2, 0.4)
    tail = Pos(BODY["x0"] + 8, 0, BODY["z1"] + 14) * Rot(0, -20, 0) * Cylinder(3.5, 30.0)
    tail += Pos(BODY["x0"] + 3, 0, BODY["z1"] + 30) * Sphere(5.0)
    return dict(body=b, bezel=bezel, screen=screen, eyes=eyes, tail=tail)


def main():
    OUT.mkdir(exist_ok=True)
    for name, s in local_parts().items():
        side = name in LG.HOLDER_SIDE
        variants = {"fl": s, "fr": s.mirror(Plane.XZ)}
        variants["rl"] = s.mirror(Plane.YZ) if side else s
        variants["rr"] = variants["rl"].mirror(Plane.XZ)
        for v, shape in variants.items():
            export_stl(shape, str(OUT / f"{name}_{v}.stl"), tolerance=0.03, angular_tolerance=0.2)
    for name, s in body_parts().items():
        export_stl(s, str(OUT / f"{name}.stl"), tolerance=0.05, angular_tolerance=0.2)
    consts = {k: getattr(LG, k) for k in ("L_THIGH", "L_SHANK", "T_HOME", "Q_HOME", "KS_Z", "ABD_Y", "RB", "RS",
                                          "ALPHA", "BETA", "KNEE_MID", "KNEE_WINDOW", "FOOT_R")}
    consts.update(HIP_X=HIP_X, LEG_Y=LEG_Y, BODY=BODY, HOLDER_SIDE=list(LG.HOLDER_SIDE),
                  parts=list(local_parts()), body_parts=list(body_parts()))
    (OUT / "consts.json").write_text(json.dumps(consts, indent=1))
    print("wrote", OUT, len(list(OUT.glob("*.stl"))), "meshes")


if __name__ == "__main__":
    main()
