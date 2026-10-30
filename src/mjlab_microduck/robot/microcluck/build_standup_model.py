#!/usr/bin/env python3
"""Build MicroCluck's stand-up (body-on-ground) model from the generated walking model.

The walking model has collision on the soles only, because the walking task never touches anything else. A
fall-recovery task needs the rest of the robot to hit the floor, so this adds:

  - Pollen's own curated ground-contact set on the legs (the lower-leg mesh, which the walking model keeps as
    self-collision only, and the hip block), as in microduck/robot_groundcontact.xml
  - the torso: the bezel plate, the body shell (the convex hull of hardware/shell.scad's envelope), the comb, the tail
    and the two wing discs, all rigid and flat against the body

Each torso piece is named ``*_collision`` so mjlab's FULL_COLLISION rule gives it the same contact settings as the
soles. Masses are unchanged: they live in the trunk inertial that build_walk_model.py wrote.

Run after build_walk_model.py, from anywhere:

    uv run python src/mjlab_microduck/robot/microcluck/build_standup_model.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    from . import build_walk_model as walk
except ImportError:  # run as a script
    import build_walk_model as walk

HERE = Path(__file__).resolve().parent
WALK_XML = HERE / "robot_walk.xml"
STANDUP_XML = HERE / "robot_standup.xml"
SHELL_ENVELOPE = walk.ASSETS / "shell_envelope.stl"
WINGS = walk.ASSETS / "wings.stl"


def fmt(values) -> str:
    return " ".join(f"{v:.6g}" for v in values)


def part(name: str) -> "walk.Part":
    return next(p for p in walk.PARTS if p.name == name)


def trunk_collision_geoms(indent: str) -> list[str]:
    """The torso collision geoms, in the trunk frame (x forward, y left, z up), as MJCF lines."""
    out = [f"{indent}<!-- MicroCluck torso collision (rigid; the body shell is the convex hull of the envelope mesh) -->\n"]
    for geom_name, p in (("bezel", part("bezel_and_bumpers")), ("comb", part("comb")), ("tail", part("tail"))):
        half = [s / 2 for s in p.size]
        out.append(f'{indent}<geom type="box" class="collision" name="{geom_name}_collision" size="{fmt(half)}" '
                   f'pos="{fmt(p.pos)}" rgba="0.95 0.72 0.02 0.0"/>\n')
    if SHELL_ENVELOPE.exists():
        out.append(f'{indent}<geom type="mesh" class="collision" name="shell_collision" mesh="mc_shell_env" '
                   f'rgba="0.95 0.72 0.02 0.0"/>\n')
    if WINGS.exists():
        pts = walk.read_stl_triangles(WINGS).reshape(-1, 3)
        for side, mask in (("left", pts[:, 1] > 0), ("right", pts[:, 1] < 0)):
            lo, hi = pts[mask].min(axis=0), pts[mask].max(axis=0)
            centre = (lo + hi) / 2
            radius = max(hi[0] - lo[0], hi[2] - lo[2]) / 2
            half_thickness = (hi[1] - lo[1]) / 2
            # a disc whose axis is the body's y axis: rotate the cylinder's z axis onto y
            out.append(f'{indent}<geom type="cylinder" class="collision" name="wing_{side}_collision" '
                       f'size="{radius:.6g} {half_thickness:.6g}" pos="{fmt(centre)}" quat="0.70710678 -0.70710678 0 0" '
                       f'rgba="0.95 0.72 0.02 0.0"/>\n')
    return out


def standup_lines(lines: list[str]) -> list[str]:
    out: list[str] = []
    seen_leg = seen_hip = seen_trunk = 0
    for l in lines:
        indent = l[: len(l) - len(l.lstrip())]
        if 'class="self_collision_only"' in l and 'mesh="leg"' in l:
            out.append(l.replace('class="self_collision_only"', 'class="collision"'))
            seen_leg += 1
            continue
        if l.strip() == "</asset>" and SHELL_ENVELOPE.exists():
            out.append('    <mesh name="mc_shell_env" file="../../microcluck/assets/shell_envelope.stl" scale="0.001 0.001 0.001"/>\n')
        out.append(l)
        if 'class="visual"' in l and 'mesh="hip_l"' in l:
            out.append(l.replace('class="visual"', 'class="collision"'))
            seen_hip += 1
        if 'name="screen_face_visual"' in l:
            out.extend(trunk_collision_geoms(indent))
            seen_trunk += 1
    assert seen_leg == 2 and seen_hip == 2 and seen_trunk == 1, (seen_leg, seen_hip, seen_trunk)
    return out


def build() -> None:
    lines = WALK_XML.read_text().splitlines(keepends=True)
    text = "".join(standup_lines(lines))
    text = text.replace("<!-- MicroCluck walking model.", "<!-- MicroCluck stand-up model (walking model + ground-contact collision).", 1)
    STANDUP_XML.write_text(text)
    print(f"wrote {STANDUP_XML.name}")


if __name__ == "__main__":
    build()
