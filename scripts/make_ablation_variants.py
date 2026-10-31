"""make_ablation_variants.py : copies of the final hips-back stand-up model in which one group of trunk collision geoms is moved into its own
massless child body that cannot touch any leg part (the geom still collides with the floor). Which trunk part do the recovery motions lean on?"""
from pathlib import Path

D = Path.home() / "microcluck_standup/src/mjlab_microduck/robot/microcluck"
SRC = D / "robot_standup_back29_fix_fan30.xml"
GROUPS = {
    "shell": ["shell_collision"],
    "tail": ["tail_collision"],
    "wings": ["wing_left_collision", "wing_right_collision"],
    "bezelcomb": ["bezel_collision", "comb_collision"],
}
LEG_BODIES = ["hip_l", "upper_leg_left", "leg", "ankle_left", "hip_l_2", "upper_leg_right", "leg_2", "ankle_right"]

text = SRC.read_text()
for label, names in GROUPS.items():
    lines = text.split("\n")
    taken, kept = [], []
    for ln in lines:
        if any(f'name="{n}"' in ln for n in names) and 'class="collision"' in ln:
            taken.append(ln)
        else:
            kept.append(ln)
    assert len(taken) == len(names), (label, len(taken))
    part = f"abl_{label}"
    body = [f'      <body name="{part}">',
            '        <inertial pos="0 0 0" mass="1e-5" diaginertia="1e-10 1e-10 1e-10"/>', *["  " + t for t in taken], "      </body>"]
    i = next(k for k, ln in enumerate(kept) if 'name="yaw2roll"' in ln and "<body" in ln)  # before the first leg body, still inside trunk_base
    kept[i:i] = body
    out = "\n".join(kept)
    anchor = '    <exclude body1="trunk_base" body2="hip_l_2"/>\n'
    assert out.count(anchor) == 1
    out = out.replace(anchor, anchor + "".join(f'    <exclude body1="{part}" body2="{b}"/>\n' for b in LEG_BODIES))
    dst = D / f"robot_standup_back29_fix_fan30_abl_{label}.xml"
    dst.write_text(out)
    print("wrote", dst.name, "moved", names)
