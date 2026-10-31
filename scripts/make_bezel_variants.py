"""make_bezel_variants.py : copies of the final hips-back stand-up model with bumper lips on the bezel's top and bottom edges.

    uv run python scripts/make_bezel_variants.py 4 8 12        # lips standing 4, 8 and 12 mm proud of the frame face

writes robot_standup_back29_fix_fan30_lip<mm>.xml next to the model. The lip shape is hardware/bezel.scad's (torso.scad: bezel_lip_proud,
bezel_lip_r); masses are unchanged (the lips are in the bezel_and_bumpers remainder of the budget).
"""
import sys
from pathlib import Path

from mjlab_microduck.robot.microcluck.build_standup_model import BEZEL_LIP_RADIUS, insert_bezel_lips

D = Path(__file__).resolve().parent.parent / "src/mjlab_microduck/robot/microcluck"
SRC = D / "robot_standup_back29_fix_fan30.xml"

for arg in sys.argv[1:]:
    proud_mm = float(arg)
    dst = D / f"robot_standup_back29_fix_fan30_lip{arg}.xml"
    dst.write_text(insert_bezel_lips(SRC.read_text(), proud_mm / 1000.0))
    print("wrote", dst.name, f"(lips {proud_mm:g} mm proud, radius {1000 * BEZEL_LIP_RADIUS:g} mm)")
