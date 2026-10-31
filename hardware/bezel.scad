// MicroCluck screen bezel: the frame around the glass, with the bumper lips on its top and bottom edges. Millimetres, in the
// sim's trunk frame (x forward, y left, z up); the layout and the lip parameters are in torso.scad.
//   openscad -o build/bezel.stl bezel.scad                          the frame with its lips: the printable shape
//   openscad -D 'part="lips"' -o build/bezel_lips.stl bezel.scad    the two lips alone
//   add -D bezel_lip_proud=8 to try another lip
// The sim collides with the same shape as primitives built from the same numbers (build_standup_model.py: a capsule across the width
// for each lip, and a web box from the frame face to it), so this file and the sim agree through torso.scad.
include <torso.scad>

part = "bezel";   // "bezel" | "lips"
$fn = 48;

hw = bezel_size[1] / 2;
x0 = bezel_pos[0] - bezel_size[0] / 2;
x1 = bezel_pos[0] + bezel_size[0] / 2;   // the frame's front face
z0 = bezel_pos[2] - bezel_size[2] / 2;
z1 = bezel_pos[2] + bezel_size[2] / 2;
zc = bezel_pos[2];

// One lip: a capsule across the width (its centre line hw - r in from each side) joined to the frame face by a thin slab.
module lip(z) {
    r = bezel_lip_r;
    xc = x1 + bezel_lip_proud - r;
    hull() {
        for (s = [-1, 1]) translate([xc, s * (hw - r), z]) sphere(r = r);
        translate([x1 - 1, -hw, z - r]) cube([1, 2 * hw, 2 * r]);
    }
}

module lips() {
    if (bezel_lip_proud > 0) {
        lip(z1 - bezel_lip_r);
        lip(z0 + bezel_lip_r);
    }
}

module frame() {
    difference() {
        translate([x0, -hw, z0]) cube([x1 - x0, 2 * hw, z1 - z0]);
        // the window for the glass, recessed behind the front face
        translate([x1 - bezel_glass_recess, -bezel_window[0] / 2, zc - bezel_window[1] / 2])
            cube([bezel_glass_recess + 1, bezel_window[0], bezel_window[1]]);
    }
}

if (part == "lips") lips();
else union() { frame(); lips(); }
