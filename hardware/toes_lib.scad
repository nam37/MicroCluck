// Shared geometry for the chunky toes: printable part, simulation visual and collision hull. See params.scad.
include <params.scad>

side = "left";   // override with: openscad -D 'side="right"' ...
$fn = 40;

// A rounded post: a cylinder topped with a flattened sphere, total height h, from z = 0.
module pill(r, h) {
    hull() {
        cylinder(r = r, h = h - 0.6 * r);
        translate([0, 0, h - 0.6 * r]) scale([1, 1, 0.6]) sphere(r);
    }
}

// One toe lobe: hull of a tall base post at the wall and a lower tip post, sloping down and out.
module lobe(reach, ang, y0) {
    tip = [front_wall_x, y0] + (reach - toe_tip_radius) * [cos(ang), sin(ang)];
    hull() {
        translate([front_wall_x, y0, 0]) pill(toe_base_radius, toe_base_height);
        translate([tip[0], tip[1], 0]) pill(toe_tip_radius, toe_tip_height);
    }
}

// Left foot: the outer toe points away from the midline (+y), the inner toe toward it (-y).
// Clipped so nothing enters the cavity inside Pollen's front wall.
module toes3d() {
    intersection() {
        union() {
            lobe(toe_reach_middle, 0, toe_center_y);
            lobe(toe_reach_side, toe_splay_deg, toe_center_y + toe_pitch);
            lobe(toe_reach_side - inner_toe_trim, -toe_splay_deg, toe_center_y - toe_pitch);
        }
        translate([wall_inner_x, -200, -1]) cube([400, 400, 200]);
    }
}

module oriented() {
    if (side == "left") children();
    else mirror([0, 1, 0]) children();
}
