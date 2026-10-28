// MicroCluck body shell: joins the screen bezel's back face to the tail's front face.
//   openscad -o build/shell.stl          shell.scad     the hollow, printable shell
//   openscad -D 'envelope=true' -o build/shell_envelope.stl shell.scad   the solid outer shape (clearance checks)
include <torso.scad>

envelope = false;   // true: the solid outer shape, no cavity and no cut-outs
$fn = 48;

// A thin rounded-rectangle slab at position x (axis along x): used as the end caps of a hull.
module slab(x, hw, z0, z1, r) {
    hull()
        for (sy = [-1, 1], z = [z0 + r, z1 - r])
            translate([x, sy * (hw - r), z]) rotate([0, 90, 0]) cylinder(r = r, h = 0.01);
}

module outer() {
    hull() {
        slab(bezel_back_x, shell_front_hw, shell_floor_z, shell_front_top, shell_corner_r);
        slab(tail_front_x, shell_rear_hw, shell_rear_z0, shell_rear_z1, shell_corner_rr);
    }
}

module cavity() {
    // open at the front (starts 2 mm in front of the bezel face), closed by a wall at the rear
    hull() {
        slab(bezel_back_x + 2, shell_front_hw - shell_wall, shell_floor_z + shell_wall,
             shell_front_top - shell_wall, shell_corner_r - shell_wall);
        slab(tail_front_x + shell_wall, shell_rear_hw - shell_wall, shell_rear_z0 + shell_wall,
             shell_rear_z1 - shell_wall, shell_corner_rr - shell_wall);
    }
}

module cutouts() {
    // tail socket through the rear wall: the tail's footprint plus clearance
    translate([tail_front_x - 1, -(tail_size[1] / 2 - 3 + tail_socket_gap), tail_pos[2] - (tail_size[2] / 2 - 6) - tail_socket_gap])
        cube([shell_wall + 2, tail_size[1] - 6 + 2 * tail_socket_gap, tail_size[2] - 12 + 2 * tail_socket_gap]);
    // USB-C slot, rear underside
    translate([tail_front_x + 6, -usb_slot[0] / 2, shell_rear_z0 - 1]) cube([10, usb_slot[0], usb_slot[1] + 1]);
    // power-switch hole, right-hand wall (-y), behind the battery
    translate([electronics_pos[0], -shell_rear_hw - 4, electronics_pos[2] + 8]) rotate([-90, 0, 0]) cylinder(d = switch_dia, h = 12);
    // vent slots in the top, behind the bezel
    for (i = [0 : vent_slots - 1])
        translate([bezel_back_x - 8 - i * 6, -vent_len / 2, 40]) cube([vent_w, vent_len, 80]);
}

if (envelope) outer();
else difference() {
    outer();
    cavity();
    cutouts();
}
