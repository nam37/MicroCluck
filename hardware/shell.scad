// MicroCluck body shell: joins the screen bezel's back face to the tail's front face.
//   openscad -o build/shell.stl          shell.scad     the hollow, printable shell
//   openscad -D 'part="envelope"' -o build/shell_envelope.stl shell.scad   the solid outer shape (clearance checks)
//   openscad -D 'part="cavity"'   -o build/shell_cavity.stl   shell.scad   the inner space (fit checks)
//   openscad -D 'part="wings"'    -o build/wings.stl          shell.scad   the two wing discs (egg style)
//   add -D 'shell_style="box"' to build the straight-sided shell instead of the egg
include <torso.scad>

envelope = false;   // true: same as part = "envelope" (kept for the older build command)
part = "shell";     // "shell" | "envelope" | "cavity" | "wings"
$fn = 48;

egg = shell_style == "egg";

// A thin rounded-rectangle slab at position x (axis along x): used as the end caps of a hull.
module slab(x, hw, z0, z1, r) {
    hull()
        for (sy = [-1, 1], z = [z0 + r, z1 - r])
            translate([x, sy * (hw - r), z]) rotate([0, 90, 0]) cylinder(r = r, h = 0.01);
}

// ---- egg profile: t = 0 at the bezel's back face, 1 at the tail's front face ----
function lerp(a, b, u) = a + (b - a) * u;
function t_of(x) = max(0, min(1, (bezel_back_x - x) / (bezel_back_x - shell_rear_x)));
function p_hw(t) = lerp(shell_rear_hw, shell_front_hw, 1 - pow(t, egg_power_hw));
function p_z1(t) = lerp(shell_rear_z1, shell_front_top, 1 - pow(t, egg_power_top));
function p_z0(t) = lerp(shell_floor_z, shell_rear_z0, t);
function p_r(t)  = lerp(shell_corner_r, shell_corner_rr, pow(t, egg_corner_pow));

// Rounded rear edge in the side view (radius r, 0 = square). The slices near the rear wall shrink by the circle's sag,
// so the top-rear and bottom-rear corners are rounded the way the sides already are.
function fil(d, r) = (r <= 0 || d >= r) ? 0 : r - sqrt(r * r - (r - d) * (r - d));
function fil_steps(r) = [for (u = [0, 0.06, 0.15, 0.28, 0.45, 0.65, 0.85]) u * r];

module outer_egg() {
    hull() {
        for (i = [0 : egg_slices - 1]) {
            t = i / egg_slices;
            slab(lerp(bezel_back_x, shell_rear_x, t), p_hw(t), p_z0(t), p_z1(t), p_r(t));
        }
        for (d = fil_steps(egg_edge_r)) {
            x = shell_rear_x + d;
            t = t_of(x);
            e = fil(d, egg_edge_r);
            slab(x, p_hw(t) - e, p_z0(t) + e, p_z1(t) - e, max(p_r(t) - e, 0.5));
        }
    }
}

// Slope of a profile along x (mm per mm) at t. A wall measured along the surface normal is thicker than the
// in-plane inset of a slice by a factor sqrt(1 + slope^2), so each slice is inset by the wall times that factor.
function slope(a, b, power, t) = abs(a - b) * power * pow(t, power - 1) / (bezel_back_x - shell_rear_x);
// egg_wall_pad covers what the slope term misses: at the rear corners the surface slopes and curves at once
function wall_at(s) = shell_wall * sqrt(1 + s * s) + egg_wall_pad;

// One slice of the cavity at x: the wall insets follow the local slope, e is the extra inset of the rounded rear edge.
module cavity_slice(x, t, e) {
    side = wall_at(slope(shell_front_hw, shell_rear_hw, egg_power_hw, t));
    top = wall_at(slope(shell_front_top, shell_rear_z1, egg_power_top, t));
    // the floor is nearly flat, so it only needs the pad at the rear corners, behind the battery
    floor_ = wall_at(abs(shell_rear_z0 - shell_floor_z) / (bezel_back_x - shell_rear_x))
             - egg_wall_pad * (1 - max(0, min(1, (t - 0.44) / 0.16)));
    slab(x, p_hw(t) - side - e, p_z0(t) + floor_ + e, p_z1(t) - top - e, max(p_r(t) - max(side, top) - e, 0.5));
}

module cavity_egg() {
    // open at the front (starts 2 mm in front of the bezel face), closed by a wall at the rear
    rc = max(egg_edge_r - shell_wall, 0);   // the offset of a fillet is a fillet around the same centre
    hull() {
        for (i = [0 : egg_slices - 1]) {
            x = lerp(bezel_back_x + 2, shell_rear_x + shell_wall, i / egg_slices);
            cavity_slice(x, i == 0 ? 0 : t_of(x), 0);
        }
        for (d = fil_steps(rc)) {
            x = shell_rear_x + shell_wall + d;
            cavity_slice(x, t_of(x), fil(d, rc));
        }
    }
}

// ---- straight-sided box: the hull of the two end caps ----
module outer_box() {
    hull() {
        slab(bezel_back_x, shell_front_hw, shell_floor_z, shell_front_top, shell_corner_r);
        slab(shell_rear_x, shell_rear_hw, shell_rear_z0, shell_rear_z1, shell_corner_rr);
    }
}

module cavity_box() {
    hull() {
        slab(bezel_back_x + 2, shell_front_hw - shell_wall, shell_floor_z + shell_wall,
             shell_front_top - shell_wall, shell_corner_r - shell_wall);
        slab(shell_rear_x + shell_wall, shell_rear_hw - shell_wall, shell_rear_z0 + shell_wall,
             shell_rear_z1 - shell_wall, shell_corner_rr - shell_wall);
    }
}

module outer()  { if (egg) outer_egg();  else outer_box(); }
module cavity() { if (egg) cavity_egg(); else cavity_box(); }

module cutouts() {
    // tail socket through the rear wall: the tail's footprint plus clearance
    translate([shell_rear_x - 1, -(tail_size[1] / 2 - 3 + tail_socket_gap), tail_pos[2] - (tail_size[2] / 2 - 6) - tail_socket_gap])
        cube([shell_wall + 2, tail_size[1] - 6 + 2 * tail_socket_gap, tail_size[2] - 12 + 2 * tail_socket_gap]);
    // USB-C slot, rear underside
    translate([shell_rear_x + 6, -usb_slot[0] / 2, shell_rear_z0 - 1]) cube([10, usb_slot[0], usb_slot[1] + 1]);
    // power-switch hole, right-hand wall (-y), behind the battery. The egg moves it behind the wing disc.
    sx = egg ? electronics_pos[0] - 10 : electronics_pos[0];
    translate([sx, -shell_front_hw - 2, electronics_pos[2] + 8]) rotate([-90, 0, 0]) cylinder(d = switch_dia, h = shell_front_hw - 14);
    // vent slots in the top, behind the bezel (tall cuts: the egg's top is higher than the box's here)
    for (i = [0 : vent_slots - 1])
        translate([bezel_back_x - 8 - i * 6, -vent_len / 2, 40]) cube([vent_w, vent_len, 100]);
}

// One wing disc on the +y side, its axis along the local surface normal so it lies flat on the sloping wall.
module wing_left() {
    t = t_of(wing_x);
    lean = atan((p_hw(t_of(wing_x + 1)) - p_hw(t_of(wing_x - 1))) / 2);   // wall slope along x
    translate([wing_x, p_hw(t) - wing_sink, wing_z])
        rotate([0, 0, lean]) rotate([-90, 0, 0])
            hull() {
                cylinder(d = wing_dia, h = wing_sink + wing_t - wing_dome);
                cylinder(d = wing_dia - 2 * wing_dome, h = wing_sink + wing_t);
            }
}

module wings() {
    wing_left();
    mirror([0, 1, 0]) wing_left();
}

if (part == "wings") { if (egg) wings(); }
else if (part == "envelope" || envelope) outer();
else if (part == "cavity") cavity();
else difference() {
    outer();
    cavity();
    cutouts();
}
