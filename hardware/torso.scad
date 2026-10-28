// MicroCluck torso layout and shell parameters. Millimetres.
//
// Frame (the simulation's trunk frame): x forward, y to the robot's left, z up. The origin is the hip level
// of Pollen's trunk. Hips sit at x = 6 mm, y = +-17.5 mm; the hip-yaw servos top out at z = 22.5 mm.
//
// PART LAYOUT. These must match PARTS in the sim repo's robot/microcluck/build_walk_model.py: a test there
// compares this file (copied into the sim's assets by tools/build_shell.sh) against PARTS. Masses live in
// the sim; this file only owns geometry. Every layout value is a PLACEHOLDER until the parts are weighed
// and measured (PRD H3).

chassis_pos     = [4, 0, 0];       chassis_size     = [40, 60, 32];
hipservos_pos   = [6, 0, 11.5];    hipservos_size   = [24, 50, 22];
board_pos       = [30, 0, 75];     board_size       = [10, 56, 98];     // Freenove 3.5" board (guess)
bezel_pos       = [29, 0, 75];     bezel_size       = [14, 66, 108];
battery_pos     = [11.5, 0, 60];   battery_size     = [19, 38, 65];     // 2S pack
electronics_pos = [-8, 0, 55];     electronics_size = [16, 46, 40];     // INA226, logic buck, buffer, wiring
comb_pos        = [25, 0, 139];    comb_size        = [10, 12, 20];
tail_pos        = [-45, 0, 60];    tail_size        = [30, 30, 70];

// SHELL: a hollow body that runs back from the screen bezel's back face. Front open (the bezel closes it), rear wall
// with a tail opening, thin PETG-like walls.
bezel_back_x    = bezel_pos[0] - bezel_size[0] / 2;     // 22
tail_front_x    = tail_pos[0] + tail_size[0] / 2;       // -30: the placeholder tail block spans x = -60 .. -30
tail_rear_x     = tail_pos[0] - tail_size[0] / 2;       // -60

// Where the outer face of the rear wall sits. -64 puts the wall just behind the tail block's rear face (-60), so the
// body encloses that block (the block is the placeholder for the rear body mass; the real tail feathers attach to the
// rear wall). -30 stops the body at the block's front face and leaves it outside, as the first shell did.
shell_rear_x    = -64;

shell_wall      = 1.6;
// The numbers below were chosen with the sim's checks (scripts/microcluck_shell_clearance.py): the battery,
// electronics and tail blocks must sit inside the cavity, and the legs must stay clear of the shell over the stock
// walker's real gait poses plus a stress sweep of the joint ranges.
shell_floor_z   = 24;     // lowest point at the front, just above the hip servos (22.5); the battery's lower edge (27.5) clears it
shell_front_hw  = 33;     // half width at the bezel (= the bezel's half width)
shell_front_top = 127;    // just under the bezel top (129)
shell_rear_hw   = 24;     // half width at the rear wall: wide enough that the 46 mm electronics block fits
shell_rear_z0   = 20;     // rear floor: low enough that the 70 mm tail block (bottom at z = 25) sits inside the rounded rear edge
shell_rear_z1   = 102;    // rear top: the back slopes down from the bezel's top at about 20 degrees
shell_corner_r  = 5;      // corner radius of the front section (small, so the battery's corners fit)
shell_corner_rr = 12;     // corner radius of the rear section: rounds the rear corners away from the thighs

tail_socket_gap = 0.4;    // clearance around the tail's mounting stub
usb_slot        = [11, 4.5];     // USB-C opening in the rear underside: width (y), height (z)
switch_dia      = 6.2;           // power-switch hole in the right-hand wall
vent_slots      = 3;             // slots in the top, behind the bezel
vent_len        = 22;
vent_w          = 2;

density_g_cm3   = 1.27;   // PETG, for the mass estimate only

// SHELL STYLE. "box" is the hull of the two end caps above (straight sides). "egg" keeps the same two caps, so the
// bezel and tail interfaces do not change, but stacks slices between them so the body stays full behind the screen and
// then rounds off toward the tail, with side wing discs. Everything is convex, so the sim's clearance check still works.
shell_style     = "egg";
egg_slices      = 9;      // slices in the hull between the caps
egg_power_hw    = 2.2;    // half width: stays near the bezel's 33 mm, then falls to the rear cap's (higher = fuller)
egg_power_top   = 1.3;    // top: 1 slopes straight down from the bezel, higher keeps it high behind the screen first
egg_corner_pow  = 0.7;    // corner radius grows from the front cap's to the rear cap's (lower = rounder sooner)
egg_edge_r      = 8;      // radius of the rounded rear edge in the side view (0 = a square edge)
egg_wall_pad    = 0.4;    // extra inset of the cavity so the wall never thins below shell_wall (tuned with the sim's wall check)

// Wing discs on both sides (egg style only). Decorative, printed in a second colour, and a place for a cover or a
// later shoulder hub. They sit high on the body, well above the legs; the sim checks the vertical gap to the legs.
wing_dia        = 30;
wing_t          = 3.5;    // thickness standing proud of the wall
wing_dome       = 1.2;    // how much the rim is chamfered
wing_sink       = 1.0;    // how far the disc sinks into the wall, so the two fuse
wing_x          = 4;      // centre, along the body (the front edge stays clear of the bezel at x = 22)
wing_z          = 72;
