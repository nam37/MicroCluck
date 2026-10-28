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

// SHELL: a hollow body that joins the screen bezel's back face to the tail's front face. Front open
// (the bezel closes it), rear wall with a tail socket, thin PETG-like walls.
bezel_back_x    = bezel_pos[0] - bezel_size[0] / 2;     // 22
tail_front_x    = tail_pos[0] + tail_size[0] / 2;       // -30

shell_wall      = 1.6;
// The numbers below were chosen with the sim's checks (scripts/microcluck_shell_clearance.py): the battery and
// electronics blocks must sit inside the cavity, and the legs must stay clear of the shell over the stock walker's
// real gait poses plus a stress sweep of the joint ranges.
shell_floor_z   = 24;     // lowest point at the front, just above the hip servos (22.5); the battery's lower edge (27.5) clears it
shell_front_hw  = 33;     // half width at the bezel (= the bezel's half width)
shell_front_top = 124;    // just under the bezel top (129)
shell_rear_hw   = 24;     // half width at the tail: wide enough that the 46 mm electronics block fits
shell_rear_z0   = 27;     // rear floor, a little higher than the front: the thighs swing back under it
shell_rear_z1   = 94;     // rear top: the back slopes down to the tail's top (95)
shell_corner_r  = 5;      // corner radius of the front section (small, so the battery's corners fit)
shell_corner_rr = 16;     // corner radius of the rear section: rounds the lower rear corners away from the thighs

tail_socket_gap = 0.4;    // clearance around the tail's mounting stub
usb_slot        = [11, 4.5];     // USB-C opening in the rear underside: width (y), height (z)
switch_dia      = 6.2;           // power-switch hole in the right-hand wall
vent_slots      = 3;             // slots in the top, behind the bezel
vent_len        = 22;
vent_w          = 2;

density_g_cm3   = 1.27;   // PETG, for the mass estimate only
