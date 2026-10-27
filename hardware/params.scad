// MicroCluck foot toes: the single parameter table behind the printed toes AND the simulation.
// Millimetres and degrees.
//
// The foot itself is Pollen's, unchanged. Three chunky, rounded toes are fused onto the outer face of the front
// wall of Pollen's sole (a 2 mm shell: material from x = 31.9 to 34.0 at z = 5..13, measured from their STL).
// The toes run down to the floor, so they are part of the contact patch: the simulation collides with the
// convex hull of Pollen's sole plus the toes.
//
// Frame ("stand-pose foot frame"): x forward, y to the robot's LEFT, z up, origin on the ground directly under
// the ankle axis, axes aligned with the world while the robot stands in its home pose. These numbers describe
// the LEFT foot; the right foot is the mirror image across y = 0 (done in toes_lib.scad).

front_wall_x      = 34;     // outer face of Pollen's sole front wall, ahead of the ankle axis
wall_inner_x      = 31.9;   // inner face of that wall: the toes are clipped to x >= this so they never reach the
                            // cavity around Pollen's foot (their foot ends at x = 29.4)
toe_center_y      = -16.2;  // centerline of the foot, relative to the ankle axis (negative = toward the midline)
toe_pitch         = 8;      // lateral spacing of the three toe bases along the wall
toe_base_radius   = 6;      // the three bases overlap, so the toes read as one webbed foot with grooves at the tips
toe_tip_radius    = 5.5;
toe_reach_middle  = 12;     // beyond the front wall: the middle toe tip is 46 mm ahead of the ankle axis
toe_reach_side    = 11;
toe_splay_deg     = 22;     // angle of the side toes from straight ahead
toe_base_height   = 12;     // height at the wall (the wall is 13 mm tall)
toe_tip_height    = 7;      // the toes slope down toward the tips
inner_toe_trim    = 0;      // mm to shorten the inner toe (PRD: shorten it if the feet clip on narrow steps)

// Printing / mass estimate only
density_g_cm3     = 1.21;   // TPU; PLA is about 1.24
infill_fraction   = 0.30;
