// Maps Pollen's sole/foot/ankle STL frame (metres) into the stand-pose foot frame (millimetres).
// Derived from forward kinematics of Pollen's model at the home pose with the sim repo's
// robot/microcluck/pollen_foot_frame.py. Independent of the hip-pitch/ankle trim (those cancel in the
// foot's orientation), so it only changes if the leg geometry or the 5 degree hip roll changes.
// After the transform Pollen's sole spans x -20..34, y -36.7..4.5, z 0..13.5 (left foot).
M_pollen_left = [
    [ 0.00000, -1000.00000,   0.00000,  22.00000],
    [ 996.19470,  0.00000,   87.15574, -63.91143],
    [-87.15574,   0.00000,  996.19470,  34.38879],
    [ 0, 0, 0, 1]
];
M_pollen_right = [
    [ 0.00000, -1000.00000,   0.00000,  22.00000],
    [ 996.19470,  0.00000,  -87.15574,  63.91143],
    [ 87.15574,   0.00000,  996.19470,  34.38879],
    [ 0, 0, 0, 1]
];
