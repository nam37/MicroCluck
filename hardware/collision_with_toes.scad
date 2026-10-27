// Simulation collision shape: the convex hull of Pollen's sole plus the toes (support on a flat floor is exactly
// the convex hull of the contact points; MuJoCo collides meshes as convex hulls anyway).
//   openscad -D 'side="left"' -D 'pollen_dir="/path/to/microduck/assets"' -o build/collision_with_toes_left.stl collision_with_toes.scad
include <toes_lib.scad>
include <pollen_frame.scad>

pollen_dir = "";

module pollen_sole() {
    multmatrix(side == "left" ? M_pollen_left : M_pollen_right)
        import(str(pollen_dir, "/sole_", side, ".stl"));
}

hull() {
    pollen_sole();
    oriented() toes3d();
}
