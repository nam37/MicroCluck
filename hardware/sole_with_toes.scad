// Printable sole: Pollen's sole STL with the three toes fused onto its front wall.
// Pollen's STL is CC BY-NC-SA 4.0 and is NOT copied into this repo: point pollen_dir at a local copy
//   openscad -D 'side="left"' -D 'pollen_dir="/path/to/microduck/assets"' -o build/sole_with_toes_left.stl sole_with_toes.scad
// The result is a derivative of their sole, so it carries the same license if you share it.
include <toes_lib.scad>
include <pollen_frame.scad>

pollen_dir = "";

module pollen_sole() {
    multmatrix(side == "left" ? M_pollen_left : M_pollen_right)
        import(str(pollen_dir, "/sole_", side, ".stl"));
}

union() {
    pollen_sole();
    oriented() toes3d();
}
