// The three toes alone, in the stand-pose foot frame. The simulation draws this as the toe visual.
//   openscad -D 'side="left"' -o build/toes_left.stl toes.scad
include <toes_lib.scad>

oriented() toes3d();
