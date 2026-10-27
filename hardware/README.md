# hardware/

Parametric parts, written in OpenSCAD. One parameter table drives the printed part and the simulation.

## The foot: Pollen's foot plus three chunky toes

The foot is Pollen's: same foot, same ankle interface. Three chunky, rounded toes are fused onto the outer face of the
sole's front wall (a 2 mm shell, measured from their STL). They run down to the floor, so they are part of the contact
patch. In the simulation the collision shape is the convex hull of Pollen's sole plus the toes, under the same name
Pollen's code looks up (`left_foot_collision`, `right_foot_collision`).

| File | What it is |
|---|---|
| `params.scad` | **The parameter table.** Toe reach, base and tip size, splay, heights, density. |
| `toes_lib.scad` | Shared toe geometry. The toes are clipped so they never enter the cavity around Pollen's foot. |
| `toes.scad` | The three toes alone, in the stand-pose foot frame. The simulation draws this. |
| `collision_with_toes.scad` | Convex hull of Pollen's sole plus the toes. The simulation collides with this. |
| `sole_with_toes.scad` | The printable part: Pollen's sole STL with the toes fused on. |
| `pollen_frame.scad` | The transform from Pollen's STL frame into the stand-pose foot frame. |
| `tools/build_toes.sh` | Exports the STLs with OpenSCAD (about 45 s). |
| `tools/foot_report.py` | Prints toe extents, volume and an estimated mass (needs `trimesh`). |

The two parts that import Pollen's sole need a local copy of their assets, which are not copied into this repo
(CC BY-NC-SA 4.0). A part derived from their sole carries the same license if you share it.

Frame: x forward, y to the robot's left, z up, origin on the ground directly under the ankle axis, axes aligned with
the world while the robot stands in its home pose.

## Build

OpenSCAD 2021.01 is installed in WSL under `~/opt/openscad` with a wrapper at `~/.local/bin/openscad`.

```sh
# STLs into hardware/build/, plus binary copies of the two simulation meshes
hardware/tools/build_toes.sh hardware/build <sim>/src/mjlab_microduck/robot/microcluck/assets
# rebuild the simulation models (plain sole = default, and the toes variant)
uv run python src/mjlab_microduck/robot/microcluck/build_walk_model.py
# train with the toes (the default is the plain sole)
MICROCLUCK_FOOT=toes uv run train Mjlab-Velocity-Flat-MicroCluck ...
```

With the defaults the toes reach 12 mm past the front wall (tips 46 mm ahead of the ankle axis), weigh about 1.2 g per
foot at 30% infill in TPU, and the fused sole is one watertight, manifold mesh (9.2 cm3).

## What the toes change

- Pollen's sole only touches the floor from about 15 mm behind to 29 mm ahead of the ankle axis (its front rises like a
  rocker). The toes extend the contact patch forward to 46 mm. The center of mass stays over the ankle axis, so the
  forward margin grows from about 28 mm to 45 mm. The backward margin does not change (about 16 mm).
- In an open-loop stand test the toes foot stays up much longer. That is not a walking test.
- Toes on the floor can drag during the swing and clip the other foot on narrow steps. Shorten the inner toe
  (`inner_toe_trim`) or the reach if they do. Real TPU toes flex; the simulation's are rigid.
- Not done: test prints, and the toes' mass in the ankle body (the sim still carries Pollen's foot mass).
- Frame, bezel, comb and tail follow the same pattern: a parameter table here, STLs out, numbers into the sim model.
