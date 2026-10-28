# hardware/

Parametric parts, written in OpenSCAD. One parameter table per part drives the printed part and the simulation.

## The foot: Pollen's foot plus three chunky toes

The foot is Pollen's: same foot, same ankle interface. Three chunky, rounded toes are fused onto the outer face of the
sole's front wall (a 2 mm shell, measured from their STL). They run down to the floor, so they are part of the contact
patch. In the simulation the collision shape is the convex hull of Pollen's sole plus the toes, under the same name
Pollen's code looks up (`left_foot_collision`, `right_foot_collision`).

| File | What it is |
|---|---|
| `params.scad` | **The foot parameter table.** Toe reach, base and tip size, splay, heights, density. |
| `toes_lib.scad` | Shared toe geometry. The toes are clipped so they never enter the cavity around Pollen's foot. |
| `toes.scad` | The three toes alone, in the stand-pose foot frame. The simulation draws this. |
| `collision_with_toes.scad` | Convex hull of Pollen's sole plus the toes. The simulation collides with this. |
| `sole_with_toes.scad` | The printable part: Pollen's sole STL with the toes fused on. |
| `pollen_frame.scad` | The transform from Pollen's STL frame into the stand-pose foot frame. |
| `tools/build_toes.sh` | Exports the foot STLs with OpenSCAD (about 45 s). |
| `tools/foot_report.py` | Prints toe extents, volume and an estimated mass (needs `trimesh`). |

The two foot parts that import Pollen's sole need a local copy of their assets, which are not copied into this repo
(CC BY-NC-SA 4.0). A part derived from their sole carries the same license if you share it.

Foot frame: x forward, y to the robot's left, z up, origin on the ground directly under the ankle axis, axes aligned
with the world while the robot stands in its home pose.

What the toes change: Pollen's sole only touches the floor from about 15 mm behind to 29 mm ahead of the ankle axis (its
front rises like a rocker). The toes extend the contact patch forward to 46 mm, so the forward margin of the center of
mass grows from about 28 mm to 45 mm. The backward margin does not change (about 16 mm). Toes on the floor can drag during
the swing and clip the other foot on narrow steps; shorten the inner toe (`inner_toe_trim`) or the reach if they do.

## The torso: layout and body shell

| File | What it is |
|---|---|
| `torso.scad` | **The torso table.** Positions and sizes of the bezel, board, battery, electronics, comb and tail in the sim's trunk frame, plus the shell parameters. Geometry only: masses live in the sim. |
| `shell.scad` | The hollow body shell that joins the screen bezel's back face to the tail's front face: front open (the bezel closes it), rear wall with a tail socket, a USB-C slot, a power-switch hole and vents. Pass `-D 'envelope=true'` for the solid outer shape. |
| `tools/build_shell.sh` | Exports `shell.stl` and `shell_envelope.stl`; with a sim assets folder it also writes binary copies and a copy of `torso.scad`. |
| `tools/shell_report.py` | Prints the shell's size, wall volume and an estimated mass (needs `trimesh`). |

The part layout in `torso.scad` must match `PARTS` in the sim repo's `build_walk_model.py`; a test there compares the two.
Every value is a placeholder until the parts are weighed and measured.

The shell parameters were chosen with the sim's checks, `scripts/microcluck_shell_clearance.py`:

- **Fit:** the battery and electronics blocks must sit inside the cavity (the check excludes the open front).
- **Leg clearance:** signed clearance between the shell's convex envelope and every leg mesh hull, over the stock walker's
  real gait poses (recorded with `scripts/torque_baseline.py --dump-joints`) plus random poses over 70% of each joint's range.

Current result with the defaults: fit margins +0.7 mm (battery) and +0.8 mm (electronics); no intersections in 4,000 sampled
poses; in real gait 99% of poses clear by more than 26 mm (5.5 mm in the stress poses); the closest approach is 1.7 mm over
real gait poses and 1.1 mm over the stress poses, both against the right thigh and the shell's lower rear corner. The leg meshes are compared as convex hulls, which
is conservative. Rerun the check with MicroCluck's own gait once a policy walks.

Shell mass: about 37 g at full wall volume in PETG (1.6 mm walls), an upper bound. In the sim the shell is a translucent,
visual-only mesh: it adds no mass and no contact, so the physics is unchanged. The placeholder frame mass already in
the budget stands in for it.

## Build

OpenSCAD 2021.01 is installed in WSL under `~/opt/openscad` with a wrapper at `~/.local/bin/openscad`.

```sh
# foot: STLs into hardware/build/, plus binary copies of the two simulation meshes
hardware/tools/build_toes.sh hardware/build <sim>/src/mjlab_microduck/robot/microcluck/assets
# torso: shell STLs, binary copies, and torso.scad for the sim's consistency test
hardware/tools/build_shell.sh hardware/build <sim>/src/mjlab_microduck/robot/microcluck/assets
# rebuild the simulation models (plain sole = default, and the toes variant)
uv run python src/mjlab_microduck/robot/microcluck/build_walk_model.py
# train with the toes (the default is the plain sole)
MICROCLUCK_FOOT=toes uv run train Mjlab-Velocity-Flat-MicroCluck ...
```

Run the clearance check single-threaded (`OMP_NUM_THREADS=1`) and at low priority while training: it is many small matrix
products, and a multithreaded BLAS makes it thrash.

## Not done

- Test prints. The ankle interface of the foot is Pollen's, so only the toes and the shell are new.
- Attachments: how the shell fastens to the bezel and the tail, and the battery door. The shell has the openings but not
  the screw bosses.
- Frame, bezel, comb and tail follow the same pattern: a parameter table here, STLs out, numbers into the sim model.
