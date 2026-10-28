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
| `shell.scad` | The hollow body shell. It runs back from the screen bezel's back face to a rear wall at `shell_rear_x` (-48 mm by default, where the tail starts): front open (the bezel closes it), a tail opening in the rear wall, a USB-C slot, a power-switch hole and vents. `-D 'part="envelope"'` gives the solid outer shape, `part="cavity"` the inner space and `part="wings"` the two wing discs; `-D 'shell_style="box"'` builds the straight-sided shell. |
| `tools/build_shell.sh` | Exports `shell.stl`, `shell_envelope.stl`, `shell_cavity.stl` and (egg style) `wings.stl`; with a sim assets folder it also writes binary copies and a copy of `torso.scad`. Optional arguments after the assets folder: the style (`egg` or `box`), then any `name=value` override of a `torso.scad` parameter for that build. |
| `tools/shell_report.py` | Prints the shell's size, wall volume and an estimated mass (needs `trimesh`). |

The part layout in `torso.scad` must match `PARTS` in the sim repo's `build_walk_model.py`; a test there compares the two.
Every value is a placeholder until the parts are weighed and measured.

### Shell styles

`shell_style` in `torso.scad` picks the shape. All styles share the bezel end cap (66 mm wide, set by the bezel), so the
bezel interface never changes. The tail is a separate piece behind the rear wall, and the shell ends where it begins.

- **`egg` (the default), the target look:** a convex stack of slices that stays full behind the screen, rounds off toward
  the rear and lifts its underside there. The top is flush with the bezel and slopes down about 20 degrees, the rear edge is
  rounded (`egg_edge_r`), the underside rises 18 mm over the rear half (`egg_power_floor`), and the rear wall sits at
  `shell_rear_x = -48`, so the body is 70 mm deep. The placeholder tail block (x = -78 .. -48) touches the rear wall from
  behind. Two wing discs (`wings.stl`, 30 mm, 3.5 mm proud) lie on the sides at x = 4 mm, z = 72 mm, printed as a separate
  part in a second colour.
- **Earlier shapes**, kept for comparison. Each is built with the `OLD` overrides, which restore the first tail position
  and opening, plus its own:

```sh
OLD='tail_pos=[-45,0,60] tail_stub_size=[24.8,58.8] tail_stub_z=60 egg_power_floor=1 egg_wall_pad_rear=0.4'
# deep egg, 86 mm, encloses the tail block
hardware/tools/build_shell.sh out "" egg shell_rear_x=-64 shell_rear_z0=20 shell_rear_z1=102 shell_corner_rr=12 $OLD
# compact egg, 52 mm
hardware/tools/build_shell.sh out "" egg shell_rear_x=-30 shell_front_top=124 shell_rear_z0=27 shell_rear_z1=94 shell_corner_rr=16 egg_power_top=2.0 egg_edge_r=0 $OLD
# box, 52 mm, straight sides
hardware/tools/build_shell.sh out "" box shell_rear_x=-30 shell_front_top=124 shell_rear_z0=27 shell_rear_z1=94 shell_corner_rr=16 $OLD
```

The shell parameters were chosen with the sim's checks, `scripts/microcluck_shell_clearance.py`:

- **Fit:** the battery and electronics blocks (and the tail block, when a shell encloses it) must sit inside the exported
  cavity, `shell_cavity.stl` (the check excludes the open front). It also measures the thinnest wall, as the distance from
  the cavity to the outer shell.
- **Leg clearance:** signed clearance between the shell's convex envelope and every leg mesh hull, over MicroCluck's own
  gait poses (recorded with `scripts/torque_baseline.py --dump-joints`) plus random poses over 70% of each joint's range.
  The leg meshes are compared as convex hulls, which is conservative. The check skips leg parts that are far from the
  shell by a distance bound and still returns the exact minimum (it matched brute force to 0.000 um on 300 poses), which
  makes it about ten times faster. `--part k --of n` splits the recorded poses over several processes.
- **Wing discs:** each disc is a convex shape, and the check reports the closest leg part to it the same way.

Results for the four shells. Fit margins and wall are against the real cavity. The clearance rows use MicroCluck's own
gait (plain sole, `model_4999.pt`) and the same 6,000 stress poses for all four. The first third of the 96,000 recorded
poses (32,000) is the common set; the box, the deep egg and the target look also have the rest:

| | Box (52 mm) | Compact egg (52 mm) | Deep egg (86 mm) | Target look (70 mm, default) |
|---|---|---|---|---|
| Battery margin to the inner wall | +0.6 mm | +0.7 mm | +2.0 mm | +1.1 mm |
| Electronics margin | +0.5 mm | +2.1 mm | +6.5 mm | +4.0 mm |
| Tail block | outside | outside | inside, +0.8 mm | behind the rear wall |
| Thinnest wall (nominal 1.6 mm) | 1.60 mm | 1.51 mm | 1.51 mm | 1.53 mm |
| Wall volume in PETG, upper bound | 37 g | 38 g | 63 g | 52 g |
| Centre of the shell's mass, x | -3.7 mm | -4.7 mm | -20.8 mm | -12.8 mm |
| Wing discs, both, solid | none | 7.9 g | 7.9 g | 7.9 g |
| Closest leg, real gait, 32,000 poses | 7.5 mm | 6.4 mm | -0.8 mm, 1 pose intersects | 4.8 mm |
| Closest leg, real gait, all 96,000 poses | 7.5 mm | not run | -0.8 mm | 4.8 mm, none intersect |
| Closest leg, stress sweep, 6,000 poses | 1.1 mm, none intersect | -0.4 mm, 2 intersect | -12.8 mm, 130 intersect (2.2%) | -2.6 mm, 21 intersect (0.4%) |
| Closest wing disc to a leg | none | 26 mm | 26 mm | 26 mm |

(The compact egg's wall was 1.56 mm when first measured; rebuilding it with the current script gives 1.51 mm, because the
rear cavity slice moved slightly when the rounded rear edge was added. Its outer shape, and so its clearance, is unchanged.)

The target look keeps the real-gait clearance the deep egg lost: nothing in all 96,000 recorded poses intersects it, and the
closest approach is 4.8 mm (left thigh), against 7.5 mm for the box. The deep egg's trouble came from its low rear floor,
which the placeholder tail block forced; with the tail behind the wall the floor rises 18 mm toward the rear and the legs
swing under it. The stress sweep is not clean: 21 of 6,000 poses intersect by up to 2.6 mm at the right thigh. It is the
same legs-swung-back pose that comes closest for the box (1.1 mm), which the walking policy does not use but a scripted sit
or a fall recovery may. The egg shapes are wider through the middle than the box's straight taper, which costs a few
millimetres at the thighs.

**Tail placeholder.** The tail block moved from x = -45 mm to -63 mm in `torso.scad` and in the sim's `PARTS`, so the body
could be 70 mm deep with the tail outside it. Its mass, size and height are unchanged. The trunk's centre of mass moves from
x = 11.1 to 10.4 mm. The zero-action stand check shows no measurable change: in the play configuration (which has pushes) 57
of 64 environments fell with the old position and 54 and 50 in two runs with the new one; with the training randomization 8
fell with the old and 8 and 5 with the new. That is within the run-to-run spread. The trained policy used the old position.

Shell mass is the full wall volume times the PETG density, an upper bound that a thinner wall or a sparse print would
reduce. In the sim the shell is a translucent, visual-only mesh: it adds no mass and no contact, and the sim's frame mass
(the 60.9 g left over for the bezel and bumpers) was set when the shell was the box. The target look with its wing discs is
about 23 g heavier than the box (52 g and 7.9 g against 37 g), and the shell's centre of mass sits 9 mm further back. Neither
is in the sim yet, so the walking numbers so far do not include them.

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
