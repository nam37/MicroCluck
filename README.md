# MicroCluck

A ~25 cm bipedal walking robot chicken whose body is a touchscreen. It follows Pollen Robotics' Microduck, with a Freenove ESP32-S3 3.5" display as the torso, ten Dynamixel XL330 leg servos, and a walking policy trained in simulation and run on the ESP32-S3.

**Status:** planning, with the body design and the simulation under way. The product requirements document is the main record.

- [Product requirements](docs/PRD.html), draft v0.14. It is a single HTML file: download it and open it in a browser, because GitHub shows the source rather than the rendered page.
- [Earlier Markdown draft (v0.3)](docs/archive/PRD-v0.3.md), superseded and kept for history.
- [Hardware design files](hardware/README.md): the body shell, wings, tail and the tools that build them.

## Simulation code

The training and simulation code is not on `main`. It is Pollen Robotics' [microduck_rl](https://github.com/pollen-robotics/microduck_rl) with MicroCluck's robot model, tasks and scripts added, and it lives on two branches of this repository:

- `sim/microcluck`: the walking model and training task, as committed on 28 October 2026.
- `sim/standup`: the fall-recovery work will land here. It currently matches `sim/microcluck`.

The branches carry Pollen's history and its licences: the code is Apache-2.0 and the Microduck robot meshes are CC BY-NC-SA 4.0 (`LICENSE` and `LICENSE-MODELS.md` on those branches). To get only the simulation code, without the other branch, clone one branch (about 60 MB):

```
git clone --branch sim/microcluck --single-branch https://github.com/nam37/MicroCluck.git microcluck_sim
```

## Notes

MicroCluck is a personal project and is not affiliated with Pollen Robotics or Hugging Face. Pollen's Microduck STL meshes are licensed CC BY-NC-SA 4.0, so anything derived from them is non-commercial.
