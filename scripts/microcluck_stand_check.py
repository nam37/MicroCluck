"""Zero-action stand check in the real mjlab env (BAM actuators): does the home pose hold?

Pollen's rule for a new model: a rest pose must hold for about 3 s from noisy starts, judged by tilt
as well as height. With action 0 the policy output is the home pose, so this exercises the model, the
home trim and the actuator model together, without any learned policy.

    uv run python scripts/microcluck_stand_check.py Mjlab-Velocity-Flat-MicroCluck
    uv run python scripts/microcluck_stand_check.py Mjlab-Velocity-Flat-MicroDuck      # stock reference

By default the play config (no pushes) is used; --train-cfg uses the full training randomization.
"""

import argparse

import torch

import mjlab_microduck.tasks  # noqa: F401  (registers the MicroDuck / MicroCluck tasks)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task")
    ap.add_argument("--num-envs", type=int, default=64)
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--train-cfg", action="store_true", help="use the training config (pushes, DR) instead of play")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    cfg = load_env_cfg(a.task, play=not a.train_cfg)
    cfg.scene.num_envs = a.num_envs
    env = ManagerBasedRlEnv(cfg=cfg, device=a.device)
    obs, _ = env.reset()

    n_act = env.action_manager.total_action_dim
    print(f"task {a.task}: {env.num_envs} envs, step dt {env.step_dt:.3f}s, action dim {n_act}")
    for group, value in obs.items():
        print(f"  obs[{group!r}] shape {tuple(value.shape)}")

    robot = env.scene["robot"]
    steps = round(a.seconds / env.step_dt)
    fell = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    fall_step = torch.full((env.num_envs,), steps, dtype=torch.float32, device=env.device)
    zero = torch.zeros(env.num_envs, n_act, device=env.device)
    for step in range(steps):
        _, _, terminated, _, _ = env.step(zero)
        newly = terminated & ~fell
        fall_step[newly] = step + 1
        fell |= terminated

    gz = robot.data.projected_gravity_b[:, 2]  # -1 when upright
    tilt = torch.rad2deg(torch.acos((-gz).clamp(-1.0, 1.0)))
    z = robot.data.root_link_pos_w[:, 2]
    print(f"after {a.seconds:g}s with zero action:")
    print(f"  terminated (fell over) in {int(fell.sum())}/{env.num_envs} envs")
    print(f"  mean time to fall (capped at the test length): {fall_step.mean() * env.step_dt:.2f} s")
    print(f"  tilt now: mean {tilt.mean():.1f} deg, max {tilt.max():.1f} deg")
    print(f"  trunk height: mean {z.mean():.4f} m, min {z.min():.4f} m")
    env.close()


if __name__ == "__main__":
    main()
