"""Per-joint actuator torque of a trained policy over a rollout: the baseline for the S3 thresholds.

    uv run python scripts/torque_baseline.py Mjlab-Velocity-Flat-MicroDuck --checkpoint ~/stock_s0_model_2000.pt

Runs the play configuration (random velocity commands, and the play config's pushes: +-0.3 m/s every 0.5-1 s) with the BAM actuator model and records the
actuator output every control step. The PRD's S3 design checks compare MicroCluck against these numbers:
RMS per joint against the XL330's ~0.10 N.m continuous rating, peak against ~70% of the stall torque.
"""

import argparse
from dataclasses import asdict
from pathlib import Path

import torch

import mjlab_microduck.tasks  # noqa: F401  (registers the MicroDuck / MicroCluck tasks)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--num-envs", type=int, default=64)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dump-joints", help="also save the leg joint positions of every env and step to this .npz "
                                          "(used by scripts/microcluck_shell_clearance.py)")
    a = ap.parse_args()

    configure_torch_backends()
    env_cfg = load_env_cfg(a.task, play=True)
    env_cfg.scene.num_envs = a.num_envs
    agent_cfg = load_rl_cfg(a.task)
    env = ManagerBasedRlEnv(cfg=env_cfg, device=a.device)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = (load_runner_cls(a.task) or MjlabOnPolicyRunner)(env, asdict(agent_cfg), device=a.device)
    runner.load(str(Path(a.checkpoint).expanduser()), load_cfg={"actor": True}, strict=True, map_location=a.device)
    policy = runner.get_inference_policy(device=a.device)

    robot = env.unwrapped.scene["robot"]
    names = list(robot.actuator_names)
    steps = round(a.seconds / env.unwrapped.step_dt)
    obs = env.get_observations()
    torques, falls, poses = [], 0, []
    leg_cols = [i for i, n in enumerate(robot.joint_names) if not n.startswith("passive_") and "neck" not in n and "head" not in n]
    with torch.inference_mode():
        for _ in range(steps):
            obs, _, _, _ = env.step(policy(obs))
            torques.append(robot.data.actuator_force.clone())
            if a.dump_joints:
                poses.append(robot.data.joint_pos[:, leg_cols].clone())
            falls += int(env.unwrapped.reset_terminated.sum())
    if a.dump_joints:
        import numpy as np

        arr = torch.stack(poses).reshape(-1, len(leg_cols)).cpu().numpy()
        np.savez(a.dump_joints, joint_pos=arr, joint_names=np.array([robot.joint_names[i] for i in leg_cols]))
        print(f"saved {arr.shape[0]} leg poses ({[robot.joint_names[i] for i in leg_cols]}) to {a.dump_joints}")
    tau = torch.stack(torques).abs()  # (steps, envs, joints)

    print(f"task {a.task}, checkpoint {Path(a.checkpoint).name}: {a.num_envs} envs x {a.seconds:g} s, "
          f"{falls} falls ({falls / (a.num_envs * a.seconds / 60):.2f} per env-minute)")
    print(f"{'joint':18s} {'RMS':>7s} {'mean':>7s} {'p99':>7s} {'peak':>7s}   (N.m)")
    flat = tau.reshape(-1, tau.shape[-1])
    for i, name in enumerate(names):
        col = flat[:, i]
        rms = col.pow(2).mean().sqrt().item()
        print(f"{name:18s} {rms:7.3f} {col.mean().item():7.3f} {torch.quantile(col, 0.99).item():7.3f} {col.max().item():7.3f}")
    leg = [i for i, n in enumerate(names) if not any(k in n for k in ("neck", "head"))]
    print(f"leg joints: worst RMS {max(flat[:, i].pow(2).mean().sqrt().item() for i in leg):.3f} N.m "
          f"(XL330 continuous rating ~0.10), worst peak {max(flat[:, i].max().item() for i in leg):.3f} N.m")
    env.close()


if __name__ == "__main__":
    main()
