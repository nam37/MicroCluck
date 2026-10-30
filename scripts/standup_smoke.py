"""Smoke test of the MicroCluck stand-up task: build it, reset, step with random actions, print what the policy sees."""

import sys

import torch

import mjlab_microduck.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

task = sys.argv[1] if len(sys.argv) > 1 else "Mjlab-StandUp-Prone-Flat-MicroCluck"
device = sys.argv[2] if len(sys.argv) > 2 else "cpu"
n_env = int(sys.argv[3]) if len(sys.argv) > 3 else 8

cfg = load_env_cfg(task)
cfg.scene.num_envs = n_env
env = ManagerBasedRlEnv(cfg=cfg, device=device)
obs, _ = env.reset()
for k, v in obs.items():
    print("obs group", k, tuple(v.shape))
print("action dim", env.action_manager.total_action_dim)
print("rewards:", sorted(env.reward_manager.active_terms))
print("events:", sorted(env.event_manager.active_terms) if hasattr(env.event_manager, "active_terms") else "?")
robot = env.scene["robot"]
print("joints:", robot.joint_names)
z = robot.data.root_link_pos_w[:, 2]
g = robot.data.projected_gravity_b
print("reset trunk z (mm):", (z * 1000).round().tolist())
print("reset gravity in body x:", g[:, 0].round(decimals=1).tolist(), "(+1 = face-down, 0 = upright)")
total = torch.zeros(n_env, device=device)
for i in range(150):
    act = 0.3 * torch.randn(n_env, env.action_manager.total_action_dim, device=device)
    obs, rew, term, trunc, info = env.step(act)
    total += rew
    if not torch.isfinite(rew).all():
        print("NON-FINITE reward at step", i)
        break
print("mean reward / step:", float(total.mean() / 150))
print("trunk z after 150 steps (mm):", (robot.data.root_link_pos_w[:, 2] * 1000).round().tolist())
env.close()
print("smoke OK")
