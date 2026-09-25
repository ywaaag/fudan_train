"""Open-loop wheel DOF test, independent of the trained policy."""

import isaacgym  # noqa: F401
import torch

from wheel_legged_gym.app.bootstrap import create_task_registry
from wheel_legged_gym.app.arguments import get_args


def main():
    task_registry = create_task_registry()
    args = get_args()
    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
    env_cfg.asset.disable_gravity = True
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_restitution = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_inertia = False
    env_cfg.domain_rand.randomize_base_com = False
    env_cfg.domain_rand.randomize_Kp = False
    env_cfg.domain_rand.randomize_Kd = False
    env_cfg.domain_rand.randomize_motor_torque = False

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    action = torch.zeros((env.num_envs, env.num_actions), device=env.device)
    action[:, 2] = 1.0
    action[:, 5] = 1.0

    print("dof_names:", env.dof_names)
    print("torque_limits:", env.torque_limits.detach().cpu().tolist())
    print("d_gains:", env.d_gains[0].detach().cpu().tolist())
    print("initial_dof_vel:", env.dof_vel[0].detach().cpu().tolist())
    for step in range(200):
        env.step(action)
        if step % 25 == 0:
            print(
                f"step={step:03d} wheel_vel="
                f"({env.dof_vel[0, 2].item():.5f},{env.dof_vel[0, 5].item():.5f}) "
                f"wheel_tau=({env.torques[0, 2].item():.5f},{env.torques[0, 5].item():.5f})"
            )
    env.gym.destroy_sim(env.sim)


if __name__ == "__main__":
    main()
