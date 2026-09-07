"""Fixed-step Isaac Gym source trace for the Fudan policy contract.

Run this with the ``fudan_leg`` environment.  It preserves the training
physics timestep/decimation and disables stochastic evaluation features.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import isaacgym  # must precede torch
from isaacgym import gymapi
import numpy as np
import onnxruntime as ort
import torch

from wheel_legged_gym.envs import *  # noqa: F401,F403
from wheel_legged_gym.utils import task_registry


PLANE_ROOT = Path(__file__).resolve().parents[2]
POLICY = PLANE_ROOT / "outputs/wheel_policy_2900.onnx"


def _gym_args() -> SimpleNamespace:
    return SimpleNamespace(
        task="wheel_legged",
        sim_device="cuda:0",
        sim_device_type="cuda",
        sim_device_id=0,
        compute_device_id=0,
        graphics_device_id=-1,
        physics_engine=gymapi.SIM_PHYSX,
        use_gpu=True,
        use_gpu_pipeline=True,
        pipeline="gpu",
        subscenes=0,
        num_threads=0,
        headless=True,
        rl_device="cuda:0",
        num_envs=1,
        seed=1,
        max_iterations=None,
        resume=False,
        experiment_name=None,
        run_name=None,
        load_run=None,
        checkpoint=None,
        horovod=False,
        exptid="",
    )


def _disable_randomization(cfg) -> None:
    cfg.noise.add_noise = False
    names = (
        "randomize_friction",
        "randomize_restitution",
        "randomize_base_mass",
        "randomize_inertia",
        "randomize_base_com",
        "randomize_Kp",
        "randomize_Kd",
        "randomize_motor_torque",
        "randomize_default_dof_pos",
        "randomize_action_delay",
        "push_robots",
        "lift_robots",
        "downward_impulse_robots",
        "vmc_force_events",
    )
    for name in names:
        if hasattr(cfg.domain_rand, name):
            setattr(cfg.domain_rand, name, False)


def run(policy_path: Path, policy_steps: int, command: np.ndarray, log_every: int) -> dict:
    if policy_steps < 1:
        raise ValueError("policy_steps must be positive")
    env_cfg, _ = task_registry.get_cfgs(name="wheel_legged")
    env_cfg.env.num_envs = 1
    env_cfg.env.episode_length_s = max(30.0, policy_steps * 0.02)
    env_cfg.terrain.mesh_type = "plane"
    env_cfg.terrain.curriculum = False
    env_cfg.commands.curriculum = False
    env_cfg.commands.ranges.lin_vel_x = [float(command[0]), float(command[0])]
    env_cfg.commands.ranges.ang_vel_yaw = [float(command[1]), float(command[1])]
    env_cfg.commands.ranges.height = [float(command[2]), float(command[2])]
    _disable_randomization(env_cfg)

    env, _ = task_registry.make_env(name="wheel_legged", args=_gym_args(), env_cfg=env_cfg)
    env.reset()
    obs, history = env.get_observations()
    session = ort.InferenceSession(str(policy_path), providers=["CPUExecutionProvider"])
    input_shapes = {item.name: item.shape for item in session.get_inputs()}
    if input_shapes.get("obs")[-1] != 25 or input_shapes.get("obs_history")[-1] != 125:
        raise ValueError(f"unexpected ONNX input contract: {input_shapes}")

    initial_position = env.root_states[0, :3].detach().cpu().numpy().copy()
    action = np.zeros(6, dtype=np.float32)
    min_height = float("inf")
    max_height = float("-inf")
    min_wheel_contacts = [1, 1]
    max_base_contacts = 0
    latest_torque = np.zeros(6, dtype=np.float32)
    for step in range(policy_steps):
        obs_np = obs[0].detach().cpu().numpy().astype(np.float32, copy=False)
        history_np = history[0].detach().cpu().numpy().astype(np.float32, copy=False)
        action = session.run(
            ["actions"],
            {"obs": obs_np[None, :], "obs_history": history_np[None, :]},
        )[0][0]
        action = np.clip(action, -100.0, 100.0).astype(np.float32)
        actions = torch.as_tensor(action, device=env.device).unsqueeze(0)
        obs, _, _, _, _, history = env.step(actions)
        height = float(env.root_states[0, 2].item())
        min_height = min(min_height, height)
        max_height = max(max_height, height)
        contacts = (env.contact_forces[0, env.feet_indices, 2] > 1.0).detach().cpu().numpy().astype(int)
        contacts = contacts[:2].tolist()
        min_wheel_contacts = [min(a, b) for a, b in zip(min_wheel_contacts, contacts)]
        base_body = int(env.base_index[0]) if hasattr(env, "base_index") else 0
        base_contacts = int((torch.linalg.norm(env.contact_forces[0, base_body]) > 1.0).item())
        max_base_contacts = max(max_base_contacts, base_contacts)
        latest_torque = env.torques[0].detach().cpu().numpy().copy()
        if step % log_every == 0 or step == policy_steps - 1:
            wheel_speed = env.dof_vel[0, [2, 5]].detach().cpu().numpy()
            print(
                f"policy_step={step} sim_s={(step + 1) * env.dt:.3f} z={height:.4f} "
                f"wheel_contacts={tuple(contacts)} base_contacts={base_contacts} "
                f"wheel_dq={np.round(wheel_speed, 4).tolist()} action={np.round(action, 4).tolist()}"
            )

    position = env.root_states[0, :3].detach().cpu().numpy()
    final_obs = obs[0].detach().cpu().numpy()
    result = {
        "engine": "Isaac Gym PhysX",
        "completed_policy_steps": policy_steps,
        "completed_physics_steps": policy_steps * int(env.cfg.control.decimation),
        "simulated_seconds": policy_steps * float(env.dt),
        "physics_dt_s": float(env.sim_params.dt),
        "physics_rate_hz": int(round(1.0 / float(env.sim_params.dt))),
        "policy_dt_s": float(env.dt),
        "policy_rate_hz": int(round(1.0 / float(env.dt))),
        "decimation": int(env.cfg.control.decimation),
        "observation_dim": int(final_obs.size),
        "history_dim": int(history[0].numel()),
        "final_observation": final_obs.tolist(),
        "policy_action": action.tolist(),
        "base_height_m": float(position[2]),
        "min_base_height_m": min_height,
        "max_base_height_m": max_height,
        "base_position_world_m": position.tolist(),
        "base_displacement_world_m": (position - initial_position).tolist(),
        "base_quaternion_xyzw": env.root_states[0, 3:7].detach().cpu().numpy().tolist(),
        "body_linear_velocity_m_s": env.base_lin_vel[0].detach().cpu().numpy().tolist(),
        "body_angular_velocity_rad_s": env.base_ang_vel[0].detach().cpu().numpy().tolist(),
        "wheel_contacts": (env.contact_forces[0, env.feet_indices[:2], 2] > 1.0).detach().cpu().numpy().astype(int).tolist(),
        "min_wheel_contacts": min_wheel_contacts,
        "max_base_contacts": max_base_contacts,
        "dof_names": list(env.dof_names),
        "dof_position": env.dof_pos[0].detach().cpu().numpy().tolist(),
        "dof_velocity": env.dof_vel[0].detach().cpu().numpy().tolist(),
        "joint_torque_nm": latest_torque.tolist(),
        "command": np.asarray(command, dtype=np.float64).tolist(),
        "noise_enabled": False,
        "domain_randomization_enabled": False,
    }
    print("RESULT " + json.dumps(result, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=POLICY)
    parser.add_argument("--policy-steps", type=int, default=2000)
    parser.add_argument("--forward", type=float, default=0.0)
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument("--height", type=float, default=0.40)
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()
    run(
        args.policy,
        args.policy_steps,
        np.array([args.forward, args.yaw, args.height], dtype=np.float64),
        args.log_every,
    )


if __name__ == "__main__":
    main()
