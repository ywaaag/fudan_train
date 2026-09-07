"""Evaluate a PyTorch checkpoint on a fixed seven-command Isaac Gym grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import isaacgym  # must precede torch
import numpy as np
import torch

from wheel_legged_gym.envs import *  # noqa: F401,F403
from wheel_legged_gym.rsl_rl.modules.actor_critic_sequence import ActorCriticSequence
from wheel_legged_gym.scripts.isaac_parity_trace import _disable_randomization, _gym_args
from wheel_legged_gym.utils import task_registry


COMMAND_GRID = np.array(
    [
        [0.00, 0.00, 0.40],
        [0.05, 0.00, 0.40],
        [-0.05, 0.00, 0.40],
        [0.10, 0.00, 0.40],
        [-0.10, 0.00, 0.40],
        [0.00, 0.05, 0.40],
        [0.00, -0.05, 0.40],
    ],
    dtype=np.float64,
)
WHEEL_RADIUS_M = 0.06


def load_policy(checkpoint: Path, device: str) -> ActorCriticSequence:
    model = ActorCriticSequence(
        num_obs=25,
        num_critic_obs=1,
        num_actions=6,
        num_encoder_obs=125,
        latent_dim=3,
        encoder_hidden_dims=[128, 64],
        actor_hidden_dims=[128, 64, 32],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    payload = torch.load(checkpoint, map_location="cpu")
    state = {
        name: value
        for name, value in payload["model_state_dict"].items()
        if not name.startswith("critic.")
    }
    missing, unexpected = model.load_state_dict(state, strict=False)
    actor_missing = [name for name in missing if not name.startswith("critic.")]
    if actor_missing or unexpected:
        raise ValueError(
            f"checkpoint actor/encoder contract mismatch: missing={actor_missing}, unexpected={unexpected}"
        )
    return model.to(device).eval()


def _set_fixed_ranges(env, commands: np.ndarray) -> None:
    command_tensor = torch.as_tensor(commands, device=env.device, dtype=torch.float)
    for key, column in (("lin_vel_x", 0), ("ang_vel_yaw", 1), ("height", 2)):
        env.command_ranges[key][:, 0] = command_tensor[:, column]
        env.command_ranges[key][:, 1] = command_tensor[:, column]


def _command_pass(command: np.ndarray, mean_vx: float, mean_yaw: float) -> bool:
    if command[0] == 0.0 and command[1] == 0.0:
        return bool(abs(mean_vx) < 0.03 and abs(mean_yaw) < 0.05)
    if command[0] != 0.0:
        return bool(abs(mean_vx - command[0]) < 0.05)
    return bool(abs(mean_yaw - command[1]) < 0.05)


def run(
    checkpoint: Path,
    *,
    seconds: float,
    warmup_seconds: float,
    seed: int,
    device_id: int,
) -> dict:
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if seconds <= 0.0 or not 0.0 <= warmup_seconds < seconds:
        raise ValueError("require seconds > warmup_seconds >= 0")

    env_cfg, _ = task_registry.get_cfgs(name="wheel_legged")
    env_cfg.seed = seed
    env_cfg.env.num_envs = len(COMMAND_GRID)
    env_cfg.env.episode_length_s = max(30.0, seconds + 5.0)
    env_cfg.terrain.mesh_type = "plane"
    env_cfg.terrain.curriculum = False
    env_cfg.commands.curriculum = False
    env_cfg.commands.sampling_strategy = "uniform"
    _disable_randomization(env_cfg)
    args = _gym_args()
    args.num_envs = len(COMMAND_GRID)
    args.seed = seed
    args.compute_device_id = device_id
    args.sim_device_id = device_id
    args.sim_device = f"cuda:{device_id}"
    args.rl_device = f"cuda:{device_id}"
    env, _ = task_registry.make_env(name="wheel_legged", args=args, env_cfg=env_cfg)
    _set_fixed_ranges(env, COMMAND_GRID)
    env.reset()
    obs, history = env.get_observations()
    policy = load_policy(checkpoint, env.device)

    total_steps = int(round(seconds / env.dt))
    warmup_steps = int(round(warmup_seconds / env.dt))
    samples = total_steps - warmup_steps
    sums = {
        "vx": torch.zeros(env.num_envs, device=env.device),
        "yaw": torch.zeros(env.num_envs, device=env.device),
        "height": torch.zeros(env.num_envs, device=env.device),
        "wheel": torch.zeros(env.num_envs, 2, device=env.device),
        "contact": torch.zeros(env.num_envs, 2, device=env.device),
        "rolling_residual": torch.zeros(env.num_envs, 2, device=env.device),
        "action": torch.zeros(env.num_envs, 6, device=env.device),
        "latent_velocity": torch.zeros(env.num_envs, 3, device=env.device),
    }
    terminations = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    last_actions = torch.zeros(env.num_envs, 6, device=env.device)
    for step in range(total_steps):
        with torch.inference_mode():
            last_actions, latent = policy.act_inference(obs, history)
            last_actions = torch.clamp(last_actions, -100.0, 100.0)
        obs, _, _, dones, _, history = env.step(last_actions)
        terminations += dones.long()
        if step < warmup_steps:
            continue
        wheel = env.dof_vel[:, [2, 5]]
        vx = env.base_lin_vel[:, 0]
        sums["vx"] += vx
        sums["yaw"] += env.base_ang_vel[:, 2]
        sums["height"] += env.base_height
        sums["wheel"] += wheel
        sums["contact"] += (env.contact_forces[:, env.feet_indices[:2], 2] > 1.0).float()
        sums["rolling_residual"] += vx.unsqueeze(1) + WHEEL_RADIUS_M * wheel
        sums["action"] += last_actions
        sums["latent_velocity"] += latent[:, :3] / env.obs_scales.lin_vel

    results = []
    for index, command in enumerate(COMMAND_GRID):
        mean_vx = float((sums["vx"][index] / samples).item())
        mean_yaw = float((sums["yaw"][index] / samples).item())
        termination_count = int(terminations[index].item())
        command_ok = _command_pass(command, mean_vx, mean_yaw)
        result = {
            "command": command.tolist(),
            "passed": bool(command_ok and termination_count == 0),
            "command_tracking_passed": command_ok,
            "mean_vx_m_s": mean_vx,
            "final_vx_m_s": float(env.base_lin_vel[index, 0].item()),
            "mean_yaw_rate_rad_s": mean_yaw,
            "final_yaw_rate_rad_s": float(env.base_ang_vel[index, 2].item()),
            "mean_base_height_m": float((sums["height"][index] / samples).item()),
            "final_base_height_m": float(env.base_height[index].item()),
            "mean_wheel_velocity_rad_s": (sums["wheel"][index] / samples).cpu().tolist(),
            "final_wheel_velocity_rad_s": env.dof_vel[index, [2, 5]].cpu().tolist(),
            "wheel_contact_fraction": (sums["contact"][index] / samples).cpu().tolist(),
            "mean_rolling_residual_m_s": (
                sums["rolling_residual"][index] / samples
            ).cpu().tolist(),
            "mean_action": (sums["action"][index] / samples).cpu().tolist(),
            "mean_latent_velocity_estimate_m_s": (
                sums["latent_velocity"][index] / samples
            ).cpu().tolist(),
            "termination_count": termination_count,
        }
        results.append(result)

    payload = {
        "checkpoint": str(checkpoint.resolve()),
        "engine": "Isaac Gym PhysX",
        "seconds": seconds,
        "warmup_seconds": warmup_seconds,
        "policy_steps": total_steps,
        "physics_steps": total_steps * int(env.cfg.control.decimation),
        "policy_rate_hz": int(round(1.0 / env.dt)),
        "physics_rate_hz": int(round(1.0 / env.sim_params.dt)),
        "seed": seed,
        "noise_enabled": False,
        "domain_randomization_enabled": False,
        "passed": all(item["passed"] for item in results),
        "results": results,
    }
    print("RESULT " + json.dumps(payload, sort_keys=True))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--warmup-seconds", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    payload = run(
        args.checkpoint,
        seconds=args.seconds,
        warmup_seconds=args.warmup_seconds,
        seed=args.seed,
        device_id=args.device_id,
    )
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
