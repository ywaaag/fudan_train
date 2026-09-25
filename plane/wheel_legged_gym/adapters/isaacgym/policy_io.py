"""Shared evaluation services; never imports a command-line entry."""
from pathlib import Path
import numpy as np
import torch
from wheel_legged_gym.learning.modules.actor_critic_sequence import ActorCriticSequence

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

def set_fixed_command_ranges(env, commands: np.ndarray) -> None:
    command_tensor = torch.as_tensor(commands, device=env.device, dtype=torch.float)
    for key, column in (("lin_vel_x", 0), ("ang_vel_yaw", 1), ("height", 2)):
        env.command_ranges[key][:, 0] = command_tensor[:, column]
        env.command_ranges[key][:, 1] = command_tensor[:, column]

