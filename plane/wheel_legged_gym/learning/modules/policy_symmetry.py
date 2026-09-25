"""Left-right mirror transforms for the frozen Fudan 25D policy contract."""

from __future__ import annotations

import torch


OBS_DIM = 25
HISTORY_FRAMES = 5
ACTION_DIM = 6


def mirror_actions(actions: torch.Tensor) -> torch.Tensor:
    """Swap canonical left/right action channels without forcing equality."""
    if actions.shape[-1] != ACTION_DIM:
        raise ValueError(f"expected {ACTION_DIM} actions, got {actions.shape}")
    return actions[..., (3, 4, 5, 0, 1, 2)]


def mirror_observations(observations: torch.Tensor) -> torch.Tensor:
    """Reflect canonical observations through the robot sagittal plane."""
    if observations.shape[-1] != OBS_DIM:
        raise ValueError(f"expected {OBS_DIM} observations, got {observations.shape}")

    mirrored = observations.clone()
    # Angular velocity is an axial vector under y -> -y reflection.
    mirrored[..., 0:3] = observations[..., 0:3] * observations.new_tensor((-1.0, 1.0, -1.0))
    # Projected gravity is a polar vector.
    mirrored[..., 3:6] = observations[..., 3:6] * observations.new_tensor((1.0, -1.0, 1.0))
    # [forward velocity, yaw rate, height].
    mirrored[..., 6:9] = observations[..., 6:9] * observations.new_tensor((1.0, -1.0, 1.0))
    # The parity URDF has matching left/right hinge axes. Equal leg angles are
    # geometrically mirrored, so only swap channels; do not add a sign flip.
    mirrored[..., 9:13] = observations[..., (11, 12, 9, 10)]
    mirrored[..., 13:19] = observations[..., (16, 17, 18, 13, 14, 15)]
    mirrored[..., 19:25] = mirror_actions(observations[..., 19:25])
    return mirrored


def mirror_history(history: torch.Tensor) -> torch.Tensor:
    """Mirror five observation frames while preserving temporal order."""
    expected = OBS_DIM * HISTORY_FRAMES
    if history.shape[-1] != expected:
        raise ValueError(f"expected history width {expected}, got {history.shape}")
    frames = history.reshape(*history.shape[:-1], HISTORY_FRAMES, OBS_DIM)
    return mirror_observations(frames).reshape_as(history)


def mirror_linear_velocity(linear_velocity: torch.Tensor) -> torch.Tensor:
    """Mirror the encoder's supervised body-linear-velocity target."""
    if linear_velocity.shape[-1] != 3:
        raise ValueError(f"expected 3D linear velocity, got {linear_velocity.shape}")
    return linear_velocity * linear_velocity.new_tensor((1.0, -1.0, 1.0))
