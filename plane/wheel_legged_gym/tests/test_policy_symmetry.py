from __future__ import annotations

import torch

from wheel_legged_gym.rsl_rl.modules.policy_symmetry import (
    mirror_actions,
    mirror_history,
    mirror_observations,
)


def test_mirror_transforms_are_involutions():
    obs = torch.arange(25, dtype=torch.float32).reshape(1, 25)
    actions = torch.arange(6, dtype=torch.float32).reshape(1, 6)
    history = torch.cat([obs + 100.0 * index for index in range(5)], dim=-1)

    torch.testing.assert_close(mirror_observations(mirror_observations(obs)), obs)
    torch.testing.assert_close(mirror_actions(mirror_actions(actions)), actions)
    torch.testing.assert_close(mirror_history(mirror_history(history)), history)
    torch.testing.assert_close(mirror_history(history)[:, :25], mirror_observations(obs))


def test_mirror_swaps_leg_and_wheel_channels_without_extra_joint_signs():
    obs = torch.zeros((1, 25), dtype=torch.float32)
    obs[:, 9:13] = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    obs[:, 13:19] = torch.tensor([[10.0, 11.0, 12.0, 13.0, 14.0, 15.0]])
    obs[:, 19:25] = torch.tensor([[20.0, 21.0, 22.0, 23.0, 24.0, 25.0]])

    mirrored = mirror_observations(obs)
    torch.testing.assert_close(
        mirrored[:, 9:13], torch.tensor([[3.0, 4.0, 1.0, 2.0]])
    )
    torch.testing.assert_close(
        mirrored[:, 13:19],
        torch.tensor([[13.0, 14.0, 15.0, 10.0, 11.0, 12.0]]),
    )
    torch.testing.assert_close(
        mirrored[:, 19:25],
        torch.tensor([[23.0, 24.0, 25.0, 20.0, 21.0, 22.0]]),
    )
