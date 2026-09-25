"""Resample explicit command buffers, preserving RNG and cohort-reset order."""
from dataclasses import dataclass
import torch
from wheel_legged_gym.domain.commands.command_sampling import (
    sample_method_v1, sample_zero_reverse_mixture,
)


@dataclass(frozen=True)
class CommandBuffers:
    commands: torch.Tensor
    sample_mode: torch.Tensor
    segment_counter: torch.Tensor


def resample(env_ids, *, state: CommandBuffers, ranges, config, device,
             reset_start_stop, sample_heading):
    """Mutate selected rows only; callbacks retain simulator-specific boundary behavior.

    The height draw runs even for height_bank, before its selected height override.
    Reset callback stays lazy and follows that draw; heading is always drawn last.
    """
    strategy = getattr(config, "sampling_strategy", "uniform")
    if strategy == "height_bank":
        from wheel_legged_gym.domain.commands.command_sampling import sample_height_bank
        height_selected = sample_height_bank(ranges['lin_vel_x'][env_ids],
            ranges['ang_vel_yaw'][env_ids], ranges['height'][env_ids],
            state.segment_counter[env_ids], config.height_bank)
        state.commands[env_ids, :2] = height_selected[:, :2]
        state.sample_mode[env_ids] = -1
        state.segment_counter[env_ids] += 1
    elif strategy == "fixed_bank":
        from wheel_legged_gym.domain.commands.command_sampling import sample_fixed_bank
        selected = sample_fixed_bank(ranges['lin_vel_x'][env_ids],
            ranges['ang_vel_yaw'][env_ids],
            state.segment_counter[env_ids], config.fixed_bank)
        state.commands[env_ids, :2] = selected
        state.sample_mode[env_ids] = -1
        state.segment_counter[env_ids] += 1
    elif strategy == "method_v1":
        linear, yaw, mode = sample_method_v1(
            ranges["lin_vel_x"][env_ids],
            ranges["ang_vel_yaw"][env_ids],
            phase=getattr(config, "training_phase", "stand"),
            small_linear_limit=float(
                getattr(config, "mixture_small_linear_limit", 0.10)
            ),
            small_yaw_limit=float(
                getattr(config, "mixture_small_yaw_limit", 0.10)
            ),
            slot_ids=state.segment_counter[env_ids],
            translation_anchors=getattr(config, 'translation_retention_anchors', None),
            zero_retention=getattr(config, 'zero_retention', False),
        )
        state.commands[env_ids, 0] = linear
        state.commands[env_ids, 1] = yaw
        state.sample_mode[env_ids] = mode
        state.segment_counter[env_ids] += 1
    elif strategy == "zero_reverse_mixture":
        linear, yaw, mode = sample_zero_reverse_mixture(
            ranges["lin_vel_x"][env_ids],
            ranges["ang_vel_yaw"][env_ids],
            zero_fraction=config.mixture_zero_fraction,
            small_fraction=config.mixture_small_fraction,
            reverse_fraction=config.mixture_reverse_fraction,
            forward_fraction=config.mixture_forward_fraction,
            small_linear_limit=config.mixture_small_linear_limit,
            small_yaw_limit=config.mixture_small_yaw_limit,
            small_yaw_only_fraction=config.mixture_small_yaw_only_fraction,
            linear_yaw_zero_fraction=config.mixture_linear_yaw_zero_fraction,
            small_linear_anchors=config.mixture_small_linear_anchors,
            small_yaw_anchors=config.mixture_small_yaw_anchors,
            endpoint_anchor_fraction=config.mixture_endpoint_anchor_fraction,
            endpoint_linear_anchors=config.mixture_endpoint_linear_anchors,
        )
        state.commands[env_ids, 0] = linear
        state.commands[env_ids, 1] = yaw
        state.sample_mode[env_ids] = mode
    elif strategy == "uniform":
        state.commands[env_ids, 0] = (
            ranges["lin_vel_x"][env_ids, 1]
            - ranges["lin_vel_x"][env_ids, 0]
        ) * torch.rand(len(env_ids), device=device) + ranges[
            "lin_vel_x"
        ][env_ids, 0]
        state.commands[env_ids, 1] = (
            ranges["ang_vel_yaw"][env_ids, 1]
            - ranges["ang_vel_yaw"][env_ids, 0]
        ) * torch.rand(len(env_ids), device=device) + ranges[
            "ang_vel_yaw"
        ][env_ids, 0]
        state.sample_mode[env_ids] = -1
    else:
        raise ValueError(f"unknown command sampling strategy: {strategy}")
    state.commands[env_ids, 2] = (
        ranges["height"][env_ids, 1]
        - ranges["height"][env_ids, 0]
    ) * torch.rand(len(env_ids), device=device) + ranges[
        "height"
    ][
        env_ids, 0
    ]

    if getattr(config,'start_stop_ramp_seconds',0.)>0:
        reset_start_stop(state.commands,env_ids)
    if strategy == 'height_bank':
        state.commands[env_ids, 2] = height_selected[:, 2]

    if config.heading_command:
        state.commands[env_ids, 3] = sample_heading(
            ranges["heading"][0],
            ranges["heading"][1],
            (len(env_ids), 1),
            device=device,
        ).squeeze(1)
