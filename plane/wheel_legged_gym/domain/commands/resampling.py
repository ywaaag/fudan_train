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
             reset_start_stop, sample_heading, reset_turn=None, reset_height_skill=None):
    """Mutate selected rows only; callbacks retain simulator-specific boundary behavior.

    The height draw runs even for height_bank, before its selected height override.
    Reset callback stays lazy and follows that draw; heading is always drawn last.
    """
    strategy = getattr(config, "sampling_strategy", "uniform")
    if strategy == "cornering_height_skill":
        from wheel_legged_gym.domain.commands.command_sampling import sample_fixed_bank
        slots = env_ids.remainder(config.cohort_cycle)
        retained_ids = env_ids[slots.remainder(2) == 1]
        height_ids = env_ids[torch.isin(slots, torch.as_tensor(config.height_slots, device=device))]
        mid_ids = env_ids[torch.isin(slots, torch.as_tensor(config.mid_slots, device=device))]
        high_ids = env_ids[torch.isin(slots, torch.as_tensor(config.high_slots, device=device))]
        if len(retained_ids):
            state.commands[retained_ids, :2] = sample_fixed_bank(
                ranges['lin_vel_x'][retained_ids], ranges['ang_vel_yaw'][retained_ids],
                state.segment_counter[retained_ids], config.retention_bank)
        if len(height_ids):
            table = state.commands.new_tensor(config.height_skill_bank)
            height_targets = table[(state.segment_counter[height_ids] +
                                    height_ids // config.cohort_cycle) % len(table)]
        if len(mid_ids):
            mid_targets = sample_fixed_bank(ranges['lin_vel_x'][mid_ids],
                ranges['ang_vel_yaw'][mid_ids],
                state.segment_counter[mid_ids] + mid_ids // config.cohort_cycle, config.turn_bank)
        if len(high_ids):
            high_targets = sample_fixed_bank(ranges['lin_vel_x'][high_ids],
                ranges['ang_vel_yaw'][high_ids],
                state.segment_counter[high_ids] + high_ids // config.cohort_cycle, config.high_turn_bank)
        state.sample_mode[env_ids] = -1
        state.segment_counter[env_ids] += 1
    elif strategy == "turn_envelope":
        from wheel_legged_gym.domain.commands.command_sampling import sample_fixed_bank
        turn = env_ids.remainder(config.turn_stride) == 0
        retained_ids, turn_ids = env_ids[~turn], env_ids[turn]
        if len(retained_ids):
            state.commands[retained_ids, :2] = sample_fixed_bank(
                ranges['lin_vel_x'][retained_ids], ranges['ang_vel_yaw'][retained_ids],
                state.segment_counter[retained_ids], config.retention_bank)
        if len(turn_ids):
            targets = sample_fixed_bank(
                ranges['lin_vel_x'][turn_ids], ranges['ang_vel_yaw'][turn_ids],
                state.segment_counter[turn_ids] // config.turn_stride, config.turn_bank)
        state.sample_mode[env_ids] = -1
        state.segment_counter[env_ids] += 1
    elif strategy == "height_bank":
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
    if strategy == 'turn_envelope' and len(turn_ids):
        reset_turn(state.commands, turn_ids, targets)
    if strategy == 'cornering_height_skill':
        if len(height_ids):
            reset_height_skill(state.commands, height_ids, height_targets)
        if len(mid_ids):
            reset_turn(state.commands, mid_ids, mid_targets)
        if len(high_ids):
            reset_turn(state.commands, high_ids, high_targets)
    if strategy == 'height_bank':
        state.commands[env_ids, 2] = height_selected[:, 2]

    if config.heading_command:
        state.commands[env_ids, 3] = sample_heading(
            ranges["heading"][0],
            ranges["heading"][1],
            (len(env_ids), 1),
            device=device,
        ).squeeze(1)
