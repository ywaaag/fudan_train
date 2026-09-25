"""Initialize joint defaults, gains and randomization with explicit inputs."""
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class InitialControlState:
    raw_default_position: torch.Tensor
    default_position: torch.Tensor
    delay_indices: torch.Tensor


def initialize(*, num_envs, dof_count, named_dof_count, dof_names, default_angles, control,
               randomization, physics_dt, device, p_gains, d_gains,
               torque_scales, delay_indices, sample_uniform, report):
    """Fill caller-owned gains/scales; return new defaults and selected delay buffer.

    Multiple matching gain keys deliberately retain last-match-wins semantics.
    Random draws remain Kp, Kd, torque, position, delay. The caller provides the
    existing sampler and reporter; importing this module has no simulator effects.
    """
    # joint positions offsets and PD gains
    raw_default_position = torch.zeros(
        dof_count,
        dtype=torch.float,
        device=device,
        requires_grad=False,
    )
    default_position = torch.zeros(
        num_envs,
        dof_count,
        dtype=torch.float,
        device=device,
        requires_grad=False,
    )
    for i in range(named_dof_count):
        name = dof_names[i]
        angle = default_angles[name]
        raw_default_position[i] = angle
        default_position[:, i] = angle
        found = False
        for dof_name in control.stiffness.keys():
            if dof_name in name:
                p_gains[:, i] = control.stiffness[dof_name]
                d_gains[:, i] = control.damping[dof_name]
                found = True
        if not found:
            p_gains[:, i] = 0.0
            d_gains[:, i] = 0.0
            if control.control_type in ["P", "V"]:
                report(
                    f"PD gain of joint {name} were not defined, setting them to zero"
                )
    if randomization.randomize_Kp:
        (
            p_gains_scale_min,
            p_gains_scale_max,
        ) = randomization.randomize_Kp_range
        p_gains *= sample_uniform(
            p_gains_scale_min,
            p_gains_scale_max,
            p_gains.shape,
            device=device,
        )
    if randomization.randomize_Kd:
        (
            d_gains_scale_min,
            d_gains_scale_max,
        ) = randomization.randomize_Kd_range
        d_gains *= sample_uniform(
            d_gains_scale_min,
            d_gains_scale_max,
            d_gains.shape,
            device=device,
        )
    if randomization.randomize_motor_torque:
        (
            torque_scale_min,
            torque_scale_max,
        ) = randomization.randomize_motor_torque_range
        torque_scales *= sample_uniform(
            torque_scale_min,
            torque_scale_max,
            torque_scales.shape,
            device=device,
        )
    if randomization.randomize_default_dof_pos:
        default_position += sample_uniform(
            randomization.randomize_default_dof_pos_range[0],
            randomization.randomize_default_dof_pos_range[1],
            (num_envs, dof_count),
            device=device,
        )
    if randomization.randomize_action_delay:
        action_delay_idx = torch.round(
            sample_uniform(
                randomization.delay_ms_range[0] / 1000 / physics_dt,
                randomization.delay_ms_range[1] / 1000 / physics_dt,
                (num_envs, 1),
                device=device,
            )
        ).squeeze(-1)
        delay_indices = action_delay_idx.long()
    return InitialControlState(raw_default_position, default_position, delay_indices)
