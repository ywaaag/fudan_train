"""Command tracking accumulation with explicit mutable metrics and state inputs."""
import torch


def accumulate(metrics, *, commands, linear_velocity, angular_velocity, dof_velocity,
               torques, torque_limits, zero_threshold, wheel_radius,
               zero_mask, method_wheel_terms, wheel_contact_history, method_v1):
    """Update only named accumulators; callbacks preserve original evaluation order."""
    command_x = commands[:, 0]
    base_vx = linear_velocity[:, 0]
    error = torch.abs(command_x - base_vx)
    metrics.count[:] += 1.0
    metrics.command_x_sum[:] += command_x
    metrics.base_vx_sum[:] += base_vx
    metrics.abs_error_sum[:] += error

    zero = zero_mask().float()
    metrics.zero_count[:] += zero
    metrics.zero_abs_vx_sum[:] += zero * torch.abs(base_vx)
    metrics.zero_abs_yaw_sum[:] += zero * torch.abs(angular_velocity[:, 2])
    metrics.zero_abs_wheel_sum[:] += zero * torch.mean(
        torch.abs(dof_velocity[:, [2, 5]]), dim=1
    )

    reverse = (command_x < -zero_threshold).float()
    metrics.reverse_count[:] += reverse
    metrics.reverse_command_sum[:] += reverse * command_x
    metrics.reverse_vx_sum[:] += reverse * base_vx
    metrics.reverse_error_sum[:] += reverse * error

    forward = (command_x > zero_threshold).float()
    metrics.forward_count[:] += forward
    metrics.forward_command_sum[:] += forward * command_x
    metrics.forward_vx_sum[:] += forward * base_vx
    metrics.forward_error_sum[:] += forward * error

    yaw_error = torch.abs(commands[:, 1] - angular_velocity[:, 2])
    yaw = (
        torch.abs(commands[:, 1])
        > zero_threshold
    ).float()
    metrics.yaw_count[:] += yaw
    metrics.yaw_command_abs_sum[:] += yaw * torch.abs(commands[:, 1])
    metrics.yaw_error_sum[:] += yaw * yaw_error
    if method_v1:
        wheel_terms = method_wheel_terms()
        slip = wheel_terms["residual_rms"]
        contact_fraction = wheel_contact_history[:, 0].float().mean(dim=1)
    else:
        wheel_speed = -torch.mean(dof_velocity[:, [2, 5]], dim=1) * float(
            wheel_radius
        )
        slip = torch.square(wheel_speed - base_vx)
        contact_fraction = torch.zeros_like(slip)
    # Keep this diagnostic unmasked so low-speed/zero-command wheel spin is visible.
    metrics.wheel_slip_sum[:] += slip
    metrics.wheel_slip_rms_sum[:] += slip
    metrics.wheel_contact_fraction_sum[:] += contact_fraction
    metrics.torque_saturation_sum[:] += (
        torch.abs(torques) >= torque_limits * 0.99
    ).float().mean(dim=1)



def summarize(metrics, env_ids):
    """Episode-weighted diagnostics before reset, with historical denominator clamps."""
    metric_count = torch.clamp(metrics.count[env_ids].sum(), min=1.0)
    zero_count = torch.clamp(metrics.zero_count[env_ids].sum(), min=1.0)
    reverse_count = torch.clamp(
        metrics.reverse_count[env_ids].sum(), min=1.0
    )
    forward_count = torch.clamp(
        metrics.forward_count[env_ids].sum(), min=1.0
    )
    command_metrics = {
        "command_x_mean": metrics.command_x_sum[env_ids].sum()
        / metric_count,
        "base_vx_mean": metrics.base_vx_sum[env_ids].sum()
        / metric_count,
        "tracking_abs_error": metrics.abs_error_sum[env_ids].sum()
        / metric_count,
        "zero_abs_vx": metrics.zero_abs_vx_sum[env_ids].sum()
        / zero_count,
        "zero_abs_yaw_rate": metrics.zero_abs_yaw_sum[env_ids].sum()
        / zero_count,
        "zero_abs_wheel_speed": metrics.zero_abs_wheel_sum[
            env_ids
        ].sum()
        / zero_count,
        "reverse_command_x_mean": metrics.reverse_command_sum[
            env_ids
        ].sum()
        / reverse_count,
        "reverse_base_vx_mean": metrics.reverse_vx_sum[env_ids].sum()
        / reverse_count,
        "reverse_tracking_abs_error": metrics.reverse_error_sum[
            env_ids
        ].sum()
        / reverse_count,
        "forward_command_x_mean": metrics.forward_command_sum[
            env_ids
        ].sum()
        / forward_count,
        "forward_base_vx_mean": metrics.forward_vx_sum[env_ids].sum()
        / forward_count,
        "forward_tracking_abs_error": metrics.forward_error_sum[
            env_ids
        ].sum()
        / forward_count,
        "wheel_slip_m_s2": metrics.wheel_slip_sum[env_ids].sum()
        / metric_count,
        "wheel_slip_rms_m_s": metrics.wheel_slip_rms_sum[env_ids].sum()
        / metric_count,
        "wheel_contact_fraction": metrics.wheel_contact_fraction_sum[
            env_ids
        ].sum()
        / metric_count,
        "torque_saturation_fraction": metrics.torque_saturation_sum[
            env_ids
        ].sum()
        / metric_count,
        "preclip_torque_saturation_fraction": metrics.preclip_torque_saturation_sum[
            env_ids
        ].sum()
        / metric_count,
        "action_clip_fraction": metrics.action_clip_sum[env_ids].sum()
        / metric_count,
    }
    return command_metrics
