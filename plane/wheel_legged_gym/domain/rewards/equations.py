"""Reward equations on explicit read-only inputs. No simulator, registry, or CLI dependency.

Equations preserve legacy evaluation order and clipping; only their ownership changed.
The environment constructs RewardInputs. Rewards do not fetch simulator state themselves.
"""
import torch
from .inputs import RewardInputs
from .terms import (CMD_HEIGHT, CMD_VX, CMD_YAW, capped_tracking_terms, normalized_huber,
                    smooth_gate, wheel_rolling_terms, bilateral_geometry_cost)
from wheel_legged_gym.domain.geometry.rotations import quat_rotate_inverse, wrap_to_pi

def _method_task_gate(state: RewardInputs):
    """Smoothly suppress task bonus when the body is badly tilted/fallen."""

    upright_error = torch.norm(state.projected_gravity[:, :2], dim=1)
    upright = smooth_gate(
        upright_error, float(state.cfg.rewards.upright_gate_tolerance)
    )
    height = smooth_gate(
        torch.abs(state.base_height - state.commands[:, CMD_HEIGHT]),
        float(state.cfg.rewards.height_gate_tolerance),
    )
    # Keep a small floor so a recovery trajectory still has a gradient;
    # safety/contact/termination costs remain ungated.
    return 0.1 + 0.9 * upright * height


def _method_wheel_terms(state: RewardInputs):
    if state.wheel_dof_indices is None:
        raise RuntimeError("method_v1 requires resolved wheel DOF indices")
    wheel_body_pos = state.rigid_body_states[:, state.wheel_body_indices, :3]
    relative = wheel_body_pos - state.root_states[:, :3].unsqueeze(1)
    base_quat = state.base_quat.unsqueeze(1).expand(-1, relative.shape[1], -1)
    relative_body = quat_rotate_inverse(
        base_quat.reshape(-1, 4), relative.reshape(-1, 3)
    ).reshape(state.num_envs, -1, 3)
    wheel_y = relative_body[:, :, 1]
    contact = state.wheel_contact_history[:, 0]
    return wheel_rolling_terms(
        state.base_lin_vel[:, CMD_VX],
        state.base_ang_vel[:, 2],
        state.dof_vel[:, state.wheel_dof_indices],
        wheel_y,
        float(state.cfg.rewards.wheel_radius),
        contact,
        slip_scale=float(state.cfg.rewards.wheel_slip_scale),
        air_spin_scale=float(state.cfg.rewards.wheel_air_spin_scale),
    )


def _reward_track_vx_coarse(state: RewardInputs):
    terms = capped_tracking_terms(
        state.commands[:, CMD_VX] - state.base_lin_vel[:, CMD_VX],
        float(state.cfg.rewards.tracking_linear_cap),
        coarse_sigma=float(state.cfg.rewards.tracking_coarse_sigma),
        fine_sigma=float(state.cfg.rewards.tracking_fine_sigma),
        gap_delta=float(state.cfg.rewards.tracking_gap_delta),
        gap_clip=float(state.cfg.rewards.tracking_gap_clip),
    )
    return terms["coarse"] * _method_task_gate(state)


def _reward_track_vx_fine(state: RewardInputs):
    error = state.commands[:, CMD_VX] - state.base_lin_vel[:, CMD_VX]
    terms = capped_tracking_terms(
        error,
        float(state.cfg.rewards.tracking_linear_cap),
        coarse_sigma=float(state.cfg.rewards.tracking_coarse_sigma),
        fine_sigma=float(state.cfg.rewards.tracking_fine_sigma),
        gap_delta=float(state.cfg.rewards.tracking_gap_delta),
        gap_clip=float(state.cfg.rewards.tracking_gap_clip),
    )
    return terms["fine"] * _method_task_gate(state)


def _reward_track_vx_gap(state: RewardInputs):
    error = state.commands[:, CMD_VX] - state.base_lin_vel[:, CMD_VX]
    terms = capped_tracking_terms(
        error,
        float(state.cfg.rewards.tracking_linear_cap),
        coarse_sigma=float(state.cfg.rewards.tracking_coarse_sigma),
        fine_sigma=float(state.cfg.rewards.tracking_fine_sigma),
        gap_delta=float(state.cfg.rewards.tracking_gap_delta),
        gap_clip=float(state.cfg.rewards.tracking_gap_clip),
    )
    return terms["gap"] * _method_task_gate(state)


def _reward_track_yaw_coarse(state: RewardInputs):
    error = state.commands[:, CMD_YAW] - state.base_ang_vel[:, 2]
    terms = capped_tracking_terms(
        error,
        float(state.cfg.rewards.tracking_yaw_cap),
        coarse_sigma=float(state.cfg.rewards.tracking_coarse_sigma),
        fine_sigma=float(state.cfg.rewards.tracking_fine_sigma),
        gap_delta=float(state.cfg.rewards.tracking_gap_delta),
        gap_clip=float(state.cfg.rewards.tracking_gap_clip),
    )
    return terms["coarse"] * _method_task_gate(state)


def _reward_track_yaw_fine(state: RewardInputs):
    error = state.commands[:, CMD_YAW] - state.base_ang_vel[:, 2]
    terms = capped_tracking_terms(
        error,
        float(state.cfg.rewards.tracking_yaw_cap),
        coarse_sigma=float(state.cfg.rewards.tracking_coarse_sigma),
        fine_sigma=float(state.cfg.rewards.tracking_fine_sigma),
        gap_delta=float(state.cfg.rewards.tracking_gap_delta),
        gap_clip=float(state.cfg.rewards.tracking_gap_clip),
    )
    return terms["fine"] * _method_task_gate(state)


def _reward_track_yaw_gap(state: RewardInputs):
    error = state.commands[:, CMD_YAW] - state.base_ang_vel[:, 2]
    terms = capped_tracking_terms(
        error,
        float(state.cfg.rewards.tracking_yaw_cap),
        coarse_sigma=float(state.cfg.rewards.tracking_coarse_sigma),
        fine_sigma=float(state.cfg.rewards.tracking_fine_sigma),
        gap_delta=float(state.cfg.rewards.tracking_gap_delta),
        gap_clip=float(state.cfg.rewards.tracking_gap_clip),
    )
    return terms["gap"] * _method_task_gate(state)


def _reward_height_cost(state: RewardInputs):
    return normalized_huber(
        state.base_height - state.commands[:, CMD_HEIGHT],
        float(state.cfg.rewards.height_gate_tolerance),
        clip=1.0,
    )


def _reward_lateral_velocity(state: RewardInputs):
    return normalized_huber(state.base_lin_vel[:, 1], 0.5, clip=1.0)


def _reward_wheel_slip(state: RewardInputs):
    return _method_wheel_terms(state)["contact_slip"]


def _reward_airborne_wheel_spin(state: RewardInputs):
    return _method_wheel_terms(state)["airborne_spin"]


def _reward_wheel_contact_loss(state: RewardInputs):
    return 1.0 - state.wheel_contact_history[:, 0].to(torch.float).mean(dim=1)


def _reward_forbidden_contact(state: RewardInputs):
    if state.termination_contact_indices.numel() == 0:
        return torch.zeros(state.num_envs, device=state.device)
    force = torch.norm(
        state.contact_forces[:, state.termination_contact_indices, :], dim=-1
    )
    return (force > float(state.cfg.rewards.forbidden_contact_force_threshold)).any(dim=1).float()


def _reward_torque_cost(state: RewardInputs):
    normalized = torch.abs(state.torques) / torch.clamp(state.torque_limits, min=1.0)
    return torch.clamp(torch.mean(torch.square(normalized), dim=1), max=1.0)


def _reward_power_cost(state: RewardInputs):
    torque_norm = torch.abs(state.torques) / torch.clamp(state.torque_limits, min=1.0)
    velocity_norm = torch.abs(state.dof_vel) / torch.clamp(state.dof_vel_limits, min=1.0)
    return torch.clamp(torch.mean(torque_norm * velocity_norm, dim=1), max=1.0)


def _reward_action_second_diff(state: RewardInputs):
    second = state.actions - 2.0 * state.last_actions[:, :, 0] + state.last_actions[:, :, 1]
    return torch.clamp(torch.mean(torch.square(second), dim=1), max=1.0)


def _reward_lin_vel_z(state: RewardInputs):
    # Penalize z axis base linear velocity
    if state._method_v1:
        return normalized_huber(state.base_lin_vel[:, 2], 0.5, clip=1.0)
    return torch.square(state.base_lin_vel[:, 2])


def _reward_ang_vel_xy(state: RewardInputs):
    # Penalize xy axes base angular velocity
    if state._method_v1:
        return torch.clamp(torch.mean(torch.square(state.base_ang_vel[:, :2]), dim=1), max=1.0)
    return torch.sum(torch.square(state.base_ang_vel[:, :2]), dim=1)


def _reward_orientation(state: RewardInputs):
    # Penalize non flat base orientation
    if state._method_v1:
        return normalized_huber(
            torch.norm(state.projected_gravity[:, :2], dim=1), 0.35, clip=1.0
        )
    return torch.sum(torch.square(state.projected_gravity[:, :2]), dim=1)


def _reward_base_height(state: RewardInputs):
    # Penalize base height away from target
    # print(state.commands[0, 2], state.base_height[0])
    if state.reward_scales["base_height"] < 0:
        return torch.abs(state.base_height - state.commands[:, 2])
    else:
        base_height_error = torch.square(state.base_height - state.commands[:, 2])
        return torch.exp(-base_height_error / 0.001)


def _reward_base_height_enhance(state: RewardInputs):
    base_height_error = torch.square(state.base_height - state.commands[:, 2])
    return torch.exp(-base_height_error / 0.001 / 10) - 1


def _reward_torques(state: RewardInputs):
    # Penalize torques
    return torch.sum(torch.square(state.torques), dim=1)


def _reward_power(state: RewardInputs):
    # Penalize torques
    return torch.sum(torch.abs(state.torques * state.dof_vel), dim=1)


def _reward_dof_vel(state: RewardInputs):
    # Penalize dof velocities
    return torch.sum(torch.square(state.dof_vel[:, :2]), dim=1) + torch.sum(
        torch.square(state.dof_vel[:, 3:5]), dim=1
    )


def _reward_dof_acc(state: RewardInputs):
    # Penalize dof accelerations
    return torch.sum(torch.square(state.dof_acc), dim=1)


def _reward_action_rate(state: RewardInputs):
    # Penalize changes in actions
    value = torch.mean(torch.square(state.last_actions[:, :, 0] - state.actions), dim=1)
    return torch.clamp(value, max=1.0) if state._method_v1 else value


def _reward_action_smooth(state: RewardInputs):
    # Penalize changes in actions
    return torch.sum(
        torch.square(
            state.actions[:, :2]
            - 2 * state.last_actions[:, :2, 0]
            + state.last_actions[:, :2, 1]
        ),
        dim=1,
    ) + torch.sum(
        torch.square(
            state.actions[:, 3:5]
            - 2 * state.last_actions[:, 3:5, 0]
            + state.last_actions[:, 3:5, 1]
        ),
        dim=1,
    )


def _reward_collision(state: RewardInputs):
    # Penalize collisions on selected bodies
    return torch.sum(
        1.0
        * (
            torch.norm(
                state.contact_forces[:, state.penalised_contact_indices, :], dim=-1
            )
            > 0.1
        ),
        dim=1,
    )


def _reward_termination(state: RewardInputs):
    # Terminal reward / penalty
    return state.reset_buf * ~state.time_out_buf


def _reward_dof_pos_limits(state: RewardInputs):
    # Penalize dof positions too close to the limit
    out_of_limits = -(state.dof_pos[:, :2] - state.dof_pos_limits[:2, 0]).clip(
        max=0.0
    )  # lower limit
    out_of_limits += (state.dof_pos[:, :2] - state.dof_pos_limits[:2, 1]).clip(
        min=0.0
    )
    out_of_limits += -(state.dof_pos[:, 3:5] - state.dof_pos_limits[3:5, 0]).clip(
        max=0.0
    )  # lower limit
    out_of_limits += (state.dof_pos[:, 3:5] - state.dof_pos_limits[3:5, 1]).clip(
        min=0.0
    )
    return torch.sum(out_of_limits, dim=1)


def _reward_dof_vel_limits(state: RewardInputs):
    # Penalize dof velocities too close to the limit
    # clip to max error = 1 rad/s per joint to avoid huge penalties
    return torch.sum(
        (
            torch.abs(state.dof_vel)
            - state.dof_vel_limits * state.cfg.rewards.soft_dof_vel_limit
        ).clip(min=0.0, max=1.0),
        dim=1,
    )


def _reward_torque_limits(state: RewardInputs):
    # penalize torques too close to the limit
    return torch.sum(
        (
            torch.abs(state.torques)
            - state.torque_limits * state.cfg.rewards.soft_torque_limit
        ).clip(min=0.0),
        dim=1,
    )


def _reward_tracking_lin_vel(state: RewardInputs):
    # Tracking of linear velocity commands (x axes)
    lin_vel_error = torch.square(state.commands[:, 0] - state.base_lin_vel[:, 0])
    return torch.exp(-lin_vel_error / state.cfg.rewards.tracking_sigma)


def _reward_tracking_lin_vel_enhance(state: RewardInputs):
    # Tracking of linear velocity commands (x axes)
    lin_vel_error = torch.square(state.commands[:, 0] - state.base_lin_vel[:, 0])
    return torch.exp(-lin_vel_error / state.cfg.rewards.tracking_sigma / 10) - 1


def _reward_high_speed_tracking(state: RewardInputs):
    """Give a useful tracking gradient only for high-speed commands."""
    error = torch.square(state.commands[:, 0] - state.base_lin_vel[:, 0])
    sigma = state.cfg.rewards.high_speed_tracking_sigma
    threshold = state.cfg.rewards.high_speed_command_threshold
    mask = torch.abs(state.commands[:, 0]) >= threshold
    return torch.exp(-error / sigma) * mask.float()


def _reward_high_speed_yaw_tracking(state: RewardInputs):
    error = torch.square(state.commands[:, 1] - state.base_ang_vel[:, 2])
    sigma = state.cfg.rewards.high_speed_yaw_tracking_sigma
    mask = torch.abs(state.commands[:, 0]) >= state.cfg.rewards.high_speed_command_threshold
    return torch.exp(-error / sigma) * mask.float()


def _reward_high_speed_yaw_penalty(state: RewardInputs):
    """Penalize squared yaw tracking error during high-speed travel."""
    # This project stores yaw command in commands[:, 1]; commands[:, 2] is height.
    error = torch.square(state.commands[:, 1] - state.base_ang_vel[:, 2])
    mask = torch.abs(state.commands[:, 0]) >= state.cfg.rewards.high_speed_yaw_penalty_command_threshold
    sigma = float(state.cfg.rewards.yaw_penalty_sigma)
    cost = 1.0 - torch.exp(-error / (sigma * sigma))
    return cost * mask.float()


def _reward_high_speed_slip(state: RewardInputs):
    wheel_speed = -torch.mean(state.dof_vel[:, [2, 5]], dim=1) * float(
        state.cfg.rewards.wheel_radius
    )
    slip = torch.square(wheel_speed - state.base_lin_vel[:, 0])
    sigma = float(state.cfg.rewards.wheel_slip_sigma)
    return 1.0 - torch.exp(-slip / (sigma * sigma))


def _reward_tracking_ang_vel(state: RewardInputs):
    # Tracking of angular velocity commands (yaw)
    ang_vel_error = torch.square(state.commands[:, 1] - state.base_ang_vel[:, 2])
    return torch.exp(-ang_vel_error / state.cfg.rewards.tracking_sigma)


def _reward_tracking_ang_vel_enhance(state: RewardInputs):
    # Tracking of angular velocity commands (x axes)
    ang_vel_error = torch.square(state.commands[:, 1] - state.base_ang_vel[:, 2])
    return torch.exp(-ang_vel_error / state.cfg.rewards.tracking_sigma / 10) - 1


def _zero_command_mask(state: RewardInputs):
    threshold = state.cfg.rewards.zero_command_threshold
    return torch.max(torch.abs(state.commands[:, :2]), dim=1).values <= threshold


def _reward_zero_base_velocity(state: RewardInputs):
    """Penalize actual forward/yaw motion only for exact zero commands."""
    penalty = torch.abs(state.base_lin_vel[:, 0])
    penalty += state.cfg.rewards.zero_yaw_rate_weight * torch.abs(
        state.base_ang_vel[:, 2]
    )
    if state._method_v1:
        penalty = normalized_huber(penalty, 0.5, clip=1.0)
    return penalty * _zero_command_mask(state)


def _reward_zero_wheel_velocity(state: RewardInputs):
    """Discourage the accepted policy's persistent zero-command wheel spin."""
    wheel_indices = (state.wheel_dof_indices if state.wheel_dof_indices is not None
                     else torch.tensor([2, 5], device=state.device))
    wheel_speed = torch.mean(torch.abs(state.dof_vel[:, wheel_indices]), dim=1)
    if state._method_v1:
        wheel_speed = normalized_huber(
            wheel_speed * float(state.cfg.rewards.wheel_radius), 0.5, clip=1.0
        )
    return wheel_speed * _zero_command_mask(state)


def _reward_low_speed_tracking(state: RewardInputs):
    """Provide a signed linear gradient at the command-grid operating points."""
    threshold = state.cfg.rewards.low_speed_command_threshold
    mask = (
        (torch.abs(state.commands[:, 0]) <= threshold)
        & (torch.abs(state.commands[:, 1]) <= threshold)
    )
    error = torch.abs(state.commands[:, 0] - state.base_lin_vel[:, 0])
    error += state.cfg.rewards.low_speed_yaw_weight * torch.abs(
        state.commands[:, 1] - state.base_ang_vel[:, 2]
    )
    return error * mask


def _reward_zero_yaw_wheel_symmetry(state: RewardInputs):
    """Keep the left/right wheel actions symmetric for straight motion.

    This targets linear commands (forward/backward and near-zero speed) while
    leaving turning-heavy commands unconstrained, so the policy does not need to
    sacrifice symmetry just to track yaw changes.
    """
    straight_mask = torch.abs(state.commands[:, 0]) <= state.cfg.rewards.max_straight_line_speed
    # Allow a small yaw buffer, but do not require exactly zero yaw before
    # symmetry is enforced. This keeps low-speed and straight-line motions
    # symmetric without forcing the turning case to be symmetric.
    straight_mask &= torch.abs(state.commands[:, 1]) <= state.cfg.rewards.zero_yaw_command_threshold
    return torch.abs(state.actions[:, 2] - state.actions[:, 5]) * straight_mask.float()


def _reward_theta0_equ_0(state: RewardInputs):    
    left_theta0_error = torch.square(state.theta0[:, 0])
    right_theta0_error = torch.square(state.theta0[:, 1])

    return torch.exp(-(left_theta0_error + right_theta0_error) / state.cfg.rewards.tracking_sigma)


def _reward_tracking_lin_vel_pbrs(state: RewardInputs):
    delta_phi = ~state.reset_buf * (
        _reward_tracking_lin_vel(state) - state.rwd_linVelTrackPrev
    )
    # return lin_vel_error
    return delta_phi


def _reward_tracking_ang_vel_pbrs(state: RewardInputs):
    delta_phi = ~state.reset_buf * (
        _reward_tracking_ang_vel(state) - state.rwd_angVelTrackPrev
    )
    # return ang_vel_error
    return delta_phi


def _reward_stumble(state: RewardInputs):
    # Penalize feet hitting vertical surfaces
    return torch.any(
        torch.norm(state.contact_forces[:, state.feet_indices, :2], dim=2)
        > 5 * torch.abs(state.contact_forces[:, state.feet_indices, 2]),
        dim=1,
    )


def _reward_stand_still(state: RewardInputs):
    # Penalize motion at zero commands
    value = torch.sum(torch.abs(state.dof_pos - state.default_dof_pos), dim=1) * (
        torch.norm(state.commands[:, :2], dim=1) < 0.1
    )
    return torch.clamp(value, max=1.0) if state._method_v1 else value


def _reward_stand_bilateral_geometry(state: RewardInputs):
    relative = state.rigid_body_states[:, state.bilateral_body_indices, :3] - state.root_states[:, None, :3]
    quat = state.base_quat[:, None, :].expand(-1, 4, -1)
    points = quat_rotate_inverse(quat.reshape(-1, 4), relative.reshape(-1, 3)).reshape(-1, 4, 3)
    cost = bilateral_geometry_cost(points,
        tolerance=float(state.cfg.rewards.bilateral_geometry_tolerance_m),
        scale=float(state.cfg.rewards.bilateral_geometry_scale_m))
    stationary_command = torch.all(torch.abs(state.commands[:, :2]) < 0.01, dim=1)
    if getattr(state.cfg.rewards,'straight_bilateral_geometry',False):
        stationary_command=torch.abs(state.commands[:,1])<.01
    return cost * stationary_command.float()


def _reward_nominal_state(state: RewardInputs):
    if getattr(state.cfg.commands, 'training_profile', '') == 'fudan_stand_v1':
        # Preserve reference squared virtual-leg angle difference, but use
        # the actual tree geometry instead of the reference robot's signs.
        p = state.rigid_body_states[:, state.fudan_leg_landmarks, :3]
        direction = p[:, 2:] - p[:, :2]
        q = state.base_quat[:, None, :].expand(-1, 2, -1)
        d = quat_rotate_inverse(q.reshape(-1, 4), direction.reshape(-1, 3)).reshape(-1, 2, 3)
        angles = torch.atan2(d[:, :, 0], -d[:, :, 2])
        return torch.square(wrap_to_pi(angles[:, 0] - angles[:, 1]))
    # return torch.square(state.theta0[:, 0] - state.theta0[:, 1])
    if state.reward_scales["nominal_state"] < 0:
        return torch.square(state.theta0[:, 0] - state.theta0[:, 1])
    else:
        ang_diff = torch.square(state.theta0[:, 0] - state.theta0[:, 1])
        return torch.exp(-ang_diff / 0.1)


def _reward_feet_contact_forces(state: RewardInputs):
    # penalize high contact forces
    return torch.sum(
        (
            torch.norm(state.contact_forces[:, state.feet_indices, :], dim=-1)
            - state.cfg.rewards.max_contact_force
        ).clip(min=0.0),
        dim=1,
    )
