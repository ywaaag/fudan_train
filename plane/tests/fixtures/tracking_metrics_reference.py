"""Frozen metric accumulation method from before extraction."""
import torch

class OriginalMetrics:
    def _accumulate_command_tracking_metrics(self):
        command_x = self.commands[:, 0]
        base_vx = self.base_lin_vel[:, 0]
        error = torch.abs(command_x - base_vx)
        self.command_metric_count += 1.0
        self.command_metric_command_x_sum += command_x
        self.command_metric_base_vx_sum += base_vx
        self.command_metric_abs_error_sum += error

        zero = self._zero_command_mask().float()
        self.command_metric_zero_count += zero
        self.command_metric_zero_abs_vx_sum += zero * torch.abs(base_vx)
        self.command_metric_zero_abs_yaw_sum += zero * torch.abs(self.base_ang_vel[:, 2])
        self.command_metric_zero_abs_wheel_sum += zero * torch.mean(
            torch.abs(self.dof_vel[:, [2, 5]]), dim=1
        )

        reverse = (command_x < -self.cfg.rewards.zero_command_threshold).float()
        self.command_metric_reverse_count += reverse
        self.command_metric_reverse_command_sum += reverse * command_x
        self.command_metric_reverse_vx_sum += reverse * base_vx
        self.command_metric_reverse_error_sum += reverse * error

        forward = (command_x > self.cfg.rewards.zero_command_threshold).float()
        self.command_metric_forward_count += forward
        self.command_metric_forward_command_sum += forward * command_x
        self.command_metric_forward_vx_sum += forward * base_vx
        self.command_metric_forward_error_sum += forward * error

        yaw_error = torch.abs(self.commands[:, 1] - self.base_ang_vel[:, 2])
        yaw = (
            torch.abs(self.commands[:, 1])
            > self.cfg.rewards.zero_command_threshold
        ).float()
        self.command_metric_yaw_count += yaw
        self.command_metric_yaw_command_abs_sum += yaw * torch.abs(self.commands[:, 1])
        self.command_metric_yaw_error_sum += yaw * yaw_error
        if self._method_v1:
            wheel_terms = self._method_wheel_terms()
            slip = wheel_terms["residual_rms"]
            contact_fraction = self.wheel_contact_history[:, 0].float().mean(dim=1)
        else:
            wheel_speed = -torch.mean(self.dof_vel[:, [2, 5]], dim=1) * float(
                self.cfg.rewards.wheel_radius
            )
            slip = torch.square(wheel_speed - base_vx)
            contact_fraction = torch.zeros_like(slip)
        # Keep this diagnostic unmasked so low-speed/zero-command wheel spin is visible.
        self.command_metric_wheel_slip_sum += slip
        self.command_metric_wheel_slip_rms_sum += slip
        self.command_metric_wheel_contact_fraction_sum += contact_fraction
        self.command_metric_torque_saturation_sum += (
            torch.abs(self.torques) >= self.torque_limits * 0.99
        ).float().mean(dim=1)


