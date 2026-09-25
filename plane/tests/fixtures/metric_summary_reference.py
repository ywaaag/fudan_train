"""Frozen reset diagnostic aggregation before extraction, 2026-09-24."""
import torch

def summarize(self, env_ids):
    metric_count = torch.clamp(self.command_metric_count[env_ids].sum(), min=1.0)
    zero_count = torch.clamp(self.command_metric_zero_count[env_ids].sum(), min=1.0)
    reverse_count = torch.clamp(
        self.command_metric_reverse_count[env_ids].sum(), min=1.0
    )
    forward_count = torch.clamp(
        self.command_metric_forward_count[env_ids].sum(), min=1.0
    )
    command_metrics = {
        "command_x_mean": self.command_metric_command_x_sum[env_ids].sum()
        / metric_count,
        "base_vx_mean": self.command_metric_base_vx_sum[env_ids].sum()
        / metric_count,
        "tracking_abs_error": self.command_metric_abs_error_sum[env_ids].sum()
        / metric_count,
        "zero_abs_vx": self.command_metric_zero_abs_vx_sum[env_ids].sum()
        / zero_count,
        "zero_abs_yaw_rate": self.command_metric_zero_abs_yaw_sum[env_ids].sum()
        / zero_count,
        "zero_abs_wheel_speed": self.command_metric_zero_abs_wheel_sum[
            env_ids
        ].sum()
        / zero_count,
        "reverse_command_x_mean": self.command_metric_reverse_command_sum[
            env_ids
        ].sum()
        / reverse_count,
        "reverse_base_vx_mean": self.command_metric_reverse_vx_sum[env_ids].sum()
        / reverse_count,
        "reverse_tracking_abs_error": self.command_metric_reverse_error_sum[
            env_ids
        ].sum()
        / reverse_count,
        "forward_command_x_mean": self.command_metric_forward_command_sum[
            env_ids
        ].sum()
        / forward_count,
        "forward_base_vx_mean": self.command_metric_forward_vx_sum[env_ids].sum()
        / forward_count,
        "forward_tracking_abs_error": self.command_metric_forward_error_sum[
            env_ids
        ].sum()
        / forward_count,
        "wheel_slip_m_s2": self.command_metric_wheel_slip_sum[env_ids].sum()
        / metric_count,
        "wheel_slip_rms_m_s": self.command_metric_wheel_slip_rms_sum[env_ids].sum()
        / metric_count,
        "wheel_contact_fraction": self.command_metric_wheel_contact_fraction_sum[
            env_ids
        ].sum()
        / metric_count,
        "torque_saturation_fraction": self.command_metric_torque_saturation_sum[
            env_ids
        ].sum()
        / metric_count,
        "preclip_torque_saturation_fraction": self.command_metric_preclip_torque_saturation_sum[
            env_ids
        ].sum()
        / metric_count,
        "action_clip_fraction": self.command_metric_action_clip_sum[env_ids].sum()
        / metric_count,
    }
    return command_metrics
