"""Named per-environment command accumulators, independently owned by each task."""
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class CommandMetrics:
    """Frozen field bindings; tensors remain mutable for step accumulation/reset."""
    count: torch.Tensor
    command_x_sum: torch.Tensor
    base_vx_sum: torch.Tensor
    abs_error_sum: torch.Tensor
    zero_count: torch.Tensor
    zero_abs_vx_sum: torch.Tensor
    zero_abs_yaw_sum: torch.Tensor
    zero_abs_wheel_sum: torch.Tensor
    reverse_count: torch.Tensor
    reverse_command_sum: torch.Tensor
    reverse_vx_sum: torch.Tensor
    reverse_error_sum: torch.Tensor
    forward_count: torch.Tensor
    forward_command_sum: torch.Tensor
    forward_vx_sum: torch.Tensor
    forward_error_sum: torch.Tensor
    yaw_count: torch.Tensor
    yaw_command_abs_sum: torch.Tensor
    yaw_error_sum: torch.Tensor
    wheel_slip_sum: torch.Tensor
    wheel_slip_rms_sum: torch.Tensor
    wheel_contact_fraction_sum: torch.Tensor
    torque_saturation_sum: torch.Tensor
    preclip_torque_saturation_sum: torch.Tensor
    action_clip_sum: torch.Tensor

    def buffers(self):
        """All fields in the historical allocation/reset order; no reflection."""
        return [
            self.count,
            self.command_x_sum,
            self.base_vx_sum,
            self.abs_error_sum,
            self.zero_count,
            self.zero_abs_vx_sum,
            self.zero_abs_yaw_sum,
            self.zero_abs_wheel_sum,
            self.reverse_count,
            self.reverse_command_sum,
            self.reverse_vx_sum,
            self.reverse_error_sum,
            self.forward_count,
            self.forward_command_sum,
            self.forward_vx_sum,
            self.forward_error_sum,
            self.yaw_count,
            self.yaw_command_abs_sum,
            self.yaw_error_sum,
            self.wheel_slip_sum,
            self.wheel_slip_rms_sum,
            self.wheel_contact_fraction_sum,
            self.torque_saturation_sum,
            self.preclip_torque_saturation_sum,
            self.action_clip_sum,
        ]


def create_command_metrics(num_envs, device):
    def empty():
        return torch.zeros(num_envs, dtype=torch.float, device=device, requires_grad=False)
    return CommandMetrics(
        count=empty(),
        command_x_sum=empty(),
        base_vx_sum=empty(),
        abs_error_sum=empty(),
        zero_count=empty(),
        zero_abs_vx_sum=empty(),
        zero_abs_yaw_sum=empty(),
        zero_abs_wheel_sum=empty(),
        reverse_count=empty(),
        reverse_command_sum=empty(),
        reverse_vx_sum=empty(),
        reverse_error_sum=empty(),
        forward_count=empty(),
        forward_command_sum=empty(),
        forward_vx_sum=empty(),
        forward_error_sum=empty(),
        yaw_count=empty(),
        yaw_command_abs_sum=empty(),
        yaw_error_sum=empty(),
        wheel_slip_sum=empty(),
        wheel_slip_rms_sum=empty(),
        wheel_contact_fraction_sum=empty(),
        torque_saturation_sum=empty(),
        preclip_torque_saturation_sum=empty(),
        action_clip_sum=empty(),
    )
