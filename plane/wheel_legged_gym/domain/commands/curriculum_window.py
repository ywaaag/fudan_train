"""Scalar staged-course aggregation, separate from promotion and reset timing."""
import torch


def create_window(device):
    names = (
        "timeouts",
        "zero_count",
        "zero_abs_vx_sum",
        "zero_abs_yaw_sum",
        "reverse_command_abs_sum",
        "reverse_error_sum",
        "forward_command_abs_sum",
        "forward_error_sum",
        "yaw_command_abs_sum",
        "yaw_error_sum",
    )
    return {
        name: torch.zeros((), dtype=torch.float, device=device)
        for name in names
    }



def accumulate_window(window, valid_ids, *, timeouts, metrics):
    window["timeouts"] += timeouts[valid_ids].float().sum()
    window["zero_count"] += metrics.zero_count[valid_ids].sum()
    window["zero_abs_vx_sum"] += metrics.zero_abs_vx_sum[
        valid_ids
    ].sum()
    window["zero_abs_yaw_sum"] += metrics.zero_abs_yaw_sum[
        valid_ids
    ].sum()
    window["reverse_command_abs_sum"] += torch.abs(
        metrics.reverse_command_sum[valid_ids]
    ).sum()
    window["reverse_error_sum"] += metrics.reverse_error_sum[
        valid_ids
    ].sum()
    window["forward_command_abs_sum"] += metrics.forward_command_sum[
        valid_ids
    ].sum()
    window["forward_error_sum"] += metrics.forward_error_sum[
        valid_ids
    ].sum()
    window["yaw_command_abs_sum"] += metrics.yaw_command_abs_sum[
        valid_ids
    ].sum()
    window["yaw_error_sum"] += metrics.yaw_error_sum[valid_ids].sum()

