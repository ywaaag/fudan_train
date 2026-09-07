"""Command samplers that keep training mixtures explicit and testable."""

from __future__ import annotations

import torch


MODE_ZERO = 0
MODE_SMALL = 1
MODE_REVERSE = 2
MODE_FORWARD = 3
MODE_NAMES = ("zero", "small", "reverse", "forward")


def _uniform(lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    return lower + (upper - lower) * torch.rand_like(lower)


def sample_zero_reverse_mixture(
    linear_ranges: torch.Tensor,
    yaw_ranges: torch.Tensor,
    *,
    zero_fraction: float,
    small_fraction: float,
    reverse_fraction: float,
    forward_fraction: float,
    small_linear_limit: float,
    small_yaw_limit: float,
    small_yaw_only_fraction: float,
    linear_yaw_zero_fraction: float,
    small_linear_anchors: tuple[float, ...] | None = None,
    small_yaw_anchors: tuple[float, ...] | None = None,
    endpoint_anchor_fraction: float = 0.0,
    endpoint_linear_anchors: tuple[float, ...] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample exact-zero, small, reverse and forward command modes."""

    if linear_ranges.ndim != 2 or linear_ranges.shape[1] != 2:
        raise ValueError("linear_ranges must have shape (N,2)")
    if yaw_ranges.shape != linear_ranges.shape:
        raise ValueError("yaw_ranges must match linear_ranges")
    fractions = torch.tensor(
        [zero_fraction, small_fraction, reverse_fraction, forward_fraction],
        dtype=torch.float64,
    )
    if torch.any(fractions < 0.0) or abs(float(fractions.sum()) - 1.0) > 1e-9:
        raise ValueError("command mixture fractions must be non-negative and sum to one")
    if small_linear_limit <= 0.0 or small_yaw_limit <= 0.0:
        raise ValueError("small command limits must be positive")
    if not 0.0 <= small_yaw_only_fraction <= 1.0:
        raise ValueError("small_yaw_only_fraction must be in [0,1]")
    if not 0.0 <= linear_yaw_zero_fraction <= 1.0:
        raise ValueError("linear_yaw_zero_fraction must be in [0,1]")
    if not 0.0 <= endpoint_anchor_fraction <= 1.0:
        raise ValueError("endpoint_anchor_fraction must be in [0,1]")

    count = linear_ranges.shape[0]
    draw = torch.rand(count, device=linear_ranges.device)
    boundaries = torch.cumsum(
        fractions.to(device=linear_ranges.device, dtype=draw.dtype), dim=0
    )
    mode = torch.bucketize(draw, boundaries[:-1])
    linear = _uniform(linear_ranges[:, 0], linear_ranges[:, 1])
    yaw = _uniform(yaw_ranges[:, 0], yaw_ranges[:, 1])

    zero = mode == MODE_ZERO
    linear[zero] = 0.0
    yaw[zero] = 0.0

    small = mode == MODE_SMALL
    small_linear_lower = torch.maximum(
        linear_ranges[:, 0], torch.full_like(linear, -small_linear_limit)
    )
    small_linear_upper = torch.minimum(
        linear_ranges[:, 1], torch.full_like(linear, small_linear_limit)
    )
    small_yaw_lower = torch.maximum(
        yaw_ranges[:, 0], torch.full_like(yaw, -small_yaw_limit)
    )
    small_yaw_upper = torch.minimum(
        yaw_ranges[:, 1], torch.full_like(yaw, small_yaw_limit)
    )
    if small_linear_anchors:
        anchor_values = torch.as_tensor(
            small_linear_anchors, device=linear.device, dtype=linear.dtype
        )
        sampled_small_linear = anchor_values[
            torch.randint(anchor_values.numel(), (count,), device=linear.device)
        ]
        sampled_small_linear = torch.maximum(
            small_linear_lower, torch.minimum(small_linear_upper, sampled_small_linear)
        )
    else:
        sampled_small_linear = _uniform(small_linear_lower, small_linear_upper)
    if small_yaw_anchors:
        anchor_values = torch.as_tensor(
            small_yaw_anchors, device=yaw.device, dtype=yaw.dtype
        )
        sampled_small_yaw = anchor_values[
            torch.randint(anchor_values.numel(), (count,), device=yaw.device)
        ]
        sampled_small_yaw = torch.maximum(
            small_yaw_lower, torch.minimum(small_yaw_upper, sampled_small_yaw)
        )
    else:
        sampled_small_yaw = _uniform(small_yaw_lower, small_yaw_upper)
    yaw_only = torch.rand(count, device=linear.device) < small_yaw_only_fraction
    linear[small] = torch.where(
        yaw_only[small], torch.zeros_like(linear[small]), sampled_small_linear[small]
    )
    yaw[small] = torch.where(
        yaw_only[small], sampled_small_yaw[small], torch.zeros_like(yaw[small])
    )

    reverse = mode == MODE_REVERSE
    reverse_upper = torch.minimum(linear_ranges[:, 1], torch.zeros_like(linear))
    linear[reverse] = _uniform(linear_ranges[:, 0], reverse_upper)[reverse]

    forward = mode == MODE_FORWARD
    forward_lower = torch.maximum(linear_ranges[:, 0], torch.zeros_like(linear))
    linear[forward] = _uniform(forward_lower, linear_ranges[:, 1])[forward]

    # Explicit endpoint anchors improve coverage of the edge of each signed
    # curriculum range. Values are clamped, so one config works at all stages.
    if endpoint_linear_anchors and endpoint_anchor_fraction > 0.0:
        anchors = torch.as_tensor(endpoint_linear_anchors, device=linear.device, dtype=linear.dtype)
        # Store magnitudes in either sign convention; the mode determines the
        # sign so a forward sample can never accidentally become zero because
        # a negative anchor was selected (and vice versa).
        anchor = anchors[torch.randint(anchors.numel(), (count,), device=linear.device)].abs()
        reverse_value = torch.maximum(linear_ranges[:, 0], torch.minimum(linear_ranges[:, 1], -anchor))
        forward_value = torch.maximum(linear_ranges[:, 0], torch.minimum(linear_ranges[:, 1], anchor))
        use_anchor = torch.rand(count, device=linear.device) < endpoint_anchor_fraction
        reverse_anchor = reverse & use_anchor
        forward_anchor = forward & use_anchor
        linear[reverse_anchor] = reverse_value[reverse_anchor]
        linear[forward_anchor] = forward_value[forward_anchor]

    linear_motion = reverse | forward
    yaw_zero = torch.rand(count, device=linear.device) < linear_yaw_zero_fraction
    yaw[linear_motion & yaw_zero] = 0.0
    return linear, yaw, mode
