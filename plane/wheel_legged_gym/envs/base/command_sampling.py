"""Command samplers that keep training mixtures explicit and testable."""

from __future__ import annotations

import torch


MODE_ZERO = 0
MODE_SMALL = 1
MODE_REVERSE = 2
MODE_FORWARD = 3
MODE_NAMES = ("zero", "small", "reverse", "forward")


def sample_fixed_bank(linear_ranges, yaw_ranges, slot_ids, bank):
    """Explicit weighted (vx, yaw) slots; reject clipping that hides lost anchors."""
    table = linear_ranges.new_tensor(bank)
    if table.ndim != 2 or table.shape[1] != 2 or not table.shape[0] or not torch.isfinite(table).all():
        raise ValueError('command bank must be finite nonempty (N,2)')
    selected = table[slot_ids % table.shape[0]]
    for col, ranges in enumerate((linear_ranges, yaw_ranges)):
        if ((selected[:, col] < ranges[:, 0]) | (selected[:, col] > ranges[:, 1])).any():
            raise ValueError('command bank exceeds configured ranges')
    return selected


def _signed_endpoint(limit: torch.Tensor, sign: torch.Tensor) -> torch.Tensor:
    """Return an endpoint command with deterministic sign coverage."""

    return sign * limit


def sample_method_v1(
    linear_ranges: torch.Tensor,
    yaw_ranges: torch.Tensor,
    *,
    phase: str,
    small_linear_limit: float = 0.10,
    small_yaw_limit: float = 0.10,
    slot_ids: torch.Tensor | None = None,
    translation_anchors: tuple[float, ...] | None = None,
    zero_retention: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample the staged method_v1 command modes.

    Commands are sampled once per episode by the environment.  The sampler is
    intentionally stratified instead of relying on a single uniform draw so
    both signs and pure-axis modes remain visible to the curriculum evaluator.
    """

    if linear_ranges.ndim != 2 or linear_ranges.shape[1] != 2:
        raise ValueError("linear_ranges must have shape (N,2)")
    if yaw_ranges.shape != linear_ranges.shape:
        raise ValueError("yaw_ranges must match linear_ranges")
    if phase not in {"stand", "translate", "yaw", "combined"}:
        raise ValueError(f"unknown method_v1 phase: {phase}")

    count = linear_ranges.shape[0]
    device = linear_ranges.device
    dtype = linear_ranges.dtype
    if slot_ids is None:
        slot_ids = torch.arange(count, device=device)
    if slot_ids.shape != (count,):
        raise ValueError("slot_ids must have shape (N,)")
    linear = torch.zeros(count, device=device, dtype=dtype)
    yaw = torch.zeros(count, device=device, dtype=dtype)
    mode = torch.full((count,), MODE_ZERO, device=device, dtype=torch.long)
    if phase == "stand":
        return linear, yaw, mode

    lin_limit = torch.maximum(torch.abs(linear_ranges[:, 0]), torch.abs(linear_ranges[:, 1]))
    yaw_limit = torch.maximum(torch.abs(yaw_ranges[:, 0]), torch.abs(yaw_ranges[:, 1]))
    sign = torch.where(
        slot_ids % 2 == 0,
        torch.ones(count, device=device, dtype=dtype),
        -torch.ones(count, device=device, dtype=dtype),
    )

    if phase == "translate":
        if zero_retention:
            if translation_anchors is not None and len(translation_anchors) > 2:
                anchors = linear.new_tensor(translation_anchors)
                if not torch.isfinite(anchors).all() or not (anchors > 0).all():
                    raise ValueError('translation anchors must be finite positive magnitudes')
                bucket = slot_ids % 10
                reverse, forward = (bucket >= 4) & (bucket < 7), bucket >= 7
                magnitude = anchors[(slot_ids // 10) % anchors.numel()]
                for mask, sign_value, label in [(reverse,-1.,MODE_REVERSE),(forward,1.,MODE_FORWARD)]:
                    linear[mask] = torch.clamp(sign_value*magnitude[mask],
                        min=linear_ranges[mask,0],max=linear_ranges[mask,1])
                    mode[mask] = label
                return linear,yaw,mode
            # Same +/-0.5 and +/-1 endpoint exposure (15% each), reallocating
            # the old +/-0.1 slots to exact zero (40%). No command offset.
            bucket = slot_ids % 20
            for lower, upper, value, label in [(8,11,-.5,MODE_REVERSE),
                    (11,14,.5,MODE_FORWARD),(14,17,-1.,MODE_REVERSE),(17,20,1.,MODE_FORWARD)]:
                mask=(bucket>=lower)&(bucket<upper)
                linear[mask]=torch.clamp(torch.full_like(linear[mask],value),
                    min=linear_ranges[mask,0],max=linear_ranges[mask,1])
                mode[mask]=label
            return linear,yaw,mode
        # 20% exact zero, 20% small anchors, then balanced reverse/forward.
        bucket = slot_ids % 10
        small = (bucket >= 2) & (bucket < 4)
        reverse = (bucket >= 4) & (bucket < 7)
        forward = bucket >= 7
        linear[small] = torch.clamp(
            sign[small] * torch.minimum(lin_limit[small], lin_limit[small].new_tensor(small_linear_limit)),
            min=linear_ranges[small, 0],
            max=linear_ranges[small, 1],
        )
        linear[reverse] = torch.minimum(linear_ranges[reverse, 1], -lin_limit[reverse])
        linear[forward] = torch.maximum(linear_ranges[forward, 0], lin_limit[forward])
        if translation_anchors is not None:
            anchors = linear.new_tensor(translation_anchors)
            if anchors.ndim != 1 or not anchors.numel() or not torch.isfinite(anchors).all() or not (anchors > 0).all():
                raise ValueError('translation anchors must be finite positive magnitudes')
            magnitude = anchors[(slot_ids // 10) % anchors.numel()]
            linear[reverse] = torch.clamp(-magnitude[reverse],
                min=linear_ranges[reverse, 0], max=linear_ranges[reverse, 1])
            linear[forward] = torch.clamp(magnitude[forward],
                min=linear_ranges[forward, 0], max=linear_ranges[forward, 1])
        mode[small] = MODE_SMALL
        mode[reverse] = MODE_REVERSE
        mode[forward] = MODE_FORWARD
        return linear, yaw, mode

    if phase == "yaw":
        bucket = slot_ids % 10
        small = (bucket >= 2) & (bucket < 4)
        turning = bucket >= 4
        yaw[small] = sign[small] * torch.minimum(
            yaw_limit[small], yaw_limit[small].new_tensor(small_yaw_limit)
        )
        yaw[turning] = sign[turning] * yaw_limit[turning]
        mode[small] = MODE_SMALL
        mode[turning] = torch.where(sign[turning] < 0, MODE_REVERSE, MODE_FORWARD)
        return linear, yaw, mode

    # Combined phase: 10% zero, 20% pure translation, 20% pure yaw, 50%
    # simultaneous signed translation and yaw.  The caller controls the
    # command envelope through the configured stage limits.
    bucket = slot_ids % 10
    pure_translation = (bucket >= 1) & (bucket < 3)
    pure_yaw = (bucket >= 3) & (bucket < 5)
    combined = bucket >= 5
    linear[pure_translation] = sign[pure_translation] * lin_limit[pure_translation]
    blend = 0.25 + 0.50 * torch.rand(count, device=device, dtype=dtype)
    linear[combined] = sign[combined] * lin_limit[combined] * blend[combined]
    yaw[pure_yaw] = sign[pure_yaw] * yaw_limit[pure_yaw]
    yaw_sign = torch.where(slot_ids % 4 < 2, torch.ones_like(sign), -torch.ones_like(sign))
    yaw[combined] = (
        yaw_sign[combined]
        * yaw_limit[combined]
        * (1.0 - blend[combined])
    )
    mode[pure_translation] = torch.where(sign[pure_translation] < 0, MODE_REVERSE, MODE_FORWARD)
    mode[pure_yaw] = MODE_SMALL
    mode[combined] = MODE_FORWARD
    return linear, yaw, mode


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
