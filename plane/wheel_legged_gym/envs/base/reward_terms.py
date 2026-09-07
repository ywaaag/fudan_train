"""Pure Torch reward primitives for the versioned wheel-leg method profile.

The helpers in this module deliberately do not import Isaac Gym.  Keeping the
math here makes the reward ledger unit-testable without constructing a physics
simulation and prevents command-axis/indexing regressions from being hidden in
the environment implementation.
"""

from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor


# The Fudan tree policy contract is [linear velocity, yaw rate, base height].
CMD_VX = 0
CMD_YAW = 1
CMD_HEIGHT = 2


def _huber(error: Tensor, delta: float = 1.0) -> Tensor:
    """Element-wise smooth-L1 cost with a configurable transition point."""

    delta_tensor = error.new_tensor(float(delta))
    absolute = torch.abs(error)
    quadratic = 0.5 * torch.square(absolute) / delta_tensor
    linear = absolute - 0.5 * delta_tensor
    return torch.where(absolute <= delta_tensor, quadratic, linear)


def capped_tracking_terms(
    error: Tensor,
    cap: float,
    *,
    coarse_sigma: float = 1.0,
    fine_sigma: float = 0.25,
    gap_delta: float = 1.0,
    gap_clip: float = 4.0,
) -> Dict[str, Tensor]:
    """Return coarse/fine tracking rewards and a non-vanishing gap cost.

    The exponential terms operate on a bounded normalized error so they remain
    numerically useful at high command speeds.  The Huber gap term uses the
    unbounded (but safely clipped) normalized error and therefore continues to
    provide a gradient outside the cap.
    """

    if cap <= 0.0:
        raise ValueError("tracking error cap must be positive")
    if coarse_sigma <= 0.0 or fine_sigma <= 0.0:
        raise ValueError("tracking sigmas must be positive")

    normalized = error / float(cap)
    bounded = torch.clamp(normalized, -1.0, 1.0)
    coarse = torch.exp(-torch.abs(bounded) / float(coarse_sigma))
    fine = torch.exp(-torch.square(bounded / float(fine_sigma)))
    gap = _huber(torch.clamp(torch.abs(normalized), max=float(gap_clip)), gap_delta)
    return {"coarse": coarse, "fine": fine, "gap": gap}


def smooth_gate(error: Tensor, tolerance: float) -> Tensor:
    """A smooth [0, 1] gate used only for task rewards."""

    if tolerance <= 0.0:
        raise ValueError("gate tolerance must be positive")
    return torch.exp(-torch.square(error / float(tolerance)))


def normalized_huber(
    value: Tensor,
    scale: float,
    *,
    delta: float = 1.0,
    clip: float = 1.0,
) -> Tensor:
    """Dimensionless bounded Huber cost for safety/regularization terms."""

    if scale <= 0.0 or clip <= 0.0:
        raise ValueError("normalization scale and clip must be positive")
    return torch.clamp(_huber(value / float(scale), delta), min=0.0, max=float(clip))


def wheel_rolling_terms(
    base_vx: Tensor,
    base_yaw_rate: Tensor,
    wheel_ang_vel: Tensor,
    wheel_y: Tensor,
    wheel_radius: float,
    contact: Tensor,
    *,
    slip_scale: float = 0.5,
    air_spin_scale: float = 1.0,
) -> Dict[str, Tensor]:
    """Compute per-wheel rolling residual and contact-conditioned costs.

    The URDF wheel axes point along ``(0, -1, 0)``, therefore positive forward
    rolling speed is ``-omega * radius``.  For a body-frame yaw rate ``wz`` and
    wheel lateral position ``y``, the expected x velocity is ``vx - wz*y``.
    ``contact`` is a float/bool tensor shaped ``(N, 2)``.
    """

    if wheel_ang_vel.ndim != 2 or wheel_ang_vel.shape[-1] != 2:
        raise ValueError("wheel_ang_vel must have shape (N, 2)")
    if contact.shape != wheel_ang_vel.shape:
        raise ValueError("contact must have the same shape as wheel_ang_vel")
    if wheel_radius <= 0.0 or slip_scale <= 0.0 or air_spin_scale <= 0.0:
        raise ValueError("wheel radius and residual scales must be positive")

    if wheel_y.ndim == 1:
        if wheel_y.shape[0] != 2:
            raise ValueError("wheel_y must have two wheel offsets")
        wheel_y = wheel_y.unsqueeze(0)
    if wheel_y.shape[-1] != 2:
        raise ValueError("wheel_y must have two wheel offsets")

    signed_rolling_speed = -wheel_ang_vel * float(wheel_radius)
    expected_speed = base_vx.unsqueeze(-1) - base_yaw_rate.unsqueeze(-1) * wheel_y
    residual = signed_rolling_speed - expected_speed
    contact_float = contact.to(dtype=wheel_ang_vel.dtype)
    airborne = 1.0 - contact_float

    residual_cost = normalized_huber(residual, slip_scale, clip=1.0)
    spin_cost = normalized_huber(
        signed_rolling_speed, air_spin_scale, clip=1.0
    )
    contact_count = torch.clamp(contact_float.sum(dim=-1), min=1.0)
    airborne_count = torch.clamp(airborne.sum(dim=-1), min=1.0)
    contact_slip = (residual_cost * contact_float).sum(dim=-1) / contact_count
    airborne_spin = (spin_cost * airborne).sum(dim=-1) / airborne_count
    residual_rms = torch.sqrt(torch.mean(torch.square(residual), dim=-1) + 1.0e-12)
    return {
        "signed_rolling_speed": signed_rolling_speed,
        "expected_speed": expected_speed,
        "residual": residual,
        "contact_slip": contact_slip,
        "airborne_spin": airborne_spin,
        "residual_rms": residual_rms,
    }
