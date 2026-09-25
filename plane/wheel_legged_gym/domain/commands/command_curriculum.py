"""Performance-gated command curriculum helpers."""

from __future__ import annotations


def validate_curriculum_stages(stages) -> tuple[tuple[float, float], ...]:
    normalized = tuple((float(linear), float(yaw)) for linear, yaw in stages)
    if not normalized:
        raise ValueError("command curriculum needs at least one stage")
    previous_linear = 0.0
    previous_yaw = 0.0
    for linear, yaw in normalized:
        if linear <= 0.0 or yaw <= 0.0:
            raise ValueError("curriculum command limits must be positive")
        if linear < previous_linear or yaw < previous_yaw:
            raise ValueError("curriculum stages must be non-decreasing")
        previous_linear, previous_yaw = linear, yaw
    return normalized


def evaluate_curriculum_window(stats: dict, cfg) -> dict:
    """Return normalized metrics and whether one promotion window passed."""

    def ratio(numerator: float, denominator: float) -> float:
        return float(numerator) / max(abs(float(denominator)), 1.0e-9)

    episodes = int(stats["episodes"])
    survival = ratio(stats["timeouts"], episodes)
    zero_abs_vx = ratio(stats["zero_abs_vx_sum"], stats["zero_count"])
    zero_abs_yaw = ratio(stats["zero_abs_yaw_sum"], stats["zero_count"])
    reverse_relative_error = ratio(
        stats["reverse_error_sum"], stats["reverse_command_abs_sum"]
    )
    forward_relative_error = ratio(
        stats["forward_error_sum"], stats["forward_command_abs_sum"]
    )
    yaw_relative_error = ratio(stats["yaw_error_sum"], stats["yaw_command_abs_sum"])
    enough_samples = (
        episodes >= int(cfg.curriculum_min_episodes)
        and stats["zero_count"] > 0
        and stats["reverse_command_abs_sum"] > 0
        and stats["forward_command_abs_sum"] > 0
        and stats["yaw_command_abs_sum"] > 0
    )
    passed = (
        enough_samples
        and survival >= float(cfg.curriculum_min_survival)
        and zero_abs_vx <= float(cfg.curriculum_max_zero_vx)
        and zero_abs_yaw <= float(cfg.curriculum_max_zero_yaw)
        and reverse_relative_error <= float(cfg.curriculum_max_linear_relative_error)
        and forward_relative_error <= float(cfg.curriculum_max_linear_relative_error)
        and yaw_relative_error <= float(cfg.curriculum_max_yaw_relative_error)
    )
    return {
        "passed": bool(passed),
        "enough_samples": bool(enough_samples),
        "episodes": episodes,
        "survival_fraction": survival,
        "zero_abs_vx": zero_abs_vx,
        "zero_abs_yaw": zero_abs_yaw,
        "reverse_relative_error": reverse_relative_error,
        "forward_relative_error": forward_relative_error,
        "yaw_relative_error": yaw_relative_error,
    }
