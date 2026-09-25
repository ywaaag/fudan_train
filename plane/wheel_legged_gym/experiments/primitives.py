"""Shared recipe construction; never imports recipe selection or simulators."""
from __future__ import annotations

from .definitions import (
    EXPERIMENTS, METHOD_V1_PHASES, METHOD_V1_BASE_REWARD_SCALES, copy_definition,
)

def apply_method_randomization(env_cfg, level: int) -> None:
    """Configure creation-time Isaac Gym randomization for one relay level."""

    if level < 0 or level > 3:
        raise ValueError("method_v1 randomization level must be in [0, 3]")
    rand = env_cfg.domain_rand
    enabled = level > 0
    rand.randomize_friction = enabled
    rand.randomize_base_mass = enabled
    rand.randomize_inertia = enabled
    rand.randomize_base_com = enabled
    rand.randomize_Kp = enabled
    rand.randomize_Kd = enabled
    rand.randomize_motor_torque = enabled
    rand.push_robots = level >= 3
    if level == 0:
        return
    if level == 1:
        rand.friction_range = [0.95, 1.05]
        rand.added_mass_range = [-0.25, 0.25]
        rand.randomize_inertia_range = [0.98, 1.02]
        rand.rand_com_vec = [0.005, 0.005, 0.005]
        rand.randomize_Kp_range = [0.99, 1.01]
        rand.randomize_Kd_range = [0.99, 1.01]
        rand.randomize_motor_torque_range = [0.98, 1.02]
    else:
        rand.friction_range = [0.85, 1.15]
        rand.added_mass_range = [-0.5, 0.5]
        rand.randomize_inertia_range = [0.95, 1.05]
        rand.rand_com_vec = [0.01, 0.01, 0.01]
        rand.randomize_Kp_range = [0.97, 1.03]
        rand.randomize_Kd_range = [0.97, 1.03]
        rand.randomize_motor_torque_range = [0.95, 1.05]


def apply_legacy_experiment(env_cfg, name, train_cfg=None):
    if name is None:
        return {"name": "baseline", "description": "unmodified baseline"}
    normalized = name.upper()
    if normalized not in EXPERIMENTS:
        raise ValueError(
            f"unknown policy experiment {name!r}; expected one of {sorted(EXPERIMENTS)}"
        )
    experiment = copy_definition(EXPERIMENTS[normalized])
    zero, small, reverse, forward = experiment["fractions"]
    env_cfg.commands.sampling_strategy = "zero_reverse_mixture"
    env_cfg.commands.mixture_zero_fraction = zero
    env_cfg.commands.mixture_small_fraction = small
    env_cfg.commands.mixture_reverse_fraction = reverse
    env_cfg.commands.mixture_forward_fraction = forward
    env_cfg.rewards.scales.zero_base_velocity = experiment["zero_base_velocity_scale"]
    env_cfg.rewards.scales.zero_wheel_velocity = experiment["zero_wheel_velocity_scale"]
    env_cfg.rewards.scales.low_speed_tracking = experiment.get(
        "low_speed_tracking_scale", 0.0
    )
    env_cfg.rewards.scales.high_speed_tracking = experiment.get(
        "high_speed_tracking_scale", 0.0
    )
    env_cfg.rewards.scales.high_speed_yaw_tracking = experiment.get(
        "high_speed_yaw_tracking_scale", 0.0
    )
    env_cfg.rewards.scales.high_speed_yaw_penalty = experiment.get(
        "high_speed_yaw_penalty_scale", 0.0
    )
    env_cfg.rewards.scales.high_speed_slip = experiment.get(
        "high_speed_slip_scale", 0.0
    )
    env_cfg.rewards.unclipped_reward_names = tuple(
        experiment.get("unclipped_reward_names", ())
    )
    if "tracking_lin_vel_scale" in experiment:
        env_cfg.rewards.scales.tracking_lin_vel = float(
            experiment["tracking_lin_vel_scale"]
        )
    env_cfg.rewards.scales.zero_yaw_wheel_symmetry = experiment.get(
        "zero_yaw_wheel_symmetry_scale", 0.0
    )
    if train_cfg is not None:
        train_cfg.algorithm.symmetry_loss_coef = float(
            experiment.get("symmetry_loss_coef", 0.0)
        )
    env_cfg.commands.mixture_small_linear_anchors = experiment.get(
        "small_linear_anchors"
    )
    env_cfg.commands.mixture_small_yaw_anchors = experiment.get("small_yaw_anchors")
    env_cfg.commands.mixture_endpoint_anchor_fraction = float(
        experiment.get("endpoint_anchor_fraction", 0.0)
    )
    env_cfg.commands.mixture_endpoint_linear_anchors = experiment.get(
        "endpoint_linear_anchors"
    )
    if experiment.get("command_curriculum", False):
        env_cfg.commands.curriculum = True
        env_cfg.commands.curriculum_mode = "staged_performance"
        first_linear, first_yaw = env_cfg.commands.curriculum_stages[0]
        env_cfg.commands.ranges.lin_vel_x = [-first_linear, first_linear]
        env_cfg.commands.ranges.ang_vel_yaw = [-first_yaw, first_yaw]
        env_cfg.commands.curriculum_initial_stage = int(
            experiment.get("curriculum_initial_stage", 0)
        )
        env_cfg.commands.curriculum_required_passes = int(
            experiment.get(
                "curriculum_required_passes",
                env_cfg.commands.curriculum_required_passes,
            )
        )
        if "curriculum_stages" in experiment:
            env_cfg.commands.curriculum_stages = tuple(experiment["curriculum_stages"])
    if "low_speed_yaw_weight" in experiment:
        env_cfg.rewards.low_speed_yaw_weight = experiment["low_speed_yaw_weight"]
    optimizer = experiment.get("optimizer")
    if optimizer is not None:
        if train_cfg is None:
            raise ValueError(f"experiment {normalized} requires train_cfg")
        train_cfg.algorithm.learning_rate = optimizer["learning_rate"]
        train_cfg.algorithm.extra_learning_rate = optimizer["extra_learning_rate"]
        train_cfg.algorithm.schedule = optimizer["schedule"]
        if "entropy_coef" in optimizer:
            train_cfg.algorithm.entropy_coef = optimizer["entropy_coef"]
    if "save_interval" in experiment:
        if train_cfg is None:
            raise ValueError(f"experiment {normalized} requires train_cfg")
        train_cfg.runner.save_interval = experiment["save_interval"]
    return {"name": normalized, **experiment}


def apply_training_profile(
    env_cfg,
    train_cfg=None,
    *,
    profile: str = "method_v1",
    phase: str = "stand",
    level: int | None = None,
    stand_randomization_level=None,
) -> dict:
    """Apply the versioned staged method profile without changing contracts."""

    normalized = profile.lower()
    if normalized in {"h3", "h7"}:
        return apply_legacy_experiment(env_cfg, normalized, train_cfg)
    if normalized not in {"method_v1", "method"}:
        raise ValueError("unknown training profile; expected method_v1, H3, or H7")
    phase = str(phase).lower()
    if phase not in METHOD_V1_PHASES:
        raise ValueError(f"unknown method_v1 phase: {phase}")
    if train_cfg is None:
        raise ValueError("method_v1 requires train_cfg")

    phase_cfg = METHOD_V1_PHASES[phase]
    env_cfg.rewards.reward_pipeline = "normalized_v1"
    env_cfg.commands.sampling_strategy = "method_v1"
    env_cfg.commands.training_profile = "method_v1"
    env_cfg.commands.training_phase = phase
    env_cfg.commands.hold_command_until_reset = True
    env_cfg.commands.curriculum = False
    env_cfg.commands.curriculum_mode = "method_v1"
    env_cfg.commands.curriculum_stages = ()
    command_level = 0 if level is None else int(level)
    if command_level < 0:
        raise ValueError("method_v1 command level must be non-negative")
    env_cfg.commands.method_v1_level = command_level
    env_cfg.commands.method_v1_phase_config = dict(phase_cfg)

    if phase == "stand":
        linear_limit = yaw_limit = 0.0
    elif phase == "translate":
        levels = phase_cfg["linear_levels"]
        if command_level >= len(levels):
            raise ValueError("method_v1 translate command level is out of range")
        idx = command_level
        linear_limit, yaw_limit = float(levels[idx]), 0.0
    elif phase == "yaw":
        levels = phase_cfg["yaw_levels"]
        if command_level >= len(levels):
            raise ValueError("method_v1 yaw command level is out of range")
        idx = command_level
        linear_limit, yaw_limit = 0.0, float(levels[idx])
    else:
        levels = phase_cfg["combined_levels"]
        if command_level >= len(levels):
            raise ValueError("method_v1 combined command level is out of range")
        idx = command_level
        linear_limit, yaw_limit = map(float, levels[idx])

    env_cfg.commands.ranges.lin_vel_x = [-linear_limit, linear_limit]
    env_cfg.commands.ranges.ang_vel_yaw = [-yaw_limit, yaw_limit]
    env_cfg.commands.ranges.height = [0.40, 0.40]
    env_cfg.commands.mixture_zero_fraction = phase_cfg["fractions"][0]
    env_cfg.commands.mixture_small_fraction = phase_cfg["fractions"][1]
    env_cfg.commands.mixture_reverse_fraction = phase_cfg["fractions"][2]
    env_cfg.commands.mixture_forward_fraction = phase_cfg["fractions"][3]
    env_cfg.commands.mixture_small_linear_limit = min(0.10, linear_limit)
    env_cfg.commands.mixture_small_yaw_limit = min(0.10, yaw_limit)
    env_cfg.commands.mixture_endpoint_anchor_fraction = 0.50
    env_cfg.commands.mixture_endpoint_linear_anchors = (
        (-linear_limit, linear_limit) if linear_limit > 0.0 else None
    )
    env_cfg.commands.mixture_small_linear_anchors = (-0.10, -0.05, 0.05, 0.10)
    env_cfg.commands.mixture_small_yaw_anchors = (-0.05, 0.05)

    for name in dir(env_cfg.rewards.scales):
        if not name.startswith("_") and isinstance(getattr(env_cfg.rewards.scales, name), (int, float)):
            setattr(env_cfg.rewards.scales, name, 0.0)
    for name, value in METHOD_V1_BASE_REWARD_SCALES.items():
        setattr(env_cfg.rewards.scales, name, value)
    vx_coarse, vx_fine, vx_gap = phase_cfg["track_vx"]
    yaw_coarse, yaw_fine, yaw_gap = phase_cfg["track_yaw"]
    env_cfg.rewards.scales.track_vx_coarse = vx_coarse
    env_cfg.rewards.scales.track_vx_fine = vx_fine
    env_cfg.rewards.scales.track_vx_gap = vx_gap
    env_cfg.rewards.scales.track_yaw_coarse = yaw_coarse
    env_cfg.rewards.scales.track_yaw_fine = yaw_fine
    env_cfg.rewards.scales.track_yaw_gap = yaw_gap
    env_cfg.rewards.scales.stand_still = phase_cfg["stand_still_scale"]
    if phase == "stand":
        # Replace all-DOF pose cost, including accumulated wheel rotation,
        # with bilateral leg geometry; keep velocity/contact costs active.
        env_cfg.rewards.scales.stand_still = 0.0
        env_cfg.rewards.scales.stand_bilateral_geometry = -0.3
        # Standing is evaluated with deterministic actor means.  Give wheel
        # speed a stronger zero-command penalty so the mean policy cannot
        # rely on exploration noise to cancel residual wheel motion.
        env_cfg.rewards.scales.zero_wheel_velocity = -3.0
    env_cfg.rewards.scales.tracking_lin_vel = 0.0
    env_cfg.rewards.scales.tracking_ang_vel = 0.0
    env_cfg.rewards.scales.high_speed_tracking = 0.0
    env_cfg.rewards.scales.high_speed_yaw_tracking = 0.0
    env_cfg.rewards.scales.high_speed_yaw_penalty = 0.0
    env_cfg.rewards.scales.high_speed_slip = 0.0
    env_cfg.rewards.scales.nominal_state = 0.0
    env_cfg.rewards.scales.base_height = 0.0
    env_cfg.rewards.scales.termination = 0.0
    env_cfg.rewards.scales.method_termination = -5.0
    env_cfg.rewards.scales.zero_yaw_wheel_symmetry = 0.0
    env_cfg.asset.penalize_contacts_on = ["base_link", "leg_0_link", "leg_1_link"]
    env_cfg.asset.terminate_after_contacts_on = ["base_link", "leg_0_link", "leg_1_link"]
    env_cfg.rewards.only_positive_rewards = False
    randomization_level = int(phase_cfg["randomization_level"])
    if phase == "stand":
        # Keep the validated deterministic baseline unchanged unless a new
        # run explicitly opts into robustness training.
        if stand_randomization_level is not None:
            randomization_level = int(stand_randomization_level)
        if not 0 <= randomization_level <= 3:
            raise ValueError("FUDAN_STAND_RANDOMIZATION_LEVEL must be in [0, 3]")
    env_cfg.domain_rand_level = randomization_level
    apply_method_randomization(env_cfg, env_cfg.domain_rand_level)

    train_cfg.algorithm.learning_rate = 1.0e-4
    train_cfg.algorithm.extra_learning_rate = 1.0e-5
    train_cfg.algorithm.schedule = "adaptive"
    train_cfg.algorithm.entropy_coef = 0.001 if phase == "stand" else 0.005
    # Apply mirror-equivariant policy regularization to the active method_v1
    # line.  The transform in policy_symmetry.py swaps the parity-tree left /
    # right channels and mirrors body-frame signs; it does not force raw live
    # actions to be equal.
    train_cfg.algorithm.symmetry_loss_coef = 0.01
    train_cfg.runner.save_interval = 100
    active_scales = {
        name: getattr(env_cfg.rewards.scales, name)
        for name in dir(env_cfg.rewards.scales)
        if not name.startswith("_")
        and isinstance(getattr(env_cfg.rewards.scales, name), (int, float))
    }
    return {
        "name": "method_v1",
        "profile": "method_v1",
        "phase": phase,
        "level": int(env_cfg.commands.method_v1_level),
        "reward_version": "normalized_v1",
        "command_limits": (linear_limit, yaw_limit),
        "reward_scales": active_scales,
        "optimizer": {
            "learning_rate": train_cfg.algorithm.learning_rate,
            "extra_learning_rate": 1.0e-5,
            "schedule": train_cfg.algorithm.schedule,
            "entropy_coef": train_cfg.algorithm.entropy_coef,
        },
    }
