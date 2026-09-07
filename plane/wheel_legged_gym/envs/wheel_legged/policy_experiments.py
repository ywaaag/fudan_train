"""Isolated policy-training experiments for zero/reverse velocity tracking."""

from __future__ import annotations

import json
from pathlib import Path


EXPERIMENTS = {
    # Retained historical baseline for reproducible recovery from model_15800.
    "H3": {
        "description": "validated low-speed recovery baseline",
        "zero_base_velocity_scale": -8.0,
        "zero_wheel_velocity_scale": -5.0,
        "low_speed_tracking_scale": -2.0,
        "zero_yaw_wheel_symmetry_scale": 0.0,
        "symmetry_loss_coef": 0.005,
        "fractions": (0.35, 0.25, 0.20, 0.20),
        "small_linear_anchors": (-0.10, -0.05, 0.05, 0.10),
        "small_yaw_anchors": (-0.05, 0.05),
        "endpoint_anchor_fraction": 0.50,
        "endpoint_linear_anchors": (-3.50, -3.25, -3.00, 3.00, 3.25, 3.50),
        "command_curriculum": True,
        "optimizer": {
            "learning_rate": 2.0e-6,
            "extra_learning_rate": 0.0,
            "schedule": "fixed",
            "entropy_coef": 0.0002,
        },
        "save_interval": 100,
    },
    # Current focused experiment: stabilize a single 3.0 m/s operating point.
    "H7": {
        "description": "single-stage 3.0 m/s stabilization with slip and yaw shaping",
        "zero_base_velocity_scale": -5.0,
        "zero_wheel_velocity_scale": -3.0,
        "high_speed_tracking_scale": 1.0,
        "high_speed_yaw_tracking_scale": 0.0,
        "high_speed_yaw_penalty_scale": -2.0,
        "high_speed_slip_scale": -7.0,
        "unclipped_reward_names": ("high_speed_slip", "high_speed_yaw_penalty"),
        "low_speed_tracking_scale": -1.0,
        "tracking_lin_vel_scale": 1.0,
        "zero_yaw_wheel_symmetry_scale": 0.0,
        "symmetry_loss_coef": 0.003,
        "fractions": (0.25, 0.25, 0.25, 0.25),
        "small_linear_anchors": (-0.10, -0.05, 0.05, 0.10),
        "small_yaw_anchors": (-0.05, 0.05),
        "endpoint_anchor_fraction": 0.50,
        "endpoint_linear_anchors": (-3.00, 3.00),
        "command_curriculum": True,
        "curriculum_stages": ((3.0, 3.0),),
        "curriculum_initial_stage": 0,
        "curriculum_required_passes": 5,
        "optimizer": {
            "learning_rate": 3.0e-6,
            "extra_learning_rate": 0.0,
            "schedule": "fixed",
            "entropy_coef": 0.0002,
        },
        "save_interval": 100,
    },
}


# The active replacement for the old H-series.  H3/H7 remain above as frozen
# compatibility profiles so historical checkpoints can still be inspected.
METHOD_V1_PHASES = {
    "stand": {
        "linear_limit": 0.0,
        "yaw_limit": 0.0,
        "randomization_level": 0,
        "fractions": (1.0, 0.0, 0.0, 0.0),
        "track_vx": (0.50, 0.25, -0.125),
        "track_yaw": (0.50, 0.25, -0.125),
        "stand_still_scale": -0.20,
    },
    "translate": {
        "linear_levels": (0.5, 1.0, 2.0, 3.0, 4.0),
        "yaw_limit": 0.0,
        "randomization_level": 0,
        "fractions": (0.20, 0.20, 0.30, 0.30),
        "track_vx": (1.00, 0.50, -0.25),
        "track_yaw": (0.50, 0.25, -0.15),
        "stand_still_scale": 0.0,
    },
    "yaw": {
        "linear_limit": 0.0,
        "yaw_levels": (0.5, 1.0, 2.0, 3.0, 4.0),
        "randomization_level": 1,
        "fractions": (0.20, 0.20, 0.30, 0.30),
        "track_vx": (0.25, 0.10, -0.10),
        "track_yaw": (1.00, 0.50, -0.25),
        "stand_still_scale": 0.0,
    },
    "combined": {
        "combined_levels": ((0.5, 0.5), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0), (4.0, 4.0)),
        "randomization_level": 2,
        "fractions": (0.10, 0.0, 0.20, 0.70),
        "track_vx": (1.00, 0.50, -0.25),
        "track_yaw": (1.00, 0.50, -0.25),
        "stand_still_scale": 0.0,
    },
}

METHOD_V1_BASE_REWARD_SCALES = {
    "track_vx_coarse": 1.0,
    "track_vx_fine": 0.5,
    "track_vx_gap": -0.25,
    "track_yaw_coarse": 0.75,
    "track_yaw_fine": 0.35,
    "track_yaw_gap": -0.15,
    "orientation": -2.0,
    "height_cost": -1.0,
    "lin_vel_z": -0.2,
    "ang_vel_xy": -0.2,
    "lateral_velocity": -0.2,
    "wheel_slip": -1.5,
    "airborne_wheel_spin": -0.75,
    "wheel_contact_loss": -1.0,
    "forbidden_contact": -3.0,
    "torque_cost": -0.01,
    "power_cost": -0.005,
    "action_rate": -0.01,
    "action_second_diff": -0.005,
    "zero_base_velocity": -1.0,
    "zero_wheel_velocity": -1.0,
    "method_termination": -5.0,
}


def _apply_method_randomization(env_cfg, level: int) -> None:
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
"""Legacy experiment definitions retained only in repository history.

The active CLI surface intentionally exposes only H3 and H7 below. Historical
runs and checkpoints remain untouched on disk.
        "description": "command mixture plus exact-zero body velocity reward",
        "zero_base_velocity_scale": -5.0,
        "zero_wheel_velocity_scale": 0.0,
        "fractions": (0.20, 0.20, 0.30, 0.30),
    },
    "B": {
        "description": "A plus exact-zero wheel velocity reward",
        "zero_base_velocity_scale": -5.0,
        "zero_wheel_velocity_scale": -0.10,
        "fractions": (0.20, 0.20, 0.30, 0.30),
    },
    "C": {
        "description": "B plus increased reverse sampling",
        "zero_base_velocity_scale": -5.0,
        "zero_wheel_velocity_scale": -0.10,
        "fractions": (0.20, 0.20, 0.40, 0.20),
    },
    "R": {
        "description": "A refinement with fixed low actor and encoder learning rates",
        "zero_base_velocity_scale": -5.0,
        "zero_wheel_velocity_scale": 0.0,
        "fractions": (0.20, 0.20, 0.30, 0.30),
        "optimizer": {
            "learning_rate": 1.0e-5,
            "extra_learning_rate": 1.0e-4,
            "schedule": "fixed",
        },
        "save_interval": 10,
    },
    "L": {
        "description": "low-speed anchor refinement with a frozen velocity encoder",
        "zero_base_velocity_scale": -1.0,
        "zero_wheel_velocity_scale": 0.0,
        "low_speed_tracking_scale": -2.0,
        "fractions": (0.20, 0.40, 0.20, 0.20),
        "small_linear_anchors": (-0.10, -0.05, 0.05, 0.10),
        "small_yaw_anchors": (-0.05, 0.05),
        "optimizer": {
            "learning_rate": 1.0e-5,
            "extra_learning_rate": 0.0,
            "schedule": "fixed",
            "entropy_coef": 0.001,
        },
        "save_interval": 10,
    },
    "Y": {
        "description": "L refinement with strong signed low-speed yaw tracking",
        "zero_base_velocity_scale": -1.0,
        "zero_wheel_velocity_scale": 0.0,
        "low_speed_tracking_scale": -2.0,
        "low_speed_yaw_weight": 10.0,
        "fractions": (0.20, 0.40, 0.20, 0.20),
        "small_linear_anchors": (-0.10, -0.05, 0.05, 0.10),
        "small_yaw_anchors": (-0.05, 0.05),
        "optimizer": {
            "learning_rate": 5.0e-6,
            "extra_learning_rate": 0.0,
            "schedule": "fixed",
            "entropy_coef": 0.001,
        },
        "save_interval": 5,
    },
    "S": {
        "description": "L refinement that removes zero-yaw wheel-action asymmetry",
        "zero_base_velocity_scale": -1.0,
        "zero_wheel_velocity_scale": 0.0,
        "low_speed_tracking_scale": -2.0,
        "zero_yaw_wheel_symmetry_scale": -2.0,
        "fractions": (0.20, 0.40, 0.20, 0.20),
        "small_linear_anchors": (-0.10, -0.05, 0.05, 0.10),
        "small_yaw_anchors": (-0.05, 0.05),
        "optimizer": {
            "learning_rate": 5.0e-6,
            "extra_learning_rate": 0.0,
            "schedule": "fixed",
            "entropy_coef": 0.001,
        },
        "save_interval": 5,
    },
    "T": {
        "description": "strong zero-yaw wheel-action symmetry refinement",
        "zero_base_velocity_scale": -1.0,
        "zero_wheel_velocity_scale": 0.0,
        "low_speed_tracking_scale": -2.0,
        "zero_yaw_wheel_symmetry_scale": -20.0,
        "fractions": (0.20, 0.40, 0.20, 0.20),
        "small_linear_anchors": (-0.10, -0.05, 0.05, 0.10),
        "small_yaw_anchors": (-0.05, 0.05),
        "optimizer": {
            "learning_rate": 5.0e-6,
            "extra_learning_rate": 0.0,
            "schedule": "fixed",
            "entropy_coef": 0.001,
        },
        "save_interval": 5,
    },
    "H": {
        "description": "performance-gated curriculum with mirror-equivariant policy regularization",
        "zero_base_velocity_scale": -20.0,
        "zero_wheel_velocity_scale": -20.0,
        "low_speed_tracking_scale": -2.0,
        # Do not force equal wheel actions in the same physical state. That
        # removes yaw/roll correction authority and moves compensation into
        # asymmetric leg actions. Use mirrored-state equivariance instead.
        "zero_yaw_wheel_symmetry_scale": -0.5,
        "symmetry_loss_coef": 0.01,
        "fractions": (0.20, 0.30, 0.25, 0.25),
        "small_linear_anchors": (-0.10, -0.05, 0.05, 0.10),
        "small_yaw_anchors": (-0.05, 0.05),
        "command_curriculum": True,
        "optimizer": {
            "learning_rate": 1.0e-5,
            "extra_learning_rate": 0.0,
            "schedule": "fixed",
            "entropy_coef": 0.001,
        },
        "save_interval": 100,
    },
    "H2": {
        "description": "zero/small command emphasis plus explicit +/-3.5 m/s endpoint anchors",
        "zero_base_velocity_scale": -15.0,
        "zero_wheel_velocity_scale": -10.0,
        "low_speed_tracking_scale": -3.0,
        "zero_yaw_wheel_symmetry_scale": 0.0,
        "symmetry_loss_coef": 0.01,
        "fractions": (0.30, 0.30, 0.20, 0.20),
        "small_linear_anchors": (-0.10, -0.05, 0.05, 0.10),
        "small_yaw_anchors": (-0.05, 0.05),
        "endpoint_anchor_fraction": 0.60,
        "endpoint_linear_anchors": (-3.50, -3.25, -3.00, -2.50, 2.50, 3.00, 3.25, 3.50),
        "command_curriculum": True,
        "optimizer": {
            "learning_rate": 1.0e-5,
            "extra_learning_rate": 0.0,
            "schedule": "fixed",
            "entropy_coef": 0.001,
        },
        "save_interval": 100,
    },
    "H3": {
        "description": "stable anchored curriculum from H2 best checkpoint with conservative updates",
        "zero_base_velocity_scale": -8.0,
        "zero_wheel_velocity_scale": -5.0,
        "low_speed_tracking_scale": -2.0,
        "zero_yaw_wheel_symmetry_scale": 0.0,
        "symmetry_loss_coef": 0.005,
        "fractions": (0.35, 0.25, 0.20, 0.20),
        "small_linear_anchors": (-0.10, -0.05, 0.05, 0.10),
        "small_yaw_anchors": (-0.05, 0.05),
        "endpoint_anchor_fraction": 0.50,
        "endpoint_linear_anchors": (-3.50, -3.25, -3.00, 3.00, 3.25, 3.50),
        "command_curriculum": True,
        "optimizer": {
            "learning_rate": 2.0e-6,
            "extra_learning_rate": 0.0,
            "schedule": "fixed",
            "entropy_coef": 0.0002,
        },
        "save_interval": 100,
    },
    "H4": {
        "description": "recover the H3 3.5 m/s plateau, then expose an explicit 4.0 m/s stage",
        "zero_base_velocity_scale": -8.0,
        "zero_wheel_velocity_scale": -5.0,
        "low_speed_tracking_scale": -2.0,
        "zero_yaw_wheel_symmetry_scale": 0.0,
        "symmetry_loss_coef": 0.005,
        "fractions": (0.30, 0.25, 0.225, 0.225),
        "small_linear_anchors": (-0.10, -0.05, 0.05, 0.10),
        "small_yaw_anchors": (-0.05, 0.05),
        "endpoint_anchor_fraction": 0.60,
        "endpoint_linear_anchors": (-4.00, -3.75, -3.50, -3.25, -3.00,
                                     3.00, 3.25, 3.50, 3.75, 4.00),
        "command_curriculum": True,
        "optimizer": {
            "learning_rate": 5.0e-6,
            "extra_learning_rate": 0.0,
            "schedule": "fixed",
            "entropy_coef": 0.0003,
        },
        "save_interval": 100,
    },
    "H5": {
        "description": "high-speed bridge from 3.5 to 4.0 m/s with explicit transition stage",
        "zero_base_velocity_scale": -3.0,
        "zero_wheel_velocity_scale": -1.0,
        "high_speed_tracking_scale": 1.0,
        "low_speed_tracking_scale": -1.0,
        "zero_yaw_wheel_symmetry_scale": 0.0,
        "symmetry_loss_coef": 0.003,
        "fractions": (0.20, 0.15, 0.325, 0.325),
        "small_linear_anchors": (-0.10, -0.05, 0.05, 0.10),
        "small_yaw_anchors": (-0.05, 0.05),
        "endpoint_anchor_fraction": 0.60,
        "endpoint_linear_anchors": (-4.00, -3.75, -3.50, 3.50, 3.75, 4.00),
        "command_curriculum": True,
        "curriculum_initial_stage": 3,
        "curriculum_required_passes": 5,
        "optimizer": {
            "learning_rate": 5.0e-6,
            "extra_learning_rate": 0.0,
            "schedule": "fixed",
            "entropy_coef": 0.0003,
        },
        "save_interval": 100,
    },
    "H6": {
        "description": "narrow 3.0 to 3.5 m/s bridge with yaw and wheel-slip shaping",
        "zero_base_velocity_scale": -5.0,
        "zero_wheel_velocity_scale": -3.0,
        "high_speed_tracking_scale": 1.0,
        "high_speed_yaw_tracking_scale": 0.0,
        "high_speed_yaw_penalty_scale": -0.25,
        "high_speed_slip_scale": -1.5,
        "low_speed_tracking_scale": -1.0,
        "tracking_lin_vel_scale": 0.75,
        "zero_yaw_wheel_symmetry_scale": 0.0,
        "symmetry_loss_coef": 0.003,
        "fractions": (0.25, 0.25, 0.25, 0.25),
        "small_linear_anchors": (-0.10, -0.05, 0.05, 0.10),
        "small_yaw_anchors": (-0.05, 0.05),
        "endpoint_anchor_fraction": 0.50,
        "endpoint_linear_anchors": (-3.50, -3.20, -3.00, 3.00, 3.20, 3.50),
        "command_curriculum": True,
        "curriculum_stages": ((3.0, 3.0), (3.2, 3.2), (3.5, 3.5)),
        "curriculum_initial_stage": 0,
        "curriculum_required_passes": 5,
        "optimizer": {
            "learning_rate": 3.0e-6,
            "extra_learning_rate": 0.0,
            "schedule": "fixed",
            "entropy_coef": 0.0002,
        },
        "save_interval": 100,
    },
    "H7": {
        "description": "H6 heavy slip and yaw penalties to break high-speed reward exploitation",
        "zero_base_velocity_scale": -5.0,
        "zero_wheel_velocity_scale": -3.0,
        "high_speed_tracking_scale": 1.0,
        "high_speed_yaw_tracking_scale": 0.0,
        "high_speed_yaw_penalty_scale": -2.0,
        "high_speed_slip_scale": -10.0,
        "low_speed_tracking_scale": -1.0,
        "tracking_lin_vel_scale": 0.75,
        "zero_yaw_wheel_symmetry_scale": 0.0,
        "symmetry_loss_coef": 0.003,
        "fractions": (0.25, 0.25, 0.25, 0.25),
        "small_linear_anchors": (-0.10, -0.05, 0.05, 0.10),
        "small_yaw_anchors": (-0.05, 0.05),
        "endpoint_anchor_fraction": 0.50,
        "endpoint_linear_anchors": (-3.50, -3.20, -3.00, 3.00, 3.20, 3.50),
        "command_curriculum": True,
        "curriculum_stages": ((3.0, 3.0), (3.2, 3.2), (3.5, 3.5)),
        "curriculum_initial_stage": 0,
        "curriculum_required_passes": 5,
        "optimizer": {
            "learning_rate": 3.0e-6,
            "extra_learning_rate": 0.0,
            "schedule": "fixed",
            "entropy_coef": 0.0002,
        },
        "save_interval": 100,
    },
}
"""


def apply_policy_experiment(env_cfg, name: str | None, train_cfg=None) -> dict:
    if name is None:
        return {"name": "baseline", "description": "unmodified baseline"}
    if str(name).lower() in {"method_v1", "method"}:
        return apply_training_profile(
            env_cfg,
            train_cfg,
            profile="method_v1",
            phase=getattr(env_cfg.commands, "training_phase", "stand"),
            level=getattr(env_cfg.commands, "method_v1_level", 0),
        )
    normalized = name.upper()
    if normalized not in EXPERIMENTS:
        raise ValueError(
            f"unknown policy experiment {name!r}; expected one of {sorted(EXPERIMENTS)}"
        )
    experiment = EXPERIMENTS[normalized]
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
) -> dict:
    """Apply the versioned staged method profile without changing contracts."""

    normalized = profile.lower()
    if normalized in {"h3", "h7"}:
        return apply_policy_experiment(env_cfg, normalized, train_cfg)
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
    env_cfg.domain_rand_level = int(phase_cfg["randomization_level"])
    _apply_method_randomization(env_cfg, env_cfg.domain_rand_level)

    train_cfg.algorithm.learning_rate = 1.0e-4
    train_cfg.algorithm.extra_learning_rate = 1.0e-5
    train_cfg.algorithm.schedule = "adaptive"
    train_cfg.algorithm.entropy_coef = 0.005
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
            "learning_rate": 1.0e-4,
            "extra_learning_rate": 1.0e-5,
            "schedule": "adaptive",
            "entropy_coef": 0.005,
        },
    }


def enforce_optimizer_overrides(runner, manifest: dict) -> None:
    optimizer = manifest.get("optimizer")
    if optimizer is None:
        return
    actor_lr = optimizer["learning_rate"]
    runner.alg.learning_rate = actor_lr
    runner.alg.schedule = optimizer["schedule"]
    if "entropy_coef" in optimizer:
        runner.alg.entropy_coef = optimizer["entropy_coef"]
    for group in runner.alg.optimizer.param_groups:
        group["lr"] = actor_lr
    if runner.alg.extra_optimizer is not None:
        for group in runner.alg.extra_optimizer.param_groups:
            group["lr"] = optimizer["extra_learning_rate"]


def write_experiment_manifest(path: Path, manifest: dict, env_cfg, train_cfg, args) -> None:
    payload = {
        **manifest,
        "seed": int(env_cfg.seed),
        "num_envs": int(env_cfg.env.num_envs),
        "max_iterations_this_run": int(train_cfg.runner.max_iterations),
        "resume": bool(train_cfg.runner.resume),
        "load_run": train_cfg.runner.load_run,
        "checkpoint": int(train_cfg.runner.checkpoint),
        "run_name": train_cfg.runner.run_name,
        "sampling_strategy": env_cfg.commands.sampling_strategy,
        "reward_pipeline": getattr(env_cfg.rewards, "reward_pipeline", "legacy_v0"),
        "training_profile": getattr(env_cfg.commands, "training_profile", "legacy"),
        "training_phase": getattr(env_cfg.commands, "training_phase", "legacy"),
        "command_level": int(getattr(env_cfg.commands, "method_v1_level", 0)),
        "randomization_level": int(getattr(env_cfg, "domain_rand_level", 0)),
        "reward_scales": {
            "zero_base_velocity": env_cfg.rewards.scales.zero_base_velocity,
            "zero_wheel_velocity": env_cfg.rewards.scales.zero_wheel_velocity,
            "low_speed_tracking": env_cfg.rewards.scales.low_speed_tracking,
            "high_speed_tracking": env_cfg.rewards.scales.high_speed_tracking,
            "high_speed_yaw_tracking": env_cfg.rewards.scales.high_speed_yaw_tracking,
            "high_speed_yaw_penalty": env_cfg.rewards.scales.high_speed_yaw_penalty,
            "high_speed_slip": env_cfg.rewards.scales.high_speed_slip,
            "zero_yaw_wheel_symmetry": (
                env_cfg.rewards.scales.zero_yaw_wheel_symmetry
            ),
            "tracking_lin_vel": env_cfg.rewards.scales.tracking_lin_vel,
            "tracking_ang_vel": env_cfg.rewards.scales.tracking_ang_vel,
            **{
                name: getattr(env_cfg.rewards.scales, name)
                for name in dir(env_cfg.rewards.scales)
                if not name.startswith("_")
                and isinstance(getattr(env_cfg.rewards.scales, name), (int, float))
            },
        },
        "command_fractions": {
            "zero": env_cfg.commands.mixture_zero_fraction,
            "small": env_cfg.commands.mixture_small_fraction,
            "reverse": env_cfg.commands.mixture_reverse_fraction,
            "forward": env_cfg.commands.mixture_forward_fraction,
        },
        "command_endpoint_anchor_fraction": float(
            env_cfg.commands.mixture_endpoint_anchor_fraction
        ),
        "command_endpoint_linear_anchors": env_cfg.commands.mixture_endpoint_linear_anchors,
        "command_curriculum": {
            "enabled": bool(env_cfg.commands.curriculum),
            "mode": env_cfg.commands.curriculum_mode,
            "stages": env_cfg.commands.curriculum_stages,
            "required_passes": env_cfg.commands.curriculum_required_passes,
        },
        "argv_policy_experiment": getattr(args, "policy_experiment", None),
        "optimizer": manifest.get("optimizer"),
        "symmetry_loss_coef": float(
            getattr(train_cfg.algorithm, "symmetry_loss_coef", 0.0)
        ),
        "save_interval": int(train_cfg.runner.save_interval),
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
