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
