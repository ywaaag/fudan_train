"""Serialize experiment evidence; no training or recipe selection."""
import json
from pathlib import Path

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
        "init_noise_std": float(train_cfg.policy.init_noise_std),
        "save_interval": int(train_cfg.runner.save_interval),
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

