"""Explicit recipe selection; concrete recipes depend only on primitives."""
from __future__ import annotations
from .primitives import apply_training_profile, apply_legacy_experiment

def apply_policy_experiment(env_cfg, name: str | None, train_cfg=None, *, spec=None,
                            stand_randomization_level=None) -> dict:
    if str(name).upper() == 'TURN_ENVELOPE':
        from wheel_legged_gym.experiments.recipes.turn_envelope import apply_turn_envelope
        return apply_turn_envelope(env_cfg, train_cfg, spec=spec)
    if str(name).upper() == 'HEIGHT_COURSE':
        from wheel_legged_gym.experiments.recipes.height_course import apply_height_course
        return apply_height_course(env_cfg, train_cfg, spec=spec)
    if str(name).upper() == 'MOTION_GOAL':
        from wheel_legged_gym.experiments.recipes.motion_goal import apply_motion_goal
        return apply_motion_goal(env_cfg, train_cfg, spec=spec)
    if str(name).upper() == 'LEGACY_YAW':
        from wheel_legged_gym.experiments.recipes.legacy_yaw import apply_legacy_yaw
        return apply_legacy_yaw(env_cfg, train_cfg)
    if str(name).upper() == 'LEGACY_SPEED2_STOP':
        from wheel_legged_gym.experiments.recipes.legacy_speed2_stop import apply_legacy_speed2_stop
        return apply_legacy_speed2_stop(env_cfg, train_cfg)
    if str(name).upper() == 'LEGACY_SPEED2':
        from wheel_legged_gym.experiments.recipes.legacy_speed2 import apply_legacy_speed2
        return apply_legacy_speed2(env_cfg, train_cfg)
    if str(name).upper() == 'LEGACY_ANCHORS':
        from wheel_legged_gym.experiments.recipes.legacy_anchors import apply_legacy_anchors
        return apply_legacy_anchors(env_cfg, train_cfg)
    if str(name).upper() == 'EXPLORE_CLEAN_OBS':
        from wheel_legged_gym.experiments.recipes.clean_observation import apply_clean_observation
        return apply_clean_observation(env_cfg, train_cfg)
    if str(name).upper() == 'EXPLORE_STOP_RETENTION':
        from wheel_legged_gym.experiments.recipes.stop_retention import apply_stop_retention
        return apply_stop_retention(env_cfg, train_cfg)
    if str(name).upper() == 'LEGACY_URDF':
        from wheel_legged_gym.experiments.recipes.legacy_urdf import apply_legacy_urdf
        return apply_legacy_urdf(env_cfg, train_cfg)
    if str(name).upper() in {'ENCODER_FROZEN', 'ENCODER_UPDATING'}:
        from wheel_legged_gym.experiments.recipes.h3_speed1 import apply_h3_speed1
        manifest = apply_h3_speed1(env_cfg, train_cfg)
        manifest.update(name=str(name).upper(), profile='encoder_ablation_v1',
            freeze_encoder_updates=str(name).upper() == 'ENCODER_FROZEN',
            command_diagnostics=True,
            ablation_variable='encoder optimizer.step enabled/disabled; all gradient calculation and RNG/batch iteration retained')
        return manifest
    if str(name).upper() in {'ENCODER_ANCHORED', 'ANCHORED_WHEEL_EXPLORE'}:
        from wheel_legged_gym.experiments.recipes.h3_speed1 import apply_h3_speed1
        manifest = apply_h3_speed1(env_cfg, train_cfg)
        manifest.update(name='ENCODER_ANCHORED', profile='encoder_ablation_v3',
            encoder_action_anchor_coef=1.0, freeze_encoder_updates=False,
            command_diagnostics=True,
            ablation_variable='encoder action-anchor penalty only; encoder remains trainable')
        if str(name).upper() == 'ANCHORED_WHEEL_EXPLORE':
            manifest.update(name='ANCHORED_WHEEL_EXPLORE',profile='anchored_wheel_explore_v1',
                wheel_exploration_initialization='left wheel std initialized to source right wheel std; all other model/Adam values retained',
                ablation_variable='initial left wheel std only, against encoder_ablation_v3')
        return manifest
    if str(name).upper() == 'H3_SPEED1':
        from wheel_legged_gym.experiments.recipes.h3_speed1 import apply_h3_speed1
        return apply_h3_speed1(env_cfg, train_cfg)
    if str(name).upper() == 'H3_LOW_SPEED':
        from wheel_legged_gym.experiments.recipes.h3_low_speed import apply_h3_low_speed
        return apply_h3_low_speed(env_cfg, train_cfg)
    if str(name).upper() in {'LOW_SPEED', 'LOW_SPEED_TRACKING'}:
        from wheel_legged_gym.experiments.recipes.low_speed import apply_low_speed
        return apply_low_speed(env_cfg, train_cfg, str(name).upper())
    if str(name).upper() in {'STAND_CONTROL', 'STAND_SYMMETRIC'}:
        from wheel_legged_gym.experiments.recipes.stand_balance import apply_stand_balance
        return apply_stand_balance(env_cfg, train_cfg, name,
                                   stand_randomization_level=stand_randomization_level)
    if str(name).upper() == 'FUDAN_STAND':
        from wheel_legged_gym.experiments.recipes.fudan_stand import apply_fudan_stand
        return apply_fudan_stand(env_cfg, train_cfg)
    if str(name).lower() in {"method_v1", "method"}:
        return apply_training_profile(env_cfg, train_cfg, profile="method_v1",
            phase=getattr(env_cfg.commands, "training_phase", "stand"),
            level=getattr(env_cfg.commands, "method_v1_level", 0),
            stand_randomization_level=stand_randomization_level)
    return apply_legacy_experiment(env_cfg, name, train_cfg)
