"""Controlled standing fine-tunes: retain state, freeze encoder, vary geometry only."""
from wheel_legged_gym.experiments.primitives import apply_training_profile, apply_method_randomization


def apply_stand_balance(env_cfg, train_cfg, name, *, stand_randomization_level=None):
    name = name.upper()
    if name not in {'STAND_CONTROL', 'STAND_SYMMETRIC'}:
        raise ValueError(name)
    manifest = apply_training_profile(env_cfg, train_cfg, phase='stand', level=0,
                                      stand_randomization_level=stand_randomization_level)
    # Reconstruct the stable model_200 reward. Add geometry only in treatment.
    env_cfg.rewards.scales.stand_still = -.2
    env_cfg.rewards.scales.stand_bilateral_geometry = -.1 if name == 'STAND_SYMMETRIC' else 0.
    env_cfg.domain_rand_level = 1
    apply_method_randomization(env_cfg, 1)
    train_cfg.algorithm.learning_rate = 1e-5
    train_cfg.algorithm.schedule = 'fixed'
    train_cfg.algorithm.extra_learning_rate = 0.
    manifest.update(name=name, profile='stand_balance_v1',
        geometry_weight=env_cfg.rewards.scales.stand_bilateral_geometry,
        resume_semantics='full state; override learning rates; frozen encoder',
        acceptance_priority='survival, geometry, tilt/contact; small balancing drift permitted')
    env_cfg.commands.training_profile = 'stand_balance_v1'
    manifest['optimizer'].update(learning_rate=1e-5, schedule='fixed', extra_learning_rate=0.)
    manifest['reward_scales'] = {k:getattr(env_cfg.rewards.scales,k) for k in manifest['reward_scales']}
    return manifest
