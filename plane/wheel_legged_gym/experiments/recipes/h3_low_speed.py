"""Explicit H3 actor/encoder/std migration into normalized low-speed rewards."""

from wheel_legged_gym.experiments.recipes.low_speed import apply_low_speed


def apply_h3_low_speed(env_cfg, train_cfg):
    manifest = apply_low_speed(env_cfg, train_cfg, 'LOW_SPEED')
    env_cfg.commands.training_profile = 'h3_low_speed_v1'
    manifest.update(name='H3_LOW_SPEED', profile='h3_low_speed_v1',
        resume_semantics='actor/encoder/std from verified H3; fresh critic, PPO/encoder optimizers, iteration and curriculum',
        resume_mode='policy', std_initialization='copy source six-channel std exactly',
        comparison_limit='Migration probe; not a causal single-variable comparison against historical full-state LOW_SPEED runs')
    return manifest
