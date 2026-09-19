"""First locomotion stage: +/-0.5 m/s, exact zero and +/-0.1 anchors."""
from .policy_experiments import apply_training_profile, _apply_method_randomization


def apply_low_speed(env_cfg, train_cfg, name='LOW_SPEED'):
    manifest = apply_training_profile(env_cfg, train_cfg, phase='translate', level=0)
    env_cfg.commands.training_profile = 'low_speed_v1'
    env_cfg.domain_rand_level = 1
    _apply_method_randomization(env_cfg, 1)
    # Keep differential balance authority; no geometric stand constraint in motion.
    env_cfg.rewards.scales.stand_bilateral_geometry = 0.
    env_cfg.rewards.scales.stand_still = 0.
    train_cfg.algorithm.learning_rate = 1e-5
    train_cfg.algorithm.extra_learning_rate = 1e-5
    train_cfg.algorithm.schedule = 'fixed'
    train_cfg.algorithm.entropy_coef = .001
    manifest.update(name='LOW_SPEED', profile='low_speed_v1',
        resume_semantics='explicit standing-to-locomotion warm start of full state; critic adapts to changed reward',
        acceptance_priority='signed tracking, zero-speed stop, contact, tilt and survival')
    manifest['optimizer'].update(learning_rate=1e-5, extra_learning_rate=1e-5,
                                 schedule='fixed', entropy_coef=.001)
    manifest['reward_scales'] = {k:getattr(env_cfg.rewards.scales,k) for k in manifest['reward_scales']}
    if name == 'LOW_SPEED_TRACKING':
        # One-variable ablation: normalize tracking errors to this stage's
        # endpoint instead of the old 1 m/s scale. No command compensation.
        env_cfg.rewards.tracking_linear_cap = .5
        manifest.update(name=name, profile='low_speed_tracking_v1')
        env_cfg.commands.training_profile = 'low_speed_tracking_v1'
    manifest['reward_parameters'] = {'tracking_linear_cap':env_cfg.rewards.tracking_linear_cap}
    return manifest
