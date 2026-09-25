"""Match training observation noise to deterministic evaluation, retain dynamics randomization."""
from wheel_legged_gym.experiments.recipes.h3_speed1 import apply_h3_speed1


def apply_clean_observation(cfg,train):
    manifest=apply_h3_speed1(cfg,train)
    cfg.noise.add_noise=False
    cfg.commands.training_profile='explore_clean_obs_v1'
    manifest.update(name='EXPLORE_CLEAN_OBS',profile='explore_clean_obs_v1',
        encoder_action_anchor_coef=1.,freeze_encoder_updates=False,command_diagnostics=True,
        ablation_variable='observation noise disabled only; source500 mean/std/Adam, parameter randomization and original20/20/30/30 command mixture retained',
        observation_noise=False,source_acceptance='near-pass exploratory500; no stage promotion')
    return manifest
