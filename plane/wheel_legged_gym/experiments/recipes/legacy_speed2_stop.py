"""Protect exact-zero stopping while retaining verified legacy speed2 motion."""
from wheel_legged_gym.experiments.recipes.legacy_speed2 import apply_legacy_speed2

def apply_legacy_speed2_stop(cfg, train):
    manifest=apply_legacy_speed2(cfg,train)
    cfg.commands.zero_retention=True
    cfg.commands.mixture_zero_fraction=.4
    cfg.commands.mixture_small_fraction=0.
    cfg.commands.training_profile='legacy_speed2_stop_v2'
    manifest.update(name='LEGACY_SPEED2_STOP',profile='legacy_speed2_stop_v2',
        exact_command_fractions={'0':.4,'-.5':.075,'+.5':.075,'-1':.075,'+1':.075,'-1.5':.075,'+1.5':.075,'-2':.075,'+2':.075},
        ablation_variable='sampling only: raise exact-zero from20 to40; preserve eight nonzero endpoint slots')
    return manifest
