"""Keep speed1 endpoints while increasing exact-zero exposure, no reward edits."""
from wheel_legged_gym.experiments.recipes.h3_speed1 import apply_h3_speed1


def apply_stop_retention(cfg,train):
    m=apply_h3_speed1(cfg,train)
    cfg.commands.zero_retention=True
    cfg.commands.training_profile='explore_stop_retention_v1'
    cfg.commands.mixture_zero_fraction=.4
    cfg.commands.mixture_small_fraction=0.
    m.update(name='EXPLORE_STOP_RETENTION',profile='explore_stop_retention_v1',
        encoder_action_anchor_coef=1.,freeze_encoder_updates=False,command_diagnostics=True,
        exact_command_fractions={'0':.4,'-.5':.15,'+.5':.15,'-1':.15,'+1':.15},
        ablation_variable='sampling only: transfer +/-0.1 slots to exact zero; keep endpoint exposure',
        source_acceptance='experimental near-pass model500; not a promoted or higher-speed model')
    return m
