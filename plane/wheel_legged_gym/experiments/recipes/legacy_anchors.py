"""Legacy-v1 branch continuation: change command curriculum, keep effective recipe."""
from wheel_legged_gym.experiments.recipes.legacy_urdf import apply_legacy_urdf


def apply_legacy_anchors(cfg,train):
    m=apply_legacy_urdf(cfg,train)
    cfg.commands.curriculum=False
    cfg.commands.sampling_strategy='method_v1'
    cfg.commands.hold_command_until_reset=True
    cfg.commands.training_profile='legacy_anchors_v1'
    cfg.commands.training_phase='translate'
    cfg.commands.method_v1_level=1
    cfg.commands.ranges.lin_vel_x=[-1.,1.]
    cfg.commands.ranges.ang_vel_yaw=[0.,0.]
    cfg.commands.translation_retention_anchors=(.5,1.)
    cfg.commands.zero_retention=False
    cfg.commands.mixture_zero_fraction=.2
    cfg.commands.mixture_small_fraction=.2
    cfg.commands.mixture_reverse_fraction=.3
    cfg.commands.mixture_forward_fraction=.3
    m.update(name='LEGACY_ANCHORS',profile='legacy_anchors_v1',resume_mode='full',
        ablation_variable='command curriculum only: fixed translation anchors/zero, no yaw, no automatic promotion',
        exact_command_fractions={'0':.2,'-.1':.1,'+.1':.1,'-.5':.15,'+.5':.15,'-1':.15,'+1':.15},
        caveat='Preserves actual legacy_urdf_v1 hybrid recipe including zero penalties; not exact upstream reproduction')
    return m
