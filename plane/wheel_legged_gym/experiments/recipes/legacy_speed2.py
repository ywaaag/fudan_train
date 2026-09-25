"""Advance verified legacy-anchor policy to 1.5/2m/s with low-speed retention."""
from wheel_legged_gym.experiments.recipes.legacy_anchors import apply_legacy_anchors


def apply_legacy_speed2(cfg,train):
    m=apply_legacy_anchors(cfg,train)
    cfg.commands.training_profile='legacy_speed2_v1'
    cfg.commands.method_v1_level=2
    cfg.commands.ranges.lin_vel_x=[-2.,2.]
    cfg.commands.translation_retention_anchors=(.5,1.,1.5,2.)
    m.update(name='LEGACY_SPEED2',profile='legacy_speed2_v1',
        ablation_variable='command level only: retain0/.1/.5/1 and add1.5/2, no yaw',
        translation_retention_anchors=[.5,1.,1.5,2.],
        exact_command_fractions={'0':.2,'-.1':.1,'+.1':.1,**{str(s*v):.075 for v in [.5,1.,1.5,2.] for s in [-1,1]}})
    return m
