"""One controlled speed-stage increase with old endpoints retained."""
from wheel_legged_gym.experiments.recipes.low_speed import apply_low_speed


def apply_h3_speed1(cfg, train):
    manifest = apply_low_speed(cfg, train)
    cfg.commands.training_profile = 'h3_speed1_v1'
    cfg.commands.method_v1_level = 1
    cfg.commands.ranges.lin_vel_x = [-1., 1.]
    cfg.commands.translation_retention_anchors = (.5, 1.)
    cfg.commands.mixture_endpoint_linear_anchors = (-1., -.5, .5, 1.)
    manifest.update(name='H3_SPEED1',profile='h3_speed1_v1',level=1,command_limits=(1.,0.),
        resume_mode='full',resume_semantics='Full network/std/Adam continuation; unchanged rewards, expanded command distribution',
        translation_retention_anchors=[.5,1.],
        exact_command_fractions={'0':.2,'-0.1':.1,'+0.1':.1,'-0.5':.15,'+0.5':.15,'-1':.15,'+1':.15})
    return manifest
