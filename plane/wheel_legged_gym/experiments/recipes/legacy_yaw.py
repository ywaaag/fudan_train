"""First yaw curriculum with explicit retention of every accepted speed2 anchor."""
from wheel_legged_gym.experiments.recipes.legacy_speed2_stop import apply_legacy_speed2_stop


def apply_legacy_yaw(cfg, train):
    manifest = apply_legacy_speed2_stop(cfg, train)
    bank = [(0., 0.)] * 8 + [(v, 0.) for v in (-.5, .5, -1., 1., -1.5, 1.5, -2., 2.)]
    bank += [(0., -.5), (0., .5)] * 2
    cfg.commands.sampling_strategy = 'fixed_bank'
    cfg.commands.fixed_bank = bank
    cfg.commands.ranges.ang_vel_yaw = [-.5, .5]
    cfg.commands.training_profile = 'legacy_yaw05_v1'
    cfg.commands.training_phase = 'yaw'
    manifest.pop('exact_command_fractions', None)
    manifest.update(name='LEGACY_YAW', profile='legacy_yaw05_v1',
        command_bank=bank, command_semantics='20 equal slots held until episode reset',
        ablation_variable='sampling only: zero40%, eight translation anchors40%, pure yaw +/-0.5 20%',
        scope='Pure yaw introduction; combined turns and within-episode switches are evaluation-only')
    return manifest
