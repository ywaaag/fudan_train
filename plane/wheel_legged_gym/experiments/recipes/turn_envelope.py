"""Matched fixed-height turn experiment from the reviewed R10200 source."""
from wheel_legged_gym.experiments.recipes.legacy_speed2_stop import apply_legacy_speed2_stop
from wheel_legged_gym.experiments.recipes.motion_goal import stage_bank


LOW_TURNS = tuple((sv * v, sw * w) for v in (.5, 1., 2.)
                  for w in (.25, .5) for sv in (-1., 1.) for sw in (-1., 1.))


def validate_spec(spec):
    required = {'source_checkpoint', 'source_sha256', 'source_iteration', 'branch'}
    if not required.issubset(spec) or spec['branch'] not in ('A', 'B'):
        raise ValueError('Turn envelope requires an explicit R10200 source and A/B branch')
    if spec['source_iteration'] != 10200 or spec['source_sha256'] != 'a8b9dc01879ddba41c54289c0367c6a3c93325bc2982357173b1899be059b790':
        raise ValueError('Unreviewed turn source')
    if set(spec) != required:
        raise ValueError('Unreviewed turn specification fields')
    return spec


def apply_turn_envelope(cfg, train, *, spec):
    spec = validate_spec(spec)
    manifest = apply_legacy_speed2_stop(cfg, train)
    retention = stage_bank('basic_motion')
    cfg.commands.sampling_strategy = 'turn_envelope'
    cfg.commands.retention_bank = retention
    cfg.commands.turn_bank = LOW_TURNS
    cfg.commands.turn_stride = 4
    cfg.commands.hold_command_until_reset = True
    cfg.commands.ranges.lin_vel_x = [-4., 4.]
    cfg.commands.ranges.ang_vel_yaw = [-4., 4.]
    cfg.commands.ranges.height = [.4, .4]
    cfg.commands.training_profile = 'turn_envelope_v1'
    cfg.commands.training_phase = 'combined'
    cfg.rewards.scales.nominal_state = 0.
    cfg.rewards.scales.stand_bilateral_geometry = -.2
    cfg.rewards.straight_bilateral_geometry = True
    cfg.rewards.bilateral_geometry_tolerance_m = .005
    cfg.rewards.bilateral_geometry_scale_m = .10
    cfg.rewards.turn_lean_max_rad = .03490658503988659 if spec['branch'] == 'B' else 0.
    train.algorithm.schedule = 'fixed'
    train.algorithm.learning_rate = 1e-6
    train.algorithm.symmetry_loss_coef = .01
    train.runner.save_interval = 100
    manifest.pop('exact_command_fractions', None)
    manifest.update(name='TURN_ENVELOPE', profile='turn_envelope_v1', turn_spec=spec,
        retention_bank=retention, turn_bank=LOW_TURNS, turn_fraction=.25,
        turn_protocol={'speed_ramp_seconds':2., 'yaw_ramp_seconds':2.,
                       'random_entry_seconds':[.5,1.5], 'random_hold_seconds':[2.,5.],
                       'exit_yaw_ramp_seconds':2.},
        orientation_target='level' if spec['branch']=='A' else 'bounded_command_lean',
        turn_lean_max_rad=cfg.rewards.turn_lean_max_rad,
        freeze_motion_encoder=True, dynamic_fixed_lr=1e-6,
        dynamic_equivariance=.01, dynamic_reference_coef=1.,
        reference_scope='Mean action / source std on env_id % 4 != 0 only',
        command_bank=retention,
        ablation_variable='orientation target formula only; A and B share sampling and optimizer',
        critic_transfer_note='Changed task/reward semantics; full state and Adam used as initialization')
    return manifest
