"""Explicit multi-stage turn/height exploration; old turn recipe is unchanged."""
import math

from wheel_legged_gym.experiments.recipes.legacy_speed2_stop import apply_legacy_speed2_stop
from wheel_legged_gym.experiments.recipes.motion_goal import stage_bank


R10200 = '/home/kellen/fudan_train/plane/logs/wheel_legged/Sep22_11-19-04_motion_goal_20260922_111611_r01_basic_motion/model_10200.pt'
R10200_SHA = 'a8b9dc01879ddba41c54289c0367c6a3c93325bc2982357173b1899be059b790'


def validate_spec(spec):
    required = {'source_checkpoint', 'source_sha256', 'source_iteration',
                'teacher_checkpoint', 'teacher_sha256', 'stage', 'course_plan'}
    if set(spec) != required or not isinstance(spec['course_plan'], list):
        raise ValueError('Turn lean requires an explicit source, teacher, stage and course plan')
    plan = spec['course_plan']
    if (len(plan) != 3 or [item.get('phase') for item in plan] != [1, 2, 3]
            or not all(isinstance(item.get('budget_max'), int) and
                       0 < item['budget_max'] <= 30000 for item in plan)
            or sum(item['budget_max'] for item in plan) > 30000):
        raise ValueError('Course plan must contain three bounded phases within 30000 iterations')
    if (spec['teacher_checkpoint'], spec['teacher_sha256']) != (R10200, R10200_SHA):
        raise ValueError('Teacher must remain the reviewed R10200')
    if not isinstance(spec['source_iteration'], int) or spec['source_iteration'] < 10200:
        raise ValueError('Turn source iteration must name a saved checkpoint')
    if spec['source_iteration'] == 10200 and (spec['source_checkpoint'], spec['source_sha256']) != (R10200, R10200_SHA):
        raise ValueError('Initial source must be the exact reviewed R10200')
    stage = spec['stage']
    fields = {'phase', 'turn_height', 'lean_max_deg', 'turn_fraction', 'pairs',
              'learning_rate', 'freeze_encoder', 'encoder_learning_rate',
              'entry_range', 'hold_range', 'speed_ramp_seconds', 'yaw_ramp_seconds',
              'hypothesis','turn_height_reward_scale'}
    if not fields.issubset(stage) or set(stage) - fields - {'height_schedule','cohort_plan'} or stage['phase'] not in (1, 2, 3):
        raise ValueError('Unreviewed turn stage fields')
    if not (.34 <= stage['turn_height'] <= .40 and 0. <= stage['lean_max_deg'] <= 10.
            and stage['turn_fraction'] in (.25, .5)
            and 1. <= stage['turn_height_reward_scale'] <= 10.
            and 3e-6 <= stage['learning_rate'] <= 3e-5
            and stage['freeze_encoder'] in (True, False)):
        raise ValueError('Turn stage exceeds reviewed bounds')
    if stage['freeze_encoder']:
        if stage['encoder_learning_rate'] is not None:
            raise ValueError('Frozen encoder must not have an override')
    elif not (0. < stage['encoder_learning_rate'] <= stage['learning_rate']):
        raise ValueError('Encoder LR must not exceed actor LR')
    if (not .5 <= stage['entry_range'][0] <= stage['entry_range'][1] <= 2.
            or not 1. <= stage['hold_range'][0] <= stage['hold_range'][1] <= 6.
            or not 1. <= stage['speed_ramp_seconds'] <= 4.
            or not 1. <= stage['yaw_ramp_seconds'] <= 4.):
        raise ValueError('Unreviewed command timing')
    schedule = stage.get('height_schedule')
    if schedule is not None:
        if (not schedule or any(len(item) != 2 for item in schedule)
                or any(not (0. <= float(t) <= 16. and .34 <= float(h) <= .40)
                       for t, h in schedule)
                or any(schedule[i][0] >= schedule[i + 1][0]
                       for i in range(len(schedule) - 1))):
            raise ValueError('Height schedule must be ascending finite a_lat thresholds')
    pairs = [tuple(pair) for pair in stage['pairs']]
    if not pairs or len(pairs) > 32 or len(set(pairs)) != len(pairs) or not all(
            math.isfinite(v) and math.isfinite(w) and 0. < v <= 4. and 0. < w <= 4.
            and not (v == 4. and w == 4.) for v, w in pairs):
        raise ValueError('Turn pairs must be distinct, finite magnitudes in the reviewed range')
    cohort = stage.get('cohort_plan')
    if cohort is not None:
        if (set(cohort) != {'fractions','height_bank','mid_pairs','high_pairs','unclip_base_height'}
                or cohort['fractions'] != {'retention':.5,'height':.2,'mid_turn':.25,'high_turn':.05}
                or cohort['unclip_base_height'] is not True
                or not cohort['height_bank']
                or any(len(row) != 2 or abs(row[0]) > .5 or row[1] not in (.38,.36)
                       for row in cohort['height_bank'])
                or not cohort['mid_pairs'] or not cohort['high_pairs']
                or any(tuple(row) not in pairs for row in cohort['mid_pairs']+cohort['high_pairs'])):
            raise ValueError('Unreviewed cornering-height cohort')
    return spec


def apply_turn_lean_long(cfg, train, *, spec):
    spec = validate_spec(spec)
    stage = spec['stage']
    manifest = apply_legacy_speed2_stop(cfg, train)
    retention = stage_bank('basic_motion')
    bank = [(sv*v, sw*w) for v, w in stage['pairs']
            for sv in (-1., 1.) for sw in (-1., 1.)]
    cfg.commands.sampling_strategy = 'turn_envelope'
    cfg.commands.retention_bank = retention
    cfg.commands.turn_bank = bank
    cfg.commands.turn_stride = round(1/stage['turn_fraction'])
    cohort = stage.get('cohort_plan')
    if cohort is not None:
        cfg.commands.sampling_strategy = 'cornering_height_skill'
        cfg.commands.cohort_cycle = 20
        cfg.commands.height_slots = (0,2,4,6)
        cfg.commands.mid_slots = (8,10,12,14,16)
        cfg.commands.high_slots = (18,)
        cfg.commands.turn_slots = cfg.commands.mid_slots + cfg.commands.high_slots
        cfg.commands.height_skill_bank = cohort['height_bank']
        cfg.commands.turn_bank = [(sv*v,sw*w) for v,w in cohort['mid_pairs']
                                  for sv in (-1.,1.) for sw in (-1.,1.)]
        cfg.commands.high_turn_bank = [(sv*v,sw*w) for v,w in cohort['high_pairs']
                                       for sv in (-1.,1.) for sw in (-1.,1.)]
    cfg.commands.turn_height = stage['turn_height']
    if stage.get('height_schedule') is not None:
        cfg.commands.turn_height_schedule = tuple(tuple(item) for item in stage['height_schedule'])
    cfg.commands.turn_entry_range = tuple(stage['entry_range'])
    cfg.commands.turn_hold_range = tuple(stage['hold_range'])
    cfg.commands.turn_speed_ramp_seconds = stage['speed_ramp_seconds']
    cfg.commands.turn_yaw_ramp_seconds = stage['yaw_ramp_seconds']
    cfg.commands.hold_command_until_reset = True
    cfg.commands.ranges.lin_vel_x = [-4., 4.]
    cfg.commands.ranges.ang_vel_yaw = [-4., 4.]
    cfg.commands.ranges.height = [.4, .4]
    cfg.commands.training_profile = 'turn_lean_long_v1'
    cfg.commands.training_phase = 'combined'
    cfg.rewards.scales.nominal_state = 0.
    cfg.rewards.scales.stand_bilateral_geometry = -.2
    cfg.rewards.straight_bilateral_geometry = True
    cfg.rewards.bilateral_geometry_tolerance_m = .005
    cfg.rewards.bilateral_geometry_scale_m = .10
    cfg.rewards.turn_lean_max_rad = math.radians(stage['lean_max_deg'])
    cfg.rewards.turn_height_reward_scale = stage['turn_height_reward_scale']
    if cohort is not None:
        cfg.rewards.unclipped_reward_names = tuple(sorted(
            set(getattr(cfg.rewards, 'unclipped_reward_names', ())) | {'base_height'}))
    train.algorithm.schedule = 'fixed'
    train.algorithm.learning_rate = stage['learning_rate']
    train.algorithm.symmetry_loss_coef = .01
    if not stage['freeze_encoder']:
        train.algorithm.extra_learning_rate = stage['encoder_learning_rate']
    train.runner.save_interval = 250
    manifest.pop('exact_command_fractions', None)
    manifest.update(name='TURN_LEAN_LONG', profile=('cornering_height_skill_v1' if cohort else 'turn_lean_long_v1'),
        turn_long_spec=spec, retention_bank=retention, turn_bank=cfg.commands.turn_bank,
        turn_fraction=stage['turn_fraction'], turn_height=stage['turn_height'],
        turn_lean_max_rad=cfg.rewards.turn_lean_max_rad,
        turn_protocol={key:stage[key] for key in ('entry_range','hold_range',
            'speed_ramp_seconds','yaw_ramp_seconds')},
        freeze_motion_encoder=stage['freeze_encoder'],
        dynamic_fixed_lr=stage['learning_rate'], dynamic_equivariance=.01,
        dynamic_reference_coef=1.,
        reference_scope='R10200 mean action/source std on retention env_id % stride != 0',
        command_bank=retention,
        training_change='Joint turn height, lean, curriculum and optimizer exploration; no single-variable attribution',
        critic_transfer_note='Changed command/reward semantics; full model and Adam are initialization')
    if cohort is not None:
        manifest.update(cohort_plan=cohort, cohort_slots={'retention':'odd mod 20',
            'height':cfg.commands.height_slots,'mid_turn':cfg.commands.mid_slots,
            'high_turn':cfg.commands.high_slots},
            reward_clip_change='base_height only: exempt from single-term clipping; effective positive reward equals exp(-height_error^2/.001) times 8 when command <.4, times dt',
            unclipped_reward_names=cfg.rewards.unclipped_reward_names,
            reference_scope='R10200 mean action/source std on odd env IDs only (50% retention)')
    return manifest
