"""Explicit multi-stage turn/height exploration; old turn recipe is unchanged."""
import math

from wheel_legged_gym.experiments.recipes.legacy_speed2_stop import apply_legacy_speed2_stop
from wheel_legged_gym.experiments.recipes.motion_goal import stage_bank


R10200 = '/home/kellen/fudan_train/plane/logs/wheel_legged/Sep22_11-19-04_motion_goal_20260922_111611_r01_basic_motion/model_10200.pt'
R10200_SHA = 'a8b9dc01879ddba41c54289c0367c6a3c93325bc2982357173b1899be059b790'


def validate_spec(spec, *, historical=False):
    required = {'source_checkpoint', 'source_sha256', 'source_iteration',
                'teacher_checkpoint', 'teacher_sha256', 'stage', 'course_plan'}
    experiment_id = spec.get('experiment_id')
    inward = experiment_id in ('inward_cornering_r10200_v1',
                                'roll_constraint_ablation_v1',
                                'cornering_sampling_ablation_v1')
    if (set(spec) != required | ({'experiment_id'} if inward else set())
            or not isinstance(spec['course_plan'], list)):
        raise ValueError('Turn lean requires an explicit source, teacher, stage and course plan')
    plan = spec['course_plan']
    phases = [1, 2] if inward else [1, 2, 3]
    limit = 25000 if inward else 30000
    if (len(plan) != len(phases) or [item.get('phase') for item in plan] != phases
            or not all(isinstance(item.get('budget_max'), int) and
                       0 < item['budget_max'] <= limit for item in plan)
            or sum(item['budget_max'] for item in plan) > limit):
        raise ValueError('Course plan exceeds the experiment iteration limit')
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
    if (not fields.issubset(stage) or
            set(stage) - fields - {'height_schedule','cohort_plan','aggressive_plan','inward_geometry_probe',
                                   'turn_roll_cost_coefficient','sampling_ablation'} or
            stage['phase'] not in phases or
            (stage.get('inward_geometry_probe',False) and not inward)):
        raise ValueError('Unreviewed turn stage fields')
    if experiment_id == 'roll_constraint_ablation_v1':
        if (stage['phase'] != 2 or stage['turn_height'] != .4 or
                stage.get('cohort_plan') is not None or
                stage.get('aggressive_plan', {}).get('sampling_version') != 2 or
                stage.get('turn_roll_cost_coefficient') not in (1., .25)):
            raise ValueError('Roll ablation requires fixed-height sampling v2 and an explicit coefficient')
    if experiment_id == 'cornering_sampling_ablation_v1':
        sampling = stage.get('sampling_ablation')
        if (stage['phase'] != 2 or stage['turn_height'] != .4 or
                stage.get('cohort_plan') is not None or
                stage.get('aggressive_plan', {}).get('sampling_version') != 2 or
                stage.get('turn_roll_cost_coefficient') != 1. or
                (sampling is not None and (
                    set(sampling) != {'version','failed_centers','other_centers','vx_half_width','yaw_half_width'}
                    or sampling['version'] != 1
                    or sampling['vx_half_width'] != .15
                    or sampling['yaw_half_width'] != .05
                    or len(sampling['failed_centers']) != 8
                    or len(sampling['other_centers']) != 4))):
            raise ValueError('Sampling ablation requires fixed-height v2 source settings')
        if sampling is not None:
            allowed = {(-4.,-1.),(-4.,1.),(-3.5,-.8),(-3.,-1.),(-3.,1.),
                       (3.,-1.),(3.5,-.8),(4.,-1.)}
            others = {(3.,1.),(-3.5,.8),(3.5,.8),(4.,1.)}
            if ({tuple(row) for row in sampling['failed_centers']} != allowed or
                    {tuple(row) for row in sampling['other_centers']} != others):
                raise ValueError('Sampling ablation centers must match frozen failure and complement points')
    if (experiment_id not in ('roll_constraint_ablation_v1',
                              'cornering_sampling_ablation_v1') and
            'turn_roll_cost_coefficient' in stage):
        raise ValueError('Roll coefficient is reserved for the ablation experiment')
    if experiment_id != 'cornering_sampling_ablation_v1' and 'sampling_ablation' in stage:
        raise ValueError('Targeted sampling is reserved for its explicit experiment')
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
    if (inward and stage['phase'] == 1 and
            (stage['turn_height'] != .4 or stage['lean_max_deg'] > 3. or
             stage.get('height_schedule') is not None or stage.get('cohort_plan') is not None)):
        raise ValueError('Inward stage 1 must use fixed 0.40m and at most 3 degree lean')
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
    aggressive = stage.get('aggressive_plan')
    if aggressive is not None:
        if (set(aggressive) - {'fractions','anchor_pairs','regional_pairs','mid_pairs',
                              'high_pairs','sampling_version'} or
                not {'fractions','anchor_pairs','regional_pairs','mid_pairs',
                     'high_pairs'}.issubset(aggressive) or
                (aggressive.get('sampling_version') != 2 and not historical)):
            raise ValueError('Unreviewed aggressive cornering plan')
        if aggressive['fractions'] != {'anchor':.15,'regional':.225,'mid':.10,'high':.025}:
            raise ValueError('Aggressive fractions must preserve 50 percent retention')
        if historical and aggressive.get('sampling_version', 1) not in (1, 2):
            raise ValueError('Unknown historical sampling version')
        if stage['turn_fraction'] != .5 or any(
                not aggressive[key] or any(tuple(pair) not in pairs
                    for pair in aggressive[key])
                for key in ('anchor_pairs','regional_pairs','mid_pairs','high_pairs')):
            raise ValueError('Aggressive banks must be nonempty reviewed turn pairs at 50 percent retention')
        if cohort is not None:
            raise ValueError('Aggressive and height cohorts are mutually exclusive')
    if cohort is not None:
        reviewed_fractions = (
            {'retention':.5,'height':.2,'mid_turn':.25,'high_turn':.05},
            {'retention':.5,'height':.3,'mid_turn':.15,'high_turn':.05},
        )
        if (set(cohort) != {'fractions','height_bank','mid_pairs','high_pairs','unclip_base_height'}
                or cohort['fractions'] not in reviewed_fractions
                or cohort['unclip_base_height'] is not True
                or not cohort['height_bank']
                or any(len(row) != 2 or abs(row[0]) > .5 or row[1] not in (.38,.36)
                       for row in cohort['height_bank'])
                or not cohort['mid_pairs'] or not cohort['high_pairs']
                or any(tuple(row) not in pairs for row in cohort['mid_pairs']+cohort['high_pairs'])):
            raise ValueError('Unreviewed cornering-height cohort')
    return spec


def apply_turn_lean_long(cfg, train, *, spec, historical=False):
    spec = validate_spec(spec, historical=historical)
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
    aggressive = stage.get('aggressive_plan')
    if cohort is not None:
        cfg.commands.sampling_strategy = 'cornering_height_skill'
        cfg.commands.cohort_cycle = 20
        dynamic_slots = tuple(range(0,20,2))
        height_count = round(cohort['fractions']['height'] * 20)
        mid_count = round(cohort['fractions']['mid_turn'] * 20)
        cfg.commands.height_slots = dynamic_slots[:height_count]
        cfg.commands.mid_slots = dynamic_slots[height_count:height_count+mid_count]
        cfg.commands.high_slots = dynamic_slots[height_count+mid_count:]
        cfg.commands.turn_slots = cfg.commands.mid_slots + cfg.commands.high_slots
        cfg.commands.height_skill_bank = cohort['height_bank']
        cfg.commands.turn_bank = [(sv*v,sw*w) for v,w in cohort['mid_pairs']
                                  for sv in (-1.,1.) for sw in (-1.,1.)]
        cfg.commands.high_turn_bank = [(sv*v,sw*w) for v,w in cohort['high_pairs']
                                       for sv in (-1.,1.) for sw in (-1.,1.)]
    if aggressive is not None:
        cfg.commands.sampling_strategy = 'aggressive_cornering'
        version = aggressive.get('sampling_version', 1)
        cfg.commands.aggressive_sampling_version = version
        if version == 2:
            cfg.commands.cohort_cycle = 40
            cfg.commands.turn_slots = tuple(range(0, 40, 2))
            cfg.commands.anchor_slots = tuple(range(0,12,2))
            cfg.commands.regional_slots = tuple(range(12,30,2))
            cfg.commands.mid_slots = tuple(range(30,38,2))
            cfg.commands.high_slots = (38,)
            cfg.commands.regional_range = (
                min(v for v, _ in aggressive['regional_pairs']),
                max(v for v, _ in aggressive['regional_pairs']),
                min(w for _, w in aggressive['regional_pairs']),
                max(w for _, w in aggressive['regional_pairs']))
        else:
            cfg.commands.cohort_cycle = 20
            cfg.commands.anchor_slots = (0,2,4)
            cfg.commands.regional_slots = (6,8,10,12,14)
            cfg.commands.mid_slots = (16,)
            cfg.commands.high_slots = (18,)
        cfg.commands.anchor_bank = [(sv*v,sw*w) for v,w in aggressive['anchor_pairs'] for sv in (-1.,1.) for sw in (-1.,1.)]
        cfg.commands.regional_bank = [(sv*v,sw*w) for v,w in aggressive['regional_pairs'] for sv in (-1.,1.) for sw in (-1.,1.)]
        cfg.commands.mid_bank = [(sv*v,sw*w) for v,w in aggressive['mid_pairs'] for sv in (-1.,1.) for sw in (-1.,1.)]
        cfg.commands.high_bank = [(sv*v,sw*w) for v,w in aggressive['high_pairs'] for sv in (-1.,1.) for sw in (-1.,1.)]
        if stage.get('sampling_ablation') is not None:
            sampling = stage['sampling_ablation']
            cfg.commands.sampling_strategy = 'targeted_cornering'
            cfg.commands.failed_slots = tuple(range(0,16,2))
            cfg.commands.original_slots = tuple(range(16,32,2))
            cfg.commands.other_slots = tuple(range(32,40,2))
            cfg.commands.failed_centers = tuple(tuple(row) for row in sampling['failed_centers'])
            cfg.commands.other_centers = tuple(tuple(row) for row in sampling['other_centers'])
            cfg.commands.targeted_half_widths = (sampling['vx_half_width'],sampling['yaw_half_width'])
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
    cfg.commands.training_profile = ('inward_cornering_r10200_v1' if spec.get('experiment_id')
                                     == 'inward_cornering_r10200_v1' else 'turn_lean_long_v1')
    cfg.commands.training_phase = 'combined'
    cfg.rewards.scales.nominal_state = 0.
    cfg.rewards.scales.stand_bilateral_geometry = -.2
    cfg.rewards.straight_bilateral_geometry = True
    cfg.rewards.bilateral_geometry_tolerance_m = .005
    cfg.rewards.bilateral_geometry_scale_m = .10
    cfg.rewards.turn_lean_max_rad = math.radians(stage['lean_max_deg'])
    cfg.rewards.turn_roll_cost_coefficient = stage.get('turn_roll_cost_coefficient', 1.)
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
    manifest.update(name='TURN_LEAN_LONG', profile=('inward_cornering_r10200_v1'
        if spec.get('experiment_id') == 'inward_cornering_r10200_v1' else
        'cornering_height_skill_v1' if cohort else 'aggressive_cornering_v1' if aggressive else 'turn_lean_long_v1'),
        turn_long_spec=spec, retention_bank=retention, turn_bank=cfg.commands.turn_bank,
        turn_fraction=stage['turn_fraction'], turn_height=stage['turn_height'],
        turn_lean_max_rad=cfg.rewards.turn_lean_max_rad,
        turn_roll_cost_coefficient=cfg.rewards.turn_roll_cost_coefficient,
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
    if aggressive is not None:
        manifest.update(aggressive_plan=aggressive,
            sampling_version=cfg.commands.aggressive_sampling_version,
            cohort_slots={'retention':'odd modulo '+str(cfg.commands.cohort_cycle),
                'anchor':cfg.commands.anchor_slots,
                'regional':cfg.commands.regional_slots,
                'mid':cfg.commands.mid_slots,'high':cfg.commands.high_slots},
            aggressive_slots={'anchor':cfg.commands.anchor_slots,
                'regional':cfg.commands.regional_slots,'mid':cfg.commands.mid_slots,
                'high':cfg.commands.high_slots},
            aggressive_sampling_version=cfg.commands.aggressive_sampling_version,
            effective_command_fractions={key:len(slots)/cfg.commands.cohort_cycle
                for key,slots in (('anchor',cfg.commands.anchor_slots),
                    ('regional',cfg.commands.regional_slots),
                    ('mid',cfg.commands.mid_slots),('high',cfg.commands.high_slots))},
            regional_command_range=(cfg.commands.regional_range
                if cfg.commands.aggressive_sampling_version == 2 else None),
            reference_scope='R10200 mean action/source std on odd env IDs only (50% retention)')
        if cfg.commands.aggressive_sampling_version == 2:
            cfg.rewards.scales.turn_wheel_slip = -2.0
            cfg.rewards.scales.turn_contact_loss = -0.5
            manifest.update(
                ablation_variable='Explicit turning slip/contact penalty with fixed sampling v2',
                training_change='Sampling v2 plus explicit turning slip/contact reward; LR, lean target and policy contract unchanged',
                turning_reward={'turn_wheel_slip': {'raw_scale': -2.0,
                    'mask': 'abs(command_vx)>=1.5 and abs(command_yaw)>=0.35',
                    'formula': 'contact-conditioned normalized Huber of per-wheel rolling residual'},
                    'turn_contact_loss': {'raw_scale': -0.5,
                    'mask': 'abs(command_vx*command_yaw)>=1.0',
                    'formula': '1 - mean(two-wheel contact history)'}},
                turning_reward_unclipped=False)
    if spec.get('experiment_id') == 'roll_constraint_ablation_v1':
        manifest.update(ablation_variable='turn_cohort_roll_error_coefficient',
            training_change='Only turn-cohort roll reference error cost differs between A and B; pitch, orientation scale and all other training settings are fixed',
            reference_scope='R10200 mean action/source std on odd env IDs only (50% retention)',
            roll_cost_scope='turn slots use coefficient {}; retention uses 1.0'.format(
                cfg.rewards.turn_roll_cost_coefficient))
    if spec.get('experiment_id') == 'cornering_sampling_ablation_v1':
        sampling = stage.get('sampling_ablation')
        manifest.update(ablation_variable='turn_command_sampling_distribution',
            training_change='Only command sampling distribution differs between A and B',
            sampling_ablation=sampling,
            declared_sampling_fractions=({'retention':.5,'failed':.2,'original':.2,'other':.1}
                if sampling is not None else {'retention':.5,'anchor':.15,
                    'regional':.225,'mid':.1,'high':.025}),
            cohort_slots=({'retention':'odd modulo 40',
                'failed':cfg.commands.failed_slots,'original':cfg.commands.original_slots,
                'other':cfg.commands.other_slots} if sampling is not None else manifest['cohort_slots']))
        if sampling is not None:
            manifest.update(effective_command_fractions={
                'failed':len(cfg.commands.failed_slots)/40,
                'original':len(cfg.commands.original_slots)/40,
                'other':len(cfg.commands.other_slots)/40},
                aggressive_slots={'failed':cfg.commands.failed_slots,
                    'original':cfg.commands.original_slots,
                    'other':cfg.commands.other_slots},
                original_turn_conditional_weights={'anchor':.30,'regional':.45,
                    'mid':.20,'high':.05},
                targeted_command_limits={'abs_vx_max':4.,'abs_yaw_max':1.,
                    'abs_vx_times_yaw_max':4.})
    return manifest
