"""Auditable command-only stages for the unattended motion curriculum."""
from wheel_legged_gym.experiments.recipes.legacy_speed2_stop import apply_legacy_speed2_stop

STAGES = ('yaw05', 'turn05', 'yaw1', 'speed3', 'yaw2', 'speed4', 'yaw3', 'yaw4', 'turns', 'switches')


def stage_bank(stage):
    if stage=='basic_motion':
        # 60% equally split across the five priority modes; 40% intermediate retention.
        bank=[(0.,0.),(-4.,0.),(4.,0.),(0.,-4.),(0.,4.)]*6
        bank += [(s*v,0.) for v in [.5,1.,1.5,2.,2.5,3.,3.5] for s in (-1,1)]
        bank += [(0.,s*w) for w in [.5,1.,2.] for s in (-1,1)]
        return bank
    if stage.startswith('envelope'):
        level=int(stage[-1])
        if level not in range(1,5):raise ValueError('Unknown envelope stage')
        bank=stage_bank('yaw4')
        pairs=[(1.,.5),(1.,1.),(1.,1.5),(1.,2.)]
        if level>=2:pairs += [(1.5,1.),(1.5,1.5),(2.,1.),(2.,1.5)]
        bank += [(s*v,t*w) for v,w in pairs for s in (-1,1) for t in (-1,1)]
        if level>=3:bank += [(3.,s*w) for w in [.5,1.] for s in (-1,1)]
        if level>=4:bank += [(v,s*w) for v in [3.5,4.] for w in [.6,.8] for s in (-1,1)]
        return bank
    if stage in ('turn1','turn2','turn3','turn4'):
        level=int(stage[-1])
        bank=stage_bank('yaw4')
        pairs=[(1.,.5),(1.,1.)]
        if level>=2:pairs += [(2.,.5),(2.,1.)]
        if level>=3:pairs += [(3.,.25),(3.,.5)]
        if level>=4:pairs += [(4.,.25),(4.,.5)]
        return bank+[(s*v,t*w) for v,w in pairs for s in (-1,1) for t in (-1,1)]
    index = STAGES.index(stage)
    speeds = [.5, 1., 1.5, 2.]
    if index >= 3: speeds += [2.5, 3.]
    if index >= 5: speeds += [3.5, 4.]
    yaws = [.5]
    if index >= 2: yaws += [1.]
    if index >= 4: yaws += [2.]
    if index >= 6: yaws += [3.]
    if index >= 7: yaws += [4.]
    bank = [(0., 0.)] * 8
    bank += [(s*v, 0.) for v in speeds for s in (-1, 1)]
    bank += [(0., s*w) for w in yaws for s in (-1, 1)] * 2
    if index >= 1:
        bank += [(s*.5, t*.5) for s in (-1, 1) for t in (-1, 1)]
    if index >= 8:
        bank += [(s*v, t*w) for v,w in [(1.,1.),(2.,1.),(4.,.5)] for s in (-1,1) for t in (-1,1)]
    return bank


def validate_spec(spec):
    allowed = set(stage_bank(spec['stage']))
    focus = [tuple(p) for p in spec.get('focus', [])]
    if not all(p in allowed for p in focus) or len(focus) > 256:
        raise ValueError('Focus must retain the stage command contract')
    return spec


def apply_motion_goal(cfg, train, *, spec):
    spec = validate_spec(spec)
    manifest = apply_legacy_speed2_stop(cfg, train)
    bank = stage_bank(spec['stage']) + [tuple(p) for p in spec.get('focus', [])]
    cfg.commands.sampling_strategy = 'fixed_bank'
    cfg.commands.fixed_bank = bank
    cfg.commands.ranges.lin_vel_x = [min(v for v,w in bank), max(v for v,w in bank)]
    cfg.commands.ranges.ang_vel_yaw = [min(w for v,w in bank), max(w for v,w in bank)]
    cfg.commands.training_profile = 'motion_goal_v1'
    cfg.commands.training_phase = 'combined'
    cfg.commands.hold_command_until_reset = spec['stage'] != 'switches'
    if spec['stage'] == 'switches': cfg.commands.resampling_time = 5.
    manifest.pop('exact_command_fractions', None)
    manifest.update(name='MOTION_GOAL', profile='motion_goal_v1', motion_goal_spec=spec,
        command_bank=bank, command_semantics='Equal deterministic slots; all base-stage slots always retained',
        ablation_variable='command curriculum only; reward, asset and optimizer unchanged')
    if spec.get('geometry_symmetry',False):
        geometry_weight=spec.get('geometry_weight',-.1)
        if geometry_weight not in (-.1,-.2):
            raise ValueError('Unreviewed geometry weight')
        cfg.rewards.scales.nominal_state=0.
        cfg.rewards.scales.stand_bilateral_geometry=geometry_weight
        cfg.rewards.straight_bilateral_geometry=True
        cfg.rewards.bilateral_geometry_tolerance_m=.005
        cfg.rewards.bilateral_geometry_scale_m=.10
        manifest.update(ablation_variable='replace legacy virtual-angle symmetry with weak real root-frame geometry on zero-yaw commands',
            geometry_symmetry={'weight':geometry_weight,'tolerance_m':.005,'scale_m':.10,'yaw_gate':.01},
            critic_transfer_note='Reward changed; critic and Adam are initialization, not same-objective continuation')
    if spec.get('command_switch_seconds',0):
        if spec['stage']!='basic_motion' or spec['command_switch_seconds']!=5:
            raise ValueError('Reviewed switching experiment requires basic_motion, 5 seconds')
        cfg.commands.hold_command_until_reset=False
        cfg.commands.resampling_time=5.
        manifest.update(ablation_variable='command timing only: resample same bank every 5 seconds',
            command_semantics='Same 50 slots; commands change within episodes; source geometry reward preserved',
            critic_transfer_note='Source reward unchanged; command distribution timing changes')
    if spec.get('freeze_motion_encoder',False):
        if spec['stage']!='basic_motion':
            raise ValueError('Encoder ablation requires basic_motion')
        manifest.update(freeze_motion_encoder=True,
            encoder_ablation='Skip encoder optimizer steps after verified full resume; preserve weights, LR and Adam state')
    if spec.get('start_stop_ramp_seconds',0.):
        if spec['stage']!='basic_motion' or spec.get('command_switch_seconds',0):
            raise ValueError('Start-stop excludes global command switching')
        if spec['start_stop_ramp_seconds'] not in (.5,1.,2.):
            raise ValueError('Unreviewed start-stop ramp')
        cfg.commands.start_stop_ramp_seconds=spec['start_stop_ramp_seconds']
        cfg.commands.start_stop_speed=spec.get('start_stop_speed',1.)
        fraction=spec.get('start_stop_fraction',.5)
        if fraction not in (0.,.25,.5):raise ValueError('Unreviewed dynamic fraction')
        cfg.commands.start_stop_stride=round(1/fraction) if fraction else 2
        if fraction==0.:cfg.commands.start_stop_ramp_seconds=0.
        if cfg.commands.start_stop_speed not in (1.,2.,3.,4.):raise ValueError('Invalid dynamic speed')
        manifest.update(start_stop={'amplitude_m_s':cfg.commands.start_stop_speed,'duration_range_seconds':[.5,4.],
            'ramp_seconds':spec['start_stop_ramp_seconds'],'dynamic_fraction':fraction,
            'initial_episode_lengths_randomized':True,'static_control':fraction==0.,
            'timing':'Static control disables timers' if fraction==0. else 'Independent random timers; ramp parameter is zero-to-endpoint time; reversal takes twice as long'},
            ablation_variable='Explicit dynamic cohort fraction; remaining environments retain original fixed command bank',
            critic_transfer_note='Reward preserved from source; command timing and episode-start distribution changed')
    if spec.get('dynamic_fixed_lr'):
        if not spec.get('start_stop_ramp_seconds') or spec['dynamic_fixed_lr']!=1e-6:
            raise ValueError('Unreviewed dynamic optimizer ablation')
        train.algorithm.schedule='fixed'
        train.algorithm.learning_rate=spec['dynamic_fixed_lr']
        manifest['dynamic_fixed_lr']=spec['dynamic_fixed_lr']
        manifest['optimizer_ablation']='Full model and both Adam states verified first; then only PPO group LR/schedule overridden, moments retained'
    if spec.get('dynamic_equivariance'):
        if not spec.get('start_stop_ramp_seconds') or spec['dynamic_equivariance']!=.01:
            raise ValueError('Unreviewed dynamic equivariance coefficient')
        train.algorithm.symmetry_loss_coef=spec['dynamic_equivariance']
        manifest['dynamic_equivariance']=spec['dynamic_equivariance']
        manifest['equivariance_note']='Mirror observations/history and actor targets; never equalize same-state left/right actions. Encoder remains frozen.'
    if spec.get('dynamic_reference_coef'):
        if not spec.get('start_stop_ramp_seconds') or spec.get('start_stop_fraction',.5) not in (.25,.5) or spec['dynamic_reference_coef']!=1.:
            raise ValueError('Unreviewed policy reference recipe')
        manifest['dynamic_reference_coef']=spec['dynamic_reference_coef']
        manifest['reference_scope']='Training-only mean-action reference on retention cohort; teacher is verified resume source, no inference assistance'
    return manifest
