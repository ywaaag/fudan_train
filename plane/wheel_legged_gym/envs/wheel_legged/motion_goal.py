"""Auditable command-only stages for the unattended motion curriculum."""
import hashlib
import json
import os
from pathlib import Path
from .legacy_speed2_stop import apply_legacy_speed2_stop

STAGES = ('yaw05', 'turn05', 'yaw1', 'speed3', 'yaw2', 'speed4', 'yaw3', 'yaw4', 'turns', 'switches')


def stage_bank(stage):
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


def read_spec():
    spec = json.loads(Path(os.environ['FUDAN_MOTION_GOAL_SPEC']).read_text())
    allowed = set(stage_bank(spec['stage']))
    focus = [tuple(p) for p in spec.get('focus', [])]
    if not all(p in allowed for p in focus) or len(focus) > 256:
        raise ValueError('Focus must retain the stage command contract')
    return spec


def apply_motion_goal(cfg, train):
    spec = read_spec()
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
    return manifest


def validate_source(path, resume, mode, cfg):
    spec = read_spec()
    if not resume or mode != 'full': raise ValueError('Motion goal requires full resume')
    path = Path(path).resolve()
    if path != Path(spec['source_checkpoint']).resolve(): raise ValueError('Source path mismatch')
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if sha != spec['source_sha256']: raise ValueError('Source checkpoint changed')
    old = json.loads((path.parent/'policy_experiment.json').read_text())
    if old['profile'] not in {'legacy_speed2_stop_v2','legacy_yaw05_v1','motion_goal_v1'}:
        raise ValueError('Unreviewed source recipe')
    for key,value in old['reward_scales'].items():
        if getattr(cfg.rewards.scales,key,0.) != value: raise ValueError('Reward mismatch: '+key)
    return {'source_checkpoint':str(path),'source_checkpoint_sha256':sha}
