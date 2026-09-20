"""Command-only height curriculum with explicit retention of accepted motion."""
import hashlib
import json
import os
from pathlib import Path
from .legacy_speed2_stop import apply_legacy_speed2_stop
from .motion_goal import stage_bank

STAGES = ('near', 'middle', 'full')


def height_bank(stage, variable_repeats=1):
    if variable_repeats not in (1, 4):
        raise ValueError('Reviewed height sampling repeats are 1 and 4')
    if stage in ('fixed35','fixed45'):
        return [(0.,0.,.35 if stage=='fixed35' else .45)]
    if stage=='dual':
        return [(0.,0.,.35),(0.,0.,.45)]
    limits = {'micro':(.39,.41),'near':(.375,.425), 'middle':(.35,.45), 'full':(.30,.45)}
    low,high=limits[stage]
    # Keep the entire accepted 40cm motion grid, not only zero/low-speed anchors.
    bank=[(v,w,.4) for v,w in stage_bank('yaw4')]
    heights=([.39,.41] if stage=='micro' else
             [h/1000 for h in range(round(low*1000),round(high*1000)+1,25) if h!=400])
    for h in heights:
        variable = [(0.,0.,h)]*4
        variable += [(v,w,h) for v,w in [(-.5,0),(.5,0),(-1,0),(1,0),(0,-.5),(0,.5)]]
        bank += variable * variable_repeats
    return bank


def spec():
    return json.loads(Path(os.environ['FUDAN_HEIGHT_SPEC']).read_text())


def apply_height_course(cfg,train):
    s=spec();manifest=apply_legacy_speed2_stop(cfg,train)
    bank=height_bank(s['stage'],s.get('variable_repeats',1))
    cfg.commands.sampling_strategy='height_bank'
    cfg.commands.height_bank=bank
    cfg.commands.ranges.lin_vel_x=[-4.,4.]
    cfg.commands.ranges.ang_vel_yaw=[-4.,4.]
    cfg.commands.ranges.height=[min(h for v,w,h in bank),max(h for v,w,h in bank)]
    cfg.commands.training_profile='height_course_v1'
    cfg.commands.hold_command_until_reset=True
    switch=s.get('height_switch_interval',0.)
    if switch not in (0.,5.) or (switch and s['stage']!='micro'):
        raise ValueError('Height switching currently reviewed only at micro, every 5s')
    cfg.commands.height_switch_interval=switch
    gain=s.get('height_reward_gain',1)
    if gain not in (1,4):raise ValueError('Reviewed height reward gains are 1 and 4')
    cfg.rewards.scales.base_height=float(gain)
    if gain != 1:
        # Otherwise legacy clip_single_reward=1 flattens the scaled reward near target.
        cfg.rewards.unclipped_reward_names=tuple(sorted(set(
            getattr(cfg.rewards,'unclipped_reward_names',()))|{'base_height'}))
    manifest.pop('exact_command_fractions',None)
    manifest.update(name='HEIGHT_COURSE',profile='height_course_v1',height_spec=s,height_bank=bank,
        ablation_variable='command bank only: variable base-root height and retained motion',
        command_semantics='Explicit vx/yaw/height slots held until reset; reset pose remains at .40m',
        scope='Different heights initially limited to low speed; full accepted motion retained at .40m')
    if s['stage'] in ('fixed35','fixed45','dual'):
        manifest.update(ablation_variable='command-only isolated fixed-height parking',
            scope='Diagnostic branch only: no motion retention training or motion acceptance; never replace motion baseline')
    if gain != 1:
        manifest.update(ablation_variable='height reward contribution x4, exemption from per-term clip only for base_height',
            height_reward_gain=gain,unclipped_reward_names=cfg.rewards.unclipped_reward_names,
            critic_transfer_note='Reward changed: old critic/Adam initialization retained, not exact same-objective continuation')
    if switch:
        manifest.update(ablation_variable='height command timing only: alternate .39/.41 every 5s in non-.40 slots',
            command_semantics='vx/yaw retained until reset; .40 motion slots never switch height; no reset at height switch')
    return manifest


def validate_source(path,resume,mode,cfg):
    s=spec();path=Path(path).resolve()
    if not resume or mode!='full' or path!=Path(s['source_checkpoint']).resolve():
        raise ValueError('Height curriculum requires exact full-resume source')
    sha=hashlib.sha256(path.read_bytes()).hexdigest()
    if sha!=s['source_sha256']:raise ValueError('Source changed')
    old=json.loads((path.parent/'policy_experiment.json').read_text())
    if old['profile'] not in {'motion_goal_v1','height_course_v1'}:raise ValueError('Unexpected recipe')
    for key,value in old['reward_scales'].items():
        if key=='base_height' and s.get('height_reward_gain',1)==4:
            if value not in (1.,4.) or cfg.rewards.scales.base_height!=4.:
                raise ValueError('Unexpected height reward migration')
            continue
        if getattr(cfg.rewards.scales,key,0.)!=value:raise ValueError('Reward mismatch: '+key)
    return {'source_checkpoint':str(path),'source_checkpoint_sha256':sha}
