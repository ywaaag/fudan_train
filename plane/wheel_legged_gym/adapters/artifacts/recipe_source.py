"""Checkpoint provenance checks with explicitly supplied recipe specifications."""
import hashlib
import json
from pathlib import Path

def validate_motion_source(path, resume, mode, cfg, *, spec):
    if not resume or mode != 'full': raise ValueError('Motion goal requires full resume')
    path = Path(path).resolve()
    if path != Path(spec['source_checkpoint']).resolve(): raise ValueError('Source path mismatch')
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if sha != spec['source_sha256']: raise ValueError('Source checkpoint changed')
    old = json.loads((path.parent/'policy_experiment.json').read_text())
    if old['profile'] not in {'legacy_speed2_stop_v2','legacy_yaw05_v1','motion_goal_v1'}:
        raise ValueError('Unreviewed source recipe')
    for key,value in old['reward_scales'].items():
        if spec.get('geometry_symmetry',False) and key in {'nominal_state','stand_bilateral_geometry'}:
            expected=0. if key=='nominal_state' else spec.get('geometry_weight',-.1)
            if getattr(cfg.rewards.scales,key)!=expected:raise ValueError('Unexpected geometry recipe')
            continue
        if getattr(cfg.rewards.scales,key,0.) != value: raise ValueError('Reward mismatch: '+key)
    return {'source_checkpoint':str(path),'source_checkpoint_sha256':sha}


def validate_height_source(path,resume,mode,cfg, *, spec):
    s=spec;path=Path(path).resolve()
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


def validate_turn_source(path, resume, mode, cfg, *, spec):
    path = Path(path).resolve()
    if not resume or mode != 'full' or path != Path(spec['source_checkpoint']).resolve():
        raise ValueError('Turn envelope requires exact R10200 full resume')
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if sha != spec['source_sha256']:
        raise ValueError('Turn source checksum mismatch')
    old = json.loads((path.parent / 'policy_experiment.json').read_text())
    if old.get('profile') != 'motion_goal_v1' or old.get('resume_verification', {}).get('initial_iteration') != 10000:
        raise ValueError('Unreviewed turn source manifest')
    for key, value in old['reward_scales'].items():
        if getattr(cfg.rewards.scales, key, 0.) != value:
            raise ValueError('Turn source reward mismatch: ' + key)
    return {'source_checkpoint': str(path), 'source_checkpoint_sha256': sha,
            'source_manifest': str(path.parent / 'policy_experiment.json')}
