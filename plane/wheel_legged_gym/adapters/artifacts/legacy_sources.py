"""Historical source gates; preserve audited evidence and rejection semantics."""
import hashlib
import json
from pathlib import Path

def validate_stop_retention_source(checkpoint,resume,mode,manifest):
    if not resume or mode!='full':raise ValueError('Stop retention requires full resume')
    p=Path(checkpoint)
    evidence=json.loads((Path(__file__).resolve().parents[4]/'docs/data/wheel_explore_20260919.json').read_text())
    sha=hashlib.sha256(p.read_bytes()).hexdigest()
    if sha!=evidence['checkpoint_sha256']['500']:raise ValueError('Expected audited exploration model500')
    old=json.loads((p.parent/'policy_experiment.json').read_text())
    for k in ['reward_scales','reward_parameters','optimizer','encoder_action_anchor_coef']:
        if old[k]!=manifest[k]:raise ValueError('Non-sampling configuration mismatch: '+k)
    if old['randomization_level']!=1 or old['symmetry_loss_coef']!=.01:raise ValueError('Source configuration mismatch')
    return {'source_checkpoint':str(p.resolve()),'source_checkpoint_sha256':sha}


def validate_legacy_anchors_source(path,resume,mode,cfg):
    if not resume or mode!='full':raise ValueError('Legacy anchors requires full resume')
    path=Path(path).resolve();root=Path(__file__).resolve().parents[4]
    record=json.loads((root/'docs/data/legacy_anchor_source_20260919.json').read_text())
    sha=hashlib.sha256(path.read_bytes()).hexdigest()
    if sha!=record['sha256']:raise ValueError('Expected evaluated legacy500 source')
    old=json.loads((path.parent/'policy_experiment.json').read_text())
    if old['name']!='LEGACY_URDF':raise ValueError('Wrong source profile')
    for k,v in old['reward_scales'].items():
        if getattr(cfg.rewards.scales,k,0.)!=v:raise ValueError('Reward mismatch: '+k)
    return {'source_checkpoint':str(path),'source_checkpoint_sha256':sha,
            'reward_scales':old['reward_scales']}


def validate_legacy_speed2_source(path,resume,mode,cfg):
    if not resume or mode!='full':raise ValueError('Speed2 requires full-state resume')
    path=Path(path).resolve()
    record=json.loads((Path(__file__).resolve().parents[4]/'docs/data/legacy_speed1_passed_20260919.json').read_text())
    sha=hashlib.sha256(path.read_bytes()).hexdigest()
    if not record['passed'] or sha!=record['sha256']:raise ValueError('Require verified speed1 model700')
    old=json.loads((path.parent/'policy_experiment.json').read_text())
    if old['name']!='LEGACY_ANCHORS':raise ValueError('Unexpected source profile')
    for k,v in old['reward_scales'].items():
        if getattr(cfg.rewards.scales,k,0.)!=v:raise ValueError('Reward changed: '+k)
    return {'source_checkpoint':str(path),'source_checkpoint_sha256':sha,'reward_scales':old['reward_scales']}


def validate_legacy_speed2_stop_source(path,resume,mode,cfg):
    if not resume or mode!='full': raise ValueError('Speed2 stop requires full resume')
    path=Path(path).resolve(); record=json.loads((Path(__file__).resolve().parents[4]/'docs/data/legacy_speed2_source_20260919.json').read_text())
    sha=hashlib.sha256(path.read_bytes()).hexdigest()
    if sha!=record['sha256']: raise ValueError('Expected legacy speed2 model900')
    old=json.loads((path.parent/'policy_experiment.json').read_text())
    if old['name']!='LEGACY_SPEED2':raise ValueError('Unexpected source profile')
    for key,value in old['reward_scales'].items():
        if getattr(cfg.rewards.scales,key,0.)!=value:raise ValueError('Reward mismatch: '+key)
    return {'source_checkpoint':str(path),'source_checkpoint_sha256':sha}


def validate_legacy_yaw_source(path, resume, mode, cfg):
    if not resume or mode != 'full':
        raise ValueError('Yaw curriculum requires full resume')
    path = Path(path).resolve()
    record = json.loads((Path(__file__).resolve().parents[4] / 'docs/data/legacy_speed2_passed_20260920.json').read_text())
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if sha != record['sha256'] or record['passed'] != 27:
        raise ValueError('Expected independently accepted speed2 model1400')
    old = json.loads((path.parent / 'policy_experiment.json').read_text())
    if old['profile'] != 'legacy_speed2_stop_v2':
        raise ValueError('Unexpected source profile')
    for key, value in old['reward_scales'].items():
        if getattr(cfg.rewards.scales, key, 0.) != value:
            raise ValueError('Reward mismatch: ' + key)
    return {'source_checkpoint': str(path), 'source_checkpoint_sha256': sha}


def validate_h3_low_speed_source(checkpoint, resume, resume_mode):
    if not resume or resume_mode != 'policy':
        raise ValueError('H3_LOW_SPEED requires --resume --resume_mode=policy')
    checkpoint = Path(checkpoint).resolve()
    evidence = Path(__file__).resolve().parents[4]/'docs/data/policy_comparison_20260919.json'
    candidate = json.loads(evidence.read_text())['candidates']['legacy_h3']
    sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if sha != candidate['sha256']:
        raise ValueError('H3_LOW_SPEED source must match the evaluated H3 15800 SHA256')
    previous = json.loads((checkpoint.parent/'policy_experiment.json').read_text())
    if previous.get('name') != 'H3':
        raise ValueError('Expected H3 source manifest')
    return {'source_checkpoint':str(checkpoint), 'source_checkpoint_sha256':sha,
            'source_reward_scales':previous['reward_scales']}


def validate_h3_speed1_source(checkpoint, resume, mode, manifest):
    if not resume or mode != 'full':
        raise ValueError('H3_SPEED1 requires --resume --resume_mode=full')
    checkpoint=Path(checkpoint)
    evidence=json.loads((Path(__file__).resolve().parents[4]/'docs/data/h3_low_speed_20260919.json').read_text())
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=evidence['checkpoint_sha256']['100']:
        raise ValueError('H3_SPEED1 requires the audited low-speed model100')
    previous=json.loads((checkpoint.parent/'policy_experiment.json').read_text())
    if previous.get('name')!='H3_LOW_SPEED' or previous.get('randomization_level')!=1:
        raise ValueError('Unexpected source profile/randomization')
    for key in ['reward_scales','reward_parameters','optimizer']:
        if previous.get(key)!=manifest[key]:raise ValueError('Source configuration mismatch: '+key)
    if previous.get('symmetry_loss_coef')!=.01:raise ValueError('Symmetry coefficient changed')
    return {'source_checkpoint':str(checkpoint.resolve()),'source_checkpoint_sha256':evidence['checkpoint_sha256']['100']}


