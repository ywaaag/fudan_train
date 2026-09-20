"""Legacy-v1 branch continuation: change command curriculum, keep effective recipe."""
import json
import hashlib
from pathlib import Path
from .legacy_urdf import apply_legacy_urdf


def apply_legacy_anchors(cfg,train):
    m=apply_legacy_urdf(cfg,train)
    cfg.commands.curriculum=False
    cfg.commands.sampling_strategy='method_v1'
    cfg.commands.hold_command_until_reset=True
    cfg.commands.training_profile='legacy_anchors_v1'
    cfg.commands.training_phase='translate'
    cfg.commands.method_v1_level=1
    cfg.commands.ranges.lin_vel_x=[-1.,1.]
    cfg.commands.ranges.ang_vel_yaw=[0.,0.]
    cfg.commands.translation_retention_anchors=(.5,1.)
    cfg.commands.zero_retention=False
    cfg.commands.mixture_zero_fraction=.2
    cfg.commands.mixture_small_fraction=.2
    cfg.commands.mixture_reverse_fraction=.3
    cfg.commands.mixture_forward_fraction=.3
    m.update(name='LEGACY_ANCHORS',profile='legacy_anchors_v1',resume_mode='full',
        ablation_variable='command curriculum only: fixed translation anchors/zero, no yaw, no automatic promotion',
        exact_command_fractions={'0':.2,'-.1':.1,'+.1':.1,'-.5':.15,'+.5':.15,'-1':.15,'+1':.15},
        caveat='Preserves actual legacy_urdf_v1 hybrid recipe including zero penalties; not exact upstream reproduction')
    return m


def validate_source(path,resume,mode,cfg):
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
