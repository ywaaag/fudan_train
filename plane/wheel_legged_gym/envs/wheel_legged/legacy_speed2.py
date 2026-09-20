"""Advance verified legacy-anchor policy to 1.5/2m/s with low-speed retention."""
import hashlib
import json
from pathlib import Path
from .legacy_anchors import apply_legacy_anchors


def apply_legacy_speed2(cfg,train):
    m=apply_legacy_anchors(cfg,train)
    cfg.commands.training_profile='legacy_speed2_v1'
    cfg.commands.method_v1_level=2
    cfg.commands.ranges.lin_vel_x=[-2.,2.]
    cfg.commands.translation_retention_anchors=(.5,1.,1.5,2.)
    m.update(name='LEGACY_SPEED2',profile='legacy_speed2_v1',
        ablation_variable='command level only: retain0/.1/.5/1 and add1.5/2, no yaw',
        translation_retention_anchors=[.5,1.,1.5,2.],
        exact_command_fractions={'0':.2,'-.1':.1,'+.1':.1,**{str(s*v):.075 for v in [.5,1.,1.5,2.] for s in [-1,1]}})
    return m


def validate_source(path,resume,mode,cfg):
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
