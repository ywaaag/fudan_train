"""Protect exact-zero stopping while retaining verified legacy speed2 motion."""
import hashlib, json
from pathlib import Path
from .legacy_speed2 import apply_legacy_speed2

def apply_legacy_speed2_stop(cfg, train):
    manifest=apply_legacy_speed2(cfg,train)
    cfg.commands.zero_retention=True
    cfg.commands.mixture_zero_fraction=.4
    cfg.commands.mixture_small_fraction=0.
    cfg.commands.training_profile='legacy_speed2_stop_v2'
    manifest.update(name='LEGACY_SPEED2_STOP',profile='legacy_speed2_stop_v2',
        exact_command_fractions={'0':.4,'-.5':.075,'+.5':.075,'-1':.075,'+1':.075,'-1.5':.075,'+1.5':.075,'-2':.075,'+2':.075},
        ablation_variable='sampling only: raise exact-zero from20 to40; preserve eight nonzero endpoint slots')
    return manifest

def validate_source(path,resume,mode,cfg):
    if not resume or mode!='full': raise ValueError('Speed2 stop requires full resume')
    path=Path(path).resolve(); record=json.loads((Path(__file__).resolve().parents[4]/'docs/data/legacy_speed2_source_20260919.json').read_text())
    sha=hashlib.sha256(path.read_bytes()).hexdigest()
    if sha!=record['sha256']: raise ValueError('Expected legacy speed2 model900')
    old=json.loads((path.parent/'policy_experiment.json').read_text())
    if old['name']!='LEGACY_SPEED2':raise ValueError('Unexpected source profile')
    for key,value in old['reward_scales'].items():
        if getattr(cfg.rewards.scales,key,0.)!=value:raise ValueError('Reward mismatch: '+key)
    return {'source_checkpoint':str(path),'source_checkpoint_sha256':sha}
