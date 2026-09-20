"""Keep speed1 endpoints while increasing exact-zero exposure, no reward edits."""
import hashlib
import json
from pathlib import Path
from .h3_speed1 import apply_h3_speed1


def apply_stop_retention(cfg,train):
    m=apply_h3_speed1(cfg,train)
    cfg.commands.zero_retention=True
    cfg.commands.training_profile='explore_stop_retention_v1'
    cfg.commands.mixture_zero_fraction=.4
    cfg.commands.mixture_small_fraction=0.
    m.update(name='EXPLORE_STOP_RETENTION',profile='explore_stop_retention_v1',
        encoder_action_anchor_coef=1.,freeze_encoder_updates=False,command_diagnostics=True,
        exact_command_fractions={'0':.4,'-.5':.15,'+.5':.15,'-1':.15,'+1':.15},
        ablation_variable='sampling only: transfer +/-0.1 slots to exact zero; keep endpoint exposure',
        source_acceptance='experimental near-pass model500; not a promoted or higher-speed model')
    return m


def validate_source(checkpoint,resume,mode,manifest):
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
