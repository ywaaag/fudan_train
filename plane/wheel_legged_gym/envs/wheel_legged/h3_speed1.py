"""One controlled speed-stage increase with old endpoints retained."""
import hashlib
import json
from pathlib import Path
from .low_speed import apply_low_speed


def apply_h3_speed1(cfg, train):
    manifest = apply_low_speed(cfg, train)
    cfg.commands.training_profile = 'h3_speed1_v1'
    cfg.commands.method_v1_level = 1
    cfg.commands.ranges.lin_vel_x = [-1., 1.]
    cfg.commands.translation_retention_anchors = (.5, 1.)
    cfg.commands.mixture_endpoint_linear_anchors = (-1., -.5, .5, 1.)
    manifest.update(name='H3_SPEED1',profile='h3_speed1_v1',level=1,command_limits=(1.,0.),
        resume_mode='full',resume_semantics='Full network/std/Adam continuation; unchanged rewards, expanded command distribution',
        translation_retention_anchors=[.5,1.],
        exact_command_fractions={'0':.2,'-0.1':.1,'+0.1':.1,'-0.5':.15,'+0.5':.15,'-1':.15,'+1':.15})
    return manifest


def validate_source(checkpoint, resume, mode, manifest):
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


def verify_full_resume(runner, checkpoint, expected_iteration=100):
    import torch
    source=torch.load(checkpoint,map_location=runner.device)
    for k,v in runner.alg.actor_critic.state_dict().items():
        if not torch.equal(v,source['model_state_dict'][k].to(v.device)):
            raise ValueError('Full resume model mismatch: '+k)
    def equal(a,b):
        if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b.to(a.device))
        if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
        if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
        return a==b
    if not equal(runner.alg.optimizer.state_dict(),source['optimizer_state_dict']):
        raise ValueError('Adam state was not retained')
    if not equal(runner.alg.extra_optimizer.state_dict(),source['extra_optimizer_state_dict']):
        raise ValueError('Encoder Adam state was not retained')
    if runner.current_learning_iteration!=expected_iteration:raise ValueError('Unexpected resume iteration')
    return {'all_model_tensors_exact':True,'both_optimizer_states_exact':True,'initial_iteration':expected_iteration,
            'initial_action_std':runner.alg.actor_critic.std.detach().cpu().tolist()}
