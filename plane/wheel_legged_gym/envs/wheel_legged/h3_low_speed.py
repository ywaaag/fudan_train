"""Explicit H3 actor/encoder/std migration into normalized low-speed rewards."""
import hashlib
import json
from pathlib import Path

from .low_speed import apply_low_speed


def apply_h3_low_speed(env_cfg, train_cfg):
    manifest = apply_low_speed(env_cfg, train_cfg, 'LOW_SPEED')
    env_cfg.commands.training_profile = 'h3_low_speed_v1'
    manifest.update(name='H3_LOW_SPEED', profile='h3_low_speed_v1',
        resume_semantics='actor/encoder/std from verified H3; fresh critic, PPO/encoder optimizers, iteration and curriculum',
        resume_mode='policy', std_initialization='copy source six-channel std exactly',
        comparison_limit='Migration probe; not a causal single-variable comparison against historical full-state LOW_SPEED runs')
    return manifest


def validate_source(checkpoint, resume, resume_mode):
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


def verify_and_restore_std(runner, checkpoint):
    import torch
    source = torch.load(checkpoint, map_location=runner.device)['model_state_dict']
    model = runner.alg.actor_critic
    for key, value in model.state_dict().items():
        if key.startswith(('actor.', 'encoder.')):
            if key not in source or not torch.equal(value, source[key].to(value.device)):
                raise ValueError('Migrated actor/encoder differs: '+key)
    if runner.current_learning_iteration != 0 or runner.alg.optimizer.state:
        raise ValueError('Migration must start with fresh iteration and optimizer')
    if runner.alg.extra_optimizer is not None and runner.alg.extra_optimizer.state:
        raise ValueError('Migration must start with fresh encoder optimizer')
    std = source['std'].to(model.std.device)
    if std.shape != model.std.shape or not torch.isfinite(std).all() or not (std > 0).all():
        raise ValueError('Invalid source action standard deviation')
    if all(torch.equal(v, source[k].to(v.device)) for k,v in model.state_dict().items()
           if k.startswith('critic.')):
        raise ValueError('Critic unexpectedly identical to source')
    with torch.no_grad():
        model.std.copy_(std)
    return {'actor_encoder_exact_match':True, 'critic_reinitialized':True,
            'optimizer_state_entries':0, 'initial_iteration':0,
            'initial_action_std':std.cpu().tolist()}
