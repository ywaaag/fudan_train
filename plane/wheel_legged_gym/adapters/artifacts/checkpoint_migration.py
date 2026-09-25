"""Verify loaded checkpoint state and restore reviewed legacy migration fields."""

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


