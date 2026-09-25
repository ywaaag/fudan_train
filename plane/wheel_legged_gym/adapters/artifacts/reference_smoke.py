import json

from pathlib import Path

import torch

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def verify(source,smoke_dir,fraction):
    source=Path(source);smoke_dir=Path(smoke_dir)
    iteration=int(source.stem.split('_')[-1])
    a=torch.load(source,map_location='cpu',weights_only=False)
    b=torch.load(smoke_dir/f'model_{iteration+1}.pt',map_location='cpu',weights_only=False)
    events=EventAccumulator(str(smoke_dir));events.Reload()
    observed=events.Scalars('Loss/policy_reference_fraction')[-1].value
    checks={'unchanged_deployment_state_keys':a['model_state_dict'].keys()==b['model_state_dict'].keys(),
        'encoder_unchanged':all(torch.equal(v,b['model_state_dict'][k]) for k,v in a['model_state_dict'].items() if k.startswith('encoder.')),
        'actor_updated':any(not torch.equal(v,b['model_state_dict'][k]) for k,v in a['model_state_dict'].items() if k.startswith('actor.')),
        'retention_fraction_correct':abs(observed-(1-fraction))<1e-6}
    if not all(checks.values()):raise RuntimeError('Reference smoke failed: '+str(checks))
    return dict(checks,observed_reference_fraction=observed,checkpoint=str(smoke_dir/f'model_{iteration+1}.pt'))

