"""Bucket IDs, values and material writes retain the original RNG sequence."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from wheel_legged_gym.adapters.isaacgym.shape_properties import ShapeRandomizationState, randomize_shape_properties


@pytest.mark.parametrize('friction,restitution', [(False,False),(True,False),(False,True),(True,True)])
def test_shape_randomization_matches_original(friction, restitution):
    spec=importlib.util.spec_from_file_location('old_shapes',Path(__file__).parent/'fixtures/shape_properties_reference.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    cfg=NS(randomize_friction=friction,friction_range=[.2,1.1],
           randomize_restitution=restitution,restitution_range=[.0,.5])
    old=module.OriginalShapes();old.cfg=NS(domain_rand=cfg);old.num_envs=5;old.device='cpu'
    old.friction_coef=torch.zeros(5);old.restitution_coef=torch.zeros(5)
    state=ShapeRandomizationState(torch.zeros(5),torch.zeros(5))
    outputs=[];rng=[]
    for legacy in [True,False]:
        torch.manual_seed(37);rows=[]
        for i in range(5):
            props=[NS(friction=.8,restitution=.1) for _ in range(3)]
            if legacy:result=old._process_rigid_shape_props(props,i)
            else:result=randomize_shape_properties(props,i,config=cfg,num_envs=5,device='cpu',state=state)
            assert result is props
            rows.append([(float(p.friction),float(p.restitution)) for p in props])
        outputs.append(rows);rng.append(torch.get_rng_state())
    assert outputs[0]==outputs[1]
    assert torch.equal(old.friction_coef,state.friction)
    assert torch.equal(old.restitution_coef,state.restitution)
    assert torch.equal(rng[0],rng[1])
