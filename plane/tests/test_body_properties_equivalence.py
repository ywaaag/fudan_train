"""All randomization flag combinations retain body values and RNG streams."""
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import itertools

import numpy as np
import pytest
import torch

from wheel_legged_gym.adapters.isaacgym.body_properties import BodyRandomizationState, randomize_body_properties


@pytest.mark.parametrize('mass,com,inertia', list(itertools.product([False,True], repeat=3)))
def test_body_randomization_matches_original(mass, com, inertia):
    spec = importlib.util.spec_from_file_location('old_body', Path(__file__).parent/'fixtures/body_properties_reference.py')
    old_module = importlib.util.module_from_spec(spec); spec.loader.exec_module(old_module)
    cfg = NS(randomize_base_mass=mass, added_mass_range=[-.1,.3], randomize_base_com=com,
             rand_com_vec=[.01,.02,.03], randomize_inertia=inertia, randomize_inertia_range=[.8,1.2])
    old = old_module.OriginalBodyProperties()
    old.cfg, old.num_envs, old.device = NS(domain_rand=cfg), 3, 'cpu'
    old.base_mass, old.base_com = torch.zeros(3), torch.zeros(3,3)
    state = BodyRandomizationState(torch.zeros(3),torch.zeros(3,3))
    props = [NS(mass=2.,com=NS(x=.1,y=.2,z=.3),
                inertia=NS(x=NS(x=.4),y=NS(y=.5),z=NS(z=.6))) for _ in range(2)]
    outputs=[]; rng=[]
    for legacy in (True,False):
        torch.manual_seed(43);np.random.seed(43)
        rows=[]
        for index in range(3):
            bodies=copy.deepcopy(props)
            if legacy:old._process_rigid_body_props(bodies,index)
            else:randomize_body_properties(bodies,index,config=cfg,num_envs=3,device='cpu',state=state)
            rows.append([[float(b.mass),float(b.com.x),float(b.com.y),float(b.com.z),
                          b.inertia.x.x,b.inertia.y.y,b.inertia.z.z] for b in bodies])
        outputs.append(rows);rng.append((torch.get_rng_state(),np.random.get_state()))
    assert outputs[0]==outputs[1]
    assert torch.equal(old.base_mass,state.mass) and torch.equal(old.base_com,state.com)
    if mass:assert torch.equal(old.base_add_mass,state.added_mass)
    assert torch.equal(rng[0][0],rng[1][0])
    assert np.array_equal(rng[0][1][1],rng[1][1][1])
    assert rng[0][1][2:]==rng[1][1][2:]
