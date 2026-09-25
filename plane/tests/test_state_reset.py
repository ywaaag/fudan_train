"""Indexed resets preserve unselected rows and submit shared tensors to Gym."""
from types import SimpleNamespace as NS

import pytest
import torch
from isaacgym.torch_utils import torch_rand_float

from wheel_legged_gym.adapters.isaacgym import state_reset


@pytest.mark.parametrize('custom', [False, True])
def test_indexed_physical_reset_and_rng(monkeypatch, custom):
    monkeypatch.setattr(state_reset.gymtorch,'unwrap_tensor',lambda value:value)
    calls=[]
    gym=NS(set_dof_state_tensor_indexed=lambda *args:calls.append(args),
           set_actor_root_state_tensor_indexed=lambda *args:calls.append(args))
    ids=torch.tensor([0,2]);dofs=torch.ones(3,6,2);defaults=torch.arange(18).view(3,6).float()
    state_reset.reset_dofs(ids,gym=gym,sim='sim',dof_position=dofs[:,:,0],
                           dof_velocity=dofs[:,:,1],default_position=defaults,dof_state=dofs)
    assert torch.equal(dofs[ids,:,0],defaults[ids]) and not dofs[ids,:,1].any()
    assert (dofs[1]==1).all() and calls[0][1] is dofs
    assert calls[0][2].dtype==torch.int32 and calls[0][3]==2
    root=torch.ones(3,13);initial=torch.arange(13).float();origins=torch.arange(9).view(3,3).float()
    torch.manual_seed(31)
    expected=root.clone();expected[ids]=initial;expected[ids,:3]+=origins[ids]
    if custom:expected[ids,:2]+=torch_rand_float(-1.,1.,(2,2),device='cpu')
    expected[ids,7:13]=torch_rand_float(-.5,.5,(2,6),device='cpu')
    expected_rng=torch.get_rng_state()
    torch.manual_seed(31)
    state_reset.reset_root_states(ids,gym=gym,sim='sim',root_states=root,
        initial_state=initial,origins=origins,custom_origins=custom,device='cpu')
    assert torch.equal(root,expected) and torch.equal(torch.get_rng_state(),expected_rng)
    assert calls[1][1] is root and calls[1][2].tolist()==[0,2]
