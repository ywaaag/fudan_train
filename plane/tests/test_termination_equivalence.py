"""Failure streaks, warmup, timeout and edge reset match the original environment."""
import ast
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from wheel_legged_gym.domain.termination import TerminationState, check_termination


@pytest.mark.parametrize('method', [False,True])
@pytest.mark.parametrize('terrain', ['plane','heightfield'])
@pytest.mark.parametrize('phase', ['stand','translate'])
def test_environment_termination_matches_frozen_method(method, terrain, phase):
    root=Path(__file__).parent
    spec=importlib.util.spec_from_file_location('original_termination',root/'fixtures/termination_reference.py')
    ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
    tree=ast.parse((root.parent/'wheel_legged_gym/envs/base/legged_robot.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='LeggedRobot')
    cls.bases=[];cls.body=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='check_termination']
    scope=dict(TerminationState=TerminationState,check_termination_state=check_termination,SimpleNamespace=NS)
    exec(compile(ast.Module(body=[cls],type_ignores=[]),'termination_delegate','exec'),scope)
    a=ref.OriginalTermination();a._method_v1=method;a.num_envs=4;a.device='cpu';a.dt=.01
    a.cfg=NS(rewards=NS(forbidden_contact_force_threshold=10.,contact_warmup_s=.02,
        forbidden_contact_grace_s=.02,wheel_loss_grace_s=.03),commands=NS(training_phase=phase),
        terrain=NS(mesh_type=terrain),env=NS(fail_to_terminal_time_s=.02))
    a.contact_forces=torch.zeros(4,2,3);a.contact_forces[0,0,0]=11.
    a.termination_contact_indices=torch.tensor([0]);a.projected_gravity=torch.tensor([[0.,0.,-1.]]*4)
    a.projected_gravity[1,2]=0.;a.episode_length_buf=torch.tensor([0,1,10,11]);a.max_episode_length=10
    a.wheel_contact_seen=torch.ones(4,dtype=torch.bool);a.wheel_contact_history=torch.zeros(4,1,2)
    a.base_position=torch.tensor([[0.,0.,0.],[0.,0.,0.],[9.5,0.,0.],[0.,-9.5,0.]])
    a.terrain_x_max=a.terrain_y_max=10.;a.terrain_x_min=a.terrain_y_min=-10.
    for key in ['forbidden_contact_streak','upright_streak','wheel_loss_streak','fail_buf']:
        setattr(a,key,torch.zeros(4))
    for key in ['time_out_buf','edge_reset_buf','reset_buf']:setattr(a,key,torch.zeros(4,dtype=torch.bool))
    b=scope['LeggedRobot']();b.__dict__.update(copy.deepcopy(a.__dict__))
    for _ in range(5):
        a.check_termination();b.check_termination()
        for key,value in a.__dict__.items():
            if torch.is_tensor(value):assert torch.equal(value,getattr(b,key)),key
        a.episode_length_buf+=1;b.episode_length_buf+=1
