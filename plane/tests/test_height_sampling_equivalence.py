"""Preserve interpolation, clamping and historical subset/error behavior."""
from pathlib import Path
from types import SimpleNamespace as NS
import importlib.util
import pytest
import torch
from wheel_legged_gym.domain.geometry.height_sampling import sample_heights


@pytest.mark.parametrize('mesh', ['plane','none','heightfield','trimesh'])
@pytest.mark.parametrize('ids', [None,[0],torch.tensor([0,1])])
def test_sampler_matches_original(mesh,ids):
    spec=importlib.util.spec_from_file_location('old_heights',Path(__file__).parent/'fixtures/height_sampling_reference.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    old=module.OriginalHeightSampler()
    old.cfg=NS(terrain=NS(mesh_type=mesh));old.num_envs=2;old.num_height_points=4;old.device='cpu'
    old.base_quat=torch.tensor([[0.,0.,0.,1.],[0.,0.,.70710678,.70710678]])
    old.root_states=torch.zeros(2,13);old.root_states[1,0]=5.
    old.height_points=torch.tensor([[[-2.,-2.,0.],[0.,0.,0.],[1.,1.,0.],[9.,9.,0.]]]*2)
    old.height_samples=torch.arange(25).reshape(5,5)
    old.terrain=NS(cfg=NS(border_size=.1,horizontal_scale=.5,vertical_scale=.01))
    def migrated():
        return sample_heights(ids,mesh_type=mesh,num_envs=2,num_height_points=4,device='cpu',
            base_quat=old.base_quat,root_states=old.root_states,height_points=old.height_points,
            height_samples=old.height_samples,terrain_config=old.terrain.cfg)
    try:expected=old._get_heights(ids)
    except (RuntimeError,NameError) as error:
        with pytest.raises(type(error)) as actual:migrated()
        assert str(actual.value)==str(error)
    else:assert torch.equal(expected,migrated())
