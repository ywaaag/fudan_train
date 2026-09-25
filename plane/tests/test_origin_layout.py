"""Terrain and flat layouts preserve buckets, bounds, dtype and RNG exactly."""
from dataclasses import fields
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from wheel_legged_gym.domain.geometry.origins import build_origin_layout


spec = importlib.util.spec_from_file_location(
    'origin_reference', Path(__file__).parent/'fixtures/origin_layout_reference.py')
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


@pytest.mark.parametrize('mesh', ['plane', None, 'heightfield', 'trimesh'])
@pytest.mark.parametrize('curriculum', [False, True])
@pytest.mark.parametrize('num_envs', [1, 7, 43])
def test_layout_matches_old_environment(mesh, curriculum, num_envs):
    cfg = NS(mesh_type=mesh, curriculum=curriculum, max_init_terrain_level=1,
             num_rows=4, num_cols=20, terrain_length=8., border_size=5.)
    origins = np.arange(4*20*3, dtype=np.float64).reshape(4, 20, 3)
    original = NS(num_envs=num_envs, device='cpu', terrain=NS(env_origins=origins),
                  cfg=NS(terrain=cfg, env=NS(env_spacing=3.)))
    torch.manual_seed(37)
    reference._get_env_origins(original)
    expected_rng = torch.get_rng_state()
    torch.manual_seed(37)
    actual = build_origin_layout(num_envs=num_envs, device='cpu', terrain_config=cfg,
                                 spacing=3., terrain_origin_array=origins)
    assert torch.equal(expected_rng, torch.get_rng_state())
    for field in fields(actual):
        value = getattr(actual, field.name)
        if not hasattr(original, field.name):
            assert value is None, field.name
        elif torch.is_tensor(value):
            expected = getattr(original, field.name)
            assert value.dtype==expected.dtype and torch.equal(value, expected), field.name
        else:
            assert value==getattr(original, field.name), field.name
