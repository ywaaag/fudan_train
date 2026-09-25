"""Explicit dispatch preserves function identity and selected-grid call order."""
from types import SimpleNamespace as NS

from isaacgym import terrain_utils
import numpy as np
import pytest

from wheel_legged_gym.adapters.isaacgym import terrain_generation as generation


@pytest.mark.parametrize('name', [
    'random_uniform_terrain', 'sloped_terrain', 'pyramid_sloped_terrain',
    'discrete_obstacles_terrain', 'wave_terrain', 'stairs_terrain',
    'pyramid_stairs_terrain', 'stepping_stones_terrain',
])
def test_isaac_generators_are_same_functions(name):
    assert generation.selected_generator('terrain_utils.'+name) is getattr(terrain_utils, name)


@pytest.mark.parametrize('name', ['gap_terrain', 'pit_terrain'])
def test_local_generators_are_same_functions(name):
    assert generation.selected_generator(name) is getattr(generation, name)


@pytest.mark.parametrize('name', ['missing', '__import__("os")', 'lambda terrain: None'])
def test_expressions_are_not_executed(name):
    with pytest.raises(ValueError, match='Unsupported selected terrain generator'):
        generation.selected_generator(name)


def test_selected_pit_grid_keeps_legacy_configuration_mutation_and_output():
    class Arguments(dict):
        terrain_kwargs = {'depth': .1, 'platform_size': .4}
    args = Arguments(type='pit_terrain')
    terrain = generation.Terrain.__new__(generation.Terrain)
    terrain.cfg = NS(terrain_kwargs=args, num_sub_terrains=2, num_rows=1, num_cols=2)
    terrain.width_per_env_pixels = 20
    terrain.vertical_scale = .01
    terrain.horizontal_scale = .05
    seen = []
    terrain.add_terrain_to_map = lambda tile, row, column: seen.append((tile, row, column))
    terrain.selected_terrain()
    assert 'type' not in args
    assert [(r,c) for _,r,c in seen] == [(0,0),(0,1)]
    expected = terrain_utils.SubTerrain('terrain', width=20, length=20,
                                         vertical_scale=.01, horizontal_scale=.05)
    generation.pit_terrain(expected, **args.terrain_kwargs)
    for tile, _, _ in seen:
        assert np.array_equal(tile.height_field_raw, expected.height_field_raw)
