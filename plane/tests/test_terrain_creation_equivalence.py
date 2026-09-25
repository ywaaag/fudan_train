"""Capture original/new simulator calls without allocating a simulation."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS

from isaacgym import gymapi
import numpy as np
import pytest
import torch

from wheel_legged_gym.adapters.isaacgym import terrain_creation


class RecordingGym:
    def __init__(self):
        self.calls = []

    def add_ground(self, sim, params):
        self.calls.append(('ground', sim, params))

    def add_heightfield(self, sim, samples, params):
        self.calls.append(('heightfield', sim, samples.copy(), params))

    def add_triangle_mesh(self, sim, vertices, triangles, params):
        self.calls.append(('trimesh', sim, vertices.copy(), triangles.copy(), params))


@pytest.mark.parametrize('kind', ['ground_plane', 'heightfield', 'trimesh'])
def test_creation_calls_and_height_tensor_are_identical(kind):
    path = Path(__file__).parent / 'fixtures/terrain_creation_reference.py'
    spec = importlib.util.spec_from_file_location('original_terrain', path)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    reference.gymapi, reference.torch = gymapi, torch
    original = reference.OriginalTerrain()
    original.gym, original.sim, original.device = RecordingGym(), 'simulation', 'cpu'
    material = NS(static_friction=0.7, dynamic_friction=0.4, restitution=0.15)
    original.cfg = NS(terrain=material)
    original.terrain = NS(
        cfg=NS(horizontal_scale=0.13, vertical_scale=0.007, border_size=2.5),
        tot_rows=3, tot_cols=4, heightsamples=np.arange(12, dtype=np.int16).reshape(3, 4),
        vertices=np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]], dtype=np.float32),
        triangles=np.array([[0, 1, 2]], dtype=np.uint32),
    )
    getattr(original, '_create_' + kind)()
    recorder = RecordingGym()
    args = [recorder, original.sim, material]
    if kind != 'ground_plane':
        args += [original.terrain, original.device]
    heights = getattr(terrain_creation, 'create_' + kind)(*args)
    before, after = original.gym.calls[0], recorder.calls[0]
    assert before[:2] == after[:2]
    for left, right in zip(before[2:-1], after[2:-1]):
        assert np.array_equal(left, right)
    a, b = before[-1], after[-1]
    fields = ['static_friction', 'dynamic_friction', 'restitution']
    if kind == 'heightfield':
        fields += ['column_scale', 'row_scale', 'vertical_scale', 'nbRows', 'nbColumns']
    if kind == 'trimesh':
        fields += ['nb_vertices', 'nb_triangles']
    for field in fields:
        assert getattr(a, field) == getattr(b, field)
    vector_a = a.normal if kind == 'ground_plane' else a.transform.p
    vector_b = b.normal if kind == 'ground_plane' else b.transform.p
    assert (vector_a.x, vector_a.y, vector_a.z) == (vector_b.x, vector_b.y, vector_b.z)
    if kind != 'ground_plane':
        assert torch.equal(original.height_samples, heights)
        assert original.height_samples.dtype == heights.dtype
