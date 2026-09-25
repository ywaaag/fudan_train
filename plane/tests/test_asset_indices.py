"""Asset lookup order, policy checks and optional fields match original code."""
from dataclasses import fields
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS

import isaacgym  # Legacy runtime requires simulator imports before Torch.
import pytest
import torch

from wheel_legged_gym.adapters.isaacgym.asset_indices import build_asset_indices

spec = importlib.util.spec_from_file_location(
    'asset_indices_reference', Path(__file__).parent/'fixtures/asset_indices_reference.py')
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


@pytest.mark.parametrize('method', [False, True])
@pytest.mark.parametrize('geometry', [False, True])
@pytest.mark.parametrize('stand', [False, True])
@pytest.mark.parametrize('existing', [False, True])
@pytest.mark.parametrize('missing', [None, 'dof', 'wheel', 'landmark'])
def test_indices_and_failure_lookup_order(method, geometry, stand, existing, missing):
    bodies = ['base', 'left_leg_0_link', 'right_leg_0_link', 'left_leg_1_link',
              'right_leg_1_link', 'left_wheel_link', 'right_wheel_link']
    dofs = ['left_leg_0', 'left_leg_1', 'left_wheel', 'right_leg_0', 'right_leg_1', 'right_wheel']
    if missing=='dof': dofs.remove('right_wheel')
    if missing=='wheel': bodies.remove('right_wheel_link')
    if missing=='landmark': bodies.remove('right_leg_1_link')
    feet = [name for name in bodies if 'wheel' in name]
    penalized = ['base', 'base']  # Repeated configured matches must stay repeated.
    terminated = ['base']
    rewards = NS(reward_pipeline='normalized_v1' if method else 'legacy_v0',
                 straight_bilateral_geometry=geometry)
    commands = NS(training_profile='fudan_stand_v1' if stand else '')
    traces = [[], []]
    def gym(index):
        def lookup(env, actor, name):
            traces[index].append((env, actor, name))
            return bodies.index(name)+10  # Distinguish handles from name-list indices.
        return NS(find_actor_rigid_body_handle=lookup)
    previous = torch.tensor([91, 92]) if existing else None
    old = NS(gym=gym(0), envs=['env'], actor_handles=['actor'], dof_names=dofs,
             device='cpu', cfg=NS(rewards=rewards, commands=commands))
    if existing: old.bilateral_body_indices=previous
    old_error = None
    try:
        reference.build(old, bodies, feet, penalized, terminated)
    except ValueError as error:
        old_error = error
    args = dict(gym=gym(1), environment='env', actor='actor', body_names=bodies,
                dof_names=dofs, feet_names=feet, penalized_contact_names=penalized,
                termination_contact_names=terminated, rewards=rewards, commands=commands,
                device='cpu', bilateral_indices_present=existing, bilateral_body_indices=previous)
    if old_error is not None:
        with pytest.raises(ValueError) as caught:
            build_asset_indices(**args)
        assert str(caught.value)==str(old_error)
    else:
        actual = build_asset_indices(**args)
        for field in fields(actual):
            value = getattr(actual, field.name)
            expected = getattr(old, field.name, None)
            if expected is None: assert value is None
            else: assert torch.equal(value, expected), field.name
        if method:
            assert actual.feet_indices.data_ptr()!=actual.wheel_body_indices.data_ptr()
        elif existing:
            assert actual.bilateral_body_indices is previous
    assert traces[0]==traces[1]
