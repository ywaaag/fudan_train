"""Shared constants cannot leak edits between experiments or supervisors."""
import pytest

from wheel_legged_gym.experiments.recipes.fudan_stand import FUDAN_SCALES
from wheel_legged_gym.workflows.completion import TERMINAL


def test_reward_table_values_order_and_copy_isolation():
    expected = {
        'tracking_lin_vel':1., 'tracking_lin_vel_enhance':0.,
        'tracking_ang_vel':1., 'tracking_ang_vel_enhance':0.,
        'base_height':2., 'nominal_state':-1., 'lin_vel_z':-1.,
        'ang_vel_xy':-.2, 'orientation':-500., 'dof_vel':-.01,
        'dof_acc':-2.5e-7, 'torques':-.0001, 'action_rate':-.01,
        'action_smooth':-.01, 'collision':-1., 'dof_pos_limits':-1.,
        'zero_base_velocity':-1., 'zero_wheel_velocity':-1.,
    }
    assert list(FUDAN_SCALES.items()) == list(expected.items())
    with pytest.raises(TypeError): FUDAN_SCALES['orientation'] = 0.
    manifest = dict(FUDAN_SCALES)
    manifest['orientation'] = 0.
    assert FUDAN_SCALES['orientation'] == -500.


def test_terminal_states_are_frozen():
    assert isinstance(TERMINAL, frozenset)
    assert len(TERMINAL) == 11
    assert 'stopped' in TERMINAL and 'training' not in TERMINAL
    with pytest.raises(AttributeError): TERMINAL.add('training')
