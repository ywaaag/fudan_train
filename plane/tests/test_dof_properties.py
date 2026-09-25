"""DOF limits and first matching armature rule retain asset semantics."""
import numpy as np
import torch

from wheel_legged_gym.adapters.isaacgym.dof_properties import read_dof_limits, apply_armature


def test_limits_and_armature_rule_order(capsys):
    props = np.zeros(3, dtype=[(key, 'f4') for key in ('lower', 'upper', 'velocity', 'effort', 'armature')])
    props['lower'] = [-2, 1, -4]
    props['upper'] = [2, 5, 0]
    props['velocity'] = [10, 20, 30]
    props['effort'] = [40, 41, 42]
    props['armature'] = 9
    original = props.copy()
    position, velocity, torque = read_dof_limits(props, num_dof=3, device='cpu', soft_position_limit=.5)
    assert torch.equal(position, torch.tensor([[-1.,1.],[2.,4.],[-3.,-1.]]))
    assert velocity.tolist() == [10,20,30] and torque.tolist() == [40,41,42]
    assert np.array_equal(original, props)
    result = apply_armature(props, ['left_leg', 'right_wheel', 'unknown'],
                            {'leg': .1, 'left': .2, 'wheel': .3})
    assert result is props
    np.testing.assert_array_equal(props['armature'], np.array([.1,.3,0.], dtype='f4'))
    assert len(capsys.readouterr().out.splitlines()) == 2
