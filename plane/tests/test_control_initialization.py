"""Match all randomization switches, gain precedence and original RNG consumption."""
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS

from isaacgym.torch_utils import torch_rand_float
import pytest
import torch

from wheel_legged_gym.domain.control.initialization import initialize


spec = importlib.util.spec_from_file_location(
    'control_initialization_reference', Path(__file__).parent/'fixtures/control_initialization_reference.py')
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


@pytest.mark.parametrize('mask', range(32))
def test_all_randomization_combinations_match_frozen_implementation(mask, capsys):
    switches = ['Kp', 'Kd', 'motor_torque', 'default_dof_pos', 'action_delay']
    randomization = NS(**{'randomize_'+name: bool(mask & (1 << i))
                         for i, name in enumerate(switches)})
    randomization.randomize_Kp_range = [.8, 1.2]
    randomization.randomize_Kd_range = [.7, 1.3]
    randomization.randomize_motor_torque_range = [.9, 1.1]
    randomization.randomize_default_dof_pos_range = [-.05, .05]
    randomization.delay_ms_range = [0., 20.]
    # 'leg_special' matches twice; the last key wins. 'unmatched' must warn.
    names = ['leg_special', 'wheel', 'unmatched']
    before = NS(num_envs=4, num_dof=3, num_dofs=3, dof_names=names, device='cpu',
                cfg=NS(init_state=NS(default_joint_angles=dict(zip(names, [.1, .2, .3]))),
                       control=NS(stiffness={'leg': 20., 'leg_special': 30., 'wheel': 0.},
                                  damping={'leg': 1., 'leg_special': 2., 'wheel': 1.}, control_type='P'),
                       domain_rand=randomization), sim_params=NS(dt=.005),
                p_gains=torch.zeros(4, 3), d_gains=torch.zeros(4, 3),
                torques_scale=torch.ones(4, 3), action_delay_idx=torch.zeros(4, dtype=torch.long))
    after = copy.deepcopy(before)
    torch.manual_seed(23)
    reference.initialize(before)
    expected_rng = torch.get_rng_state()
    expected_output = capsys.readouterr().out
    calls = []
    def sample(*args, **kwargs):
        calls.append(args[:2])
        return torch_rand_float(*args, **kwargs)
    torch.manual_seed(23)
    original_delay = after.action_delay_idx
    result = initialize(num_envs=after.num_envs, dof_count=after.num_dof,
                        named_dof_count=after.num_dofs, dof_names=names,
                        default_angles=after.cfg.init_state.default_joint_angles,
                        control=after.cfg.control, randomization=randomization,
                        physics_dt=.005, device='cpu', p_gains=after.p_gains,
                        d_gains=after.d_gains, torque_scales=after.torques_scale,
                        delay_indices=original_delay, sample_uniform=sample, report=print)
    assert capsys.readouterr().out == expected_output
    assert torch.equal(expected_rng, torch.get_rng_state())
    for field in ('p_gains', 'd_gains', 'torques_scale'):
        assert torch.equal(getattr(before, field), getattr(after, field)), field
    for old, new in [('raw_default_dof_pos', 'raw_default_position'),
                     ('default_dof_pos', 'default_position'), ('action_delay_idx', 'delay_indices')]:
        assert torch.equal(getattr(before, old), getattr(result, new)), old
    assert (result.delay_indices is original_delay) == (not randomization.randomize_action_delay)
    expected_ranges = [(.8, 1.2), (.7, 1.3), (.9, 1.1), (-.05, .05), (0., 4.)]
    assert calls == [value for i, value in enumerate(expected_ranges) if mask & (1 << i)]
