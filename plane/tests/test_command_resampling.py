"""All strategy outputs and RNG state match the frozen environment implementation."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS

from isaacgym.torch_utils import torch_rand_float
import pytest
import torch

from wheel_legged_gym.contracts.wheel_legged_config import WheelLeggedCfg
from wheel_legged_gym.domain.commands.resampling import CommandBuffers, resample


spec = importlib.util.spec_from_file_location(
    'resampling_reference', Path(__file__).parent/'fixtures/resampling_reference.py')
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


def fixture(strategy, heading, dynamic):
    cfg = WheelLeggedCfg()
    cfg.commands.sampling_strategy = strategy
    cfg.commands.heading_command = heading
    cfg.commands.start_stop_ramp_seconds = .5 if dynamic else 0.
    cfg.commands.fixed_bank = [[1., 0.], [-1., 0.]]
    cfg.commands.height_bank = [[1., 0., .3], [-1., 0., .45]]
    cfg.commands.training_phase = 'stand'
    commands = torch.ones(8, 4)
    calls = []
    def reset(buffer, ids):
        calls.append(buffer.clone())
        buffer[ids, 0] = torch.rand(len(ids))
    env = NS(cfg=cfg, commands=commands, device='cpu',
             command_sample_mode=torch.full((8,), -9),
             method_v1_segment_counter=torch.arange(8),
             command_ranges={
                 'lin_vel_x': torch.tensor([[-4., 4.]]*8),
                 'ang_vel_yaw': torch.tensor([[-4., 4.]]*8),
                 'height': torch.tensor([[.3, .45]]*8), 'heading': [-3., 3.],
             }, _get_start_stop_scheduler=lambda: NS(reset=reset))
    return env, calls, reset


@pytest.mark.parametrize('strategy', ['uniform', 'method_v1', 'fixed_bank', 'height_bank', 'zero_reverse_mixture'])
@pytest.mark.parametrize('heading', [False, True])
@pytest.mark.parametrize('dynamic', [False, True])
def test_resampling_matches_old_buffers_rng_and_reset_order(strategy, heading, dynamic):
    ids = torch.tensor([0, 2, 5, 7])
    old, old_calls, _ = fixture(strategy, heading, dynamic)
    new, new_calls, reset = fixture(strategy, heading, dynamic)
    torch.manual_seed(19)
    reference._resample_commands(old, ids)
    old_rng = torch.get_rng_state()
    torch.manual_seed(19)
    resample(ids, state=CommandBuffers(new.commands, new.command_sample_mode,
                                      new.method_v1_segment_counter),
             ranges=new.command_ranges, config=new.cfg.commands, device=new.device,
             reset_start_stop=reset, sample_heading=torch_rand_float)
    assert torch.equal(torch.get_rng_state(), old_rng)
    for name in ('commands', 'command_sample_mode', 'method_v1_segment_counter'):
        assert torch.equal(getattr(old, name), getattr(new, name)), name
    assert len(old_calls)==len(new_calls)==int(dynamic)
    for a, b in zip(old_calls, new_calls):
        assert torch.equal(a, b)
    assert torch.equal(new.commands[[1, 3, 4, 6]], torch.ones(4, 4))


def test_unknown_strategy_fails_before_mutation_or_rng():
    env, calls, reset = fixture('unknown', True, True)
    before = torch.get_rng_state()
    with pytest.raises(ValueError, match='unknown command sampling strategy'):
        resample(torch.tensor([0]), state=CommandBuffers(env.commands, env.command_sample_mode,
                 env.method_v1_segment_counter), ranges=env.command_ranges,
                 config=env.cfg.commands, device=env.device, reset_start_stop=reset, sample_heading=torch_rand_float)
    assert not calls and torch.equal(before, torch.get_rng_state())
    assert torch.equal(env.commands, torch.ones(8, 4))
