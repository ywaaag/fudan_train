"""Both tracking modes preserve accumulation values and callback ordering."""
from dataclasses import fields
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from wheel_legged_gym.domain.commands.metrics_state import create_command_metrics
from wheel_legged_gym.domain.commands.tracking_metrics import accumulate, summarize


@pytest.mark.parametrize('empty_counts', [False, True])
def test_reset_summary_matches_original_and_keeps_buffers(empty_counts):
    spec = importlib.util.spec_from_file_location(
        'old_summary', Path(__file__).parent/'fixtures/metric_summary_reference.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    metrics = create_command_metrics(4, 'cpu')
    original = NS()
    for index, field in enumerate(fields(metrics)):
        buffer = getattr(metrics, field.name)
        buffer[:] = torch.arange(4).float() + index
        if empty_counts and field.name.endswith('count'):
            buffer[:] = 0
        setattr(original, 'command_metric_'+field.name, buffer)
    before = [buffer.clone() for buffer in metrics.buffers()]
    ids = torch.tensor([0, 3])
    expected = module.summarize(original, ids)
    actual = summarize(metrics, ids)
    assert actual.keys()==expected.keys()
    for name in actual:
        assert torch.equal(actual[name], expected[name]), name
    for a, b in zip(before, metrics.buffers()):
        assert torch.equal(a, b)


@pytest.mark.parametrize('method', [False, True])
def test_repeated_accumulation_matches_original(method):
    spec = importlib.util.spec_from_file_location('old_metrics', Path(__file__).parent/'fixtures/tracking_metrics_reference.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    old = module.OriginalMetrics()
    a, b = create_command_metrics(4, 'cpu'), create_command_metrics(4, 'cpu')
    for field in fields(a):
        setattr(old, 'command_metric_' + field.name, getattr(a, field.name))
    old.commands = torch.tensor([[0.,0.,.4],[-1.,0.,.4],[1.,0.,.4],[0.,1.,.4]])
    old.base_lin_vel = torch.arange(12, dtype=torch.float).reshape(4,3)/10
    old.base_ang_vel = -old.base_lin_vel
    old.dof_vel = torch.arange(24, dtype=torch.float).reshape(4,6)/10
    old.torques = old.dof_vel
    old.torque_limits = torch.ones(6)
    old.wheel_contact_history = torch.tensor([[[True, False]]] * 4)
    old.cfg = NS(rewards=NS(zero_command_threshold=.05, wheel_radius=.06))
    old._method_v1 = method
    calls = []
    def zero():
        calls.append('zero')
        return torch.tensor([True,False,False,False])
    def wheel():
        calls.append('wheel')
        return {'residual_rms': torch.tensor([.1,.2,.3,.4])}
    old._zero_command_mask, old._method_wheel_terms = zero, wheel
    for _ in range(3):
        old._accumulate_command_tracking_metrics()
    expected_calls = calls.copy(); calls.clear()
    for _ in range(3):
        accumulate(b, commands=old.commands, linear_velocity=old.base_lin_vel,
            angular_velocity=old.base_ang_vel, dof_velocity=old.dof_vel,
            torques=old.torques, torque_limits=old.torque_limits,
            zero_threshold=.05, wheel_radius=.06, zero_mask=zero, method_wheel_terms=wheel,
            wheel_contact_history=old.wheel_contact_history, method_v1=method)
    assert calls == expected_calls
    for field in fields(a):
        assert torch.equal(getattr(a,field.name),getattr(b,field.name)),field.name
