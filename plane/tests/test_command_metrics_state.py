"""Named command buffers must retain reset aliases without sharing task storage."""
from dataclasses import fields
import torch

from wheel_legged_gym.domain.commands.metrics_state import CommandMetrics, create_command_metrics


def test_named_buffers_reset_aliases_and_instance_isolation():
    before = torch.get_rng_state()
    first = create_command_metrics(3, 'cpu')
    second = create_command_metrics(3, 'cpu')
    assert torch.equal(before, torch.get_rng_state())
    buffers = first.buffers()
    assert len(buffers) == len(fields(CommandMetrics)) == 25
    assert len({buffer.data_ptr() for buffer in buffers}) == 25
    for field, buffer in zip(fields(CommandMetrics), buffers):
        assert buffer is getattr(first, field.name)
        assert buffer.shape == (3,) and buffer.dtype == torch.float32
        assert buffer.device.type == 'cpu' and not buffer.requires_grad
        buffer.fill_(2)
        assert not getattr(second, field.name).any()
    for buffer in buffers:
        buffer[1] = 0
    assert first.count.tolist() == [2., 0., 2.]
    assert first.action_clip_sum.tolist() == [2., 0., 2.]
