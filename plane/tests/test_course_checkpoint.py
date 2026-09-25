"""Checkpoint schema, migration rejection and shape compatibility."""
from types import SimpleNamespace

import pytest
import torch

from wheel_legged_gym.domain.commands.checkpoint_state import (
    method_state, restore_method_counter, staged_state, restored_stage,
)


def test_method_payload_and_counter_roundtrip():
    source = torch.tensor([1, 7, 11])
    payload = method_state(SimpleNamespace(training_phase='yaw', method_v1_level=3), 2, source)
    assert set(payload) == {'reward_pipeline', 'training_profile', 'training_phase',
                            'command_level', 'randomization_level', 'segment_counter'}
    assert payload['training_phase'] == 'yaw'
    assert payload['command_level'] == 3
    target = torch.zeros(3, dtype=torch.long)
    restore_method_counter(payload, target, 'cpu')
    assert torch.equal(source, target)


@pytest.mark.parametrize('payload', [None, {}, {'segment_counter': [4, 5]}])
def test_absent_or_wrong_shape_counter_preserves_current(payload):
    target = torch.tensor([1, 2, 3])
    restore_method_counter(payload, target, 'cpu')
    assert target.tolist() == [1, 2, 3]


def test_incompatible_pipeline_rejected_before_counter_update():
    target = torch.tensor([1])
    with pytest.raises(ValueError, match='incompatible'):
        restore_method_counter({'reward_pipeline': 'other', 'segment_counter': [4]}, target, 'cpu')
    assert target.item() == 1


def test_staged_payload_preserves_metric_identity_and_defaults():
    metrics = {'passed': False}
    payload = staged_state(2, 1, metrics)
    assert restored_stage(payload, [0, 1, 2]) == (2, payload['staged_command_curriculum'])
    assert payload['staged_command_curriculum']['last_metrics'] is metrics
    assert restored_stage({'staged_command_curriculum': {}}, [0]) == (0, {})
    assert restored_stage({}, [0]) is None
    with pytest.raises(ValueError, match='invalid command curriculum stage'):
        restored_stage(payload, [0])
