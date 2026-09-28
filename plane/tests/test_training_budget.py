import json

import pytest

from wheel_legged_gym.app.training_budget import reserve, settle


def ledger(tmp_path, limit=10):
    path = tmp_path/'budget_ledger.json'
    path.write_text(json.dumps({'formal_limit_iterations':limit,
        'formal_used_iterations':0,'source_checkpoint_sha256':'source'}))
    return path


def test_reservation_is_locked_idempotently_settled_and_released(tmp_path):
    path = ledger(tmp_path)
    reserve(path,'a','run_a','source',7)
    with pytest.raises(ValueError, match='Insufficient'):
        reserve(path,'b','run_b','source',4)
    settle(path,'a',3,'evidence')
    settle(path,'a',3,'evidence')
    reserve(path,'b','run_b','source',7)
    settle(path,'b',7,'evidence')
    data=json.loads(path.read_text())
    assert data['formal_used_iterations']==10
    assert len(data['attempts'])==2
    with pytest.raises(ValueError, match='Conflicting'):
        settle(path,'a',4,'evidence')


def test_repeated_source_checkpoint_attempts_count_separately(tmp_path):
    path=ledger(tmp_path)
    reserve(path,'a','first','source',5)
    settle(path,'a',5,'first-status')
    reserve(path,'b','second','source',5)
    settle(path,'b',5,'second-status')
    assert json.loads(path.read_text())['formal_used_iterations']==10


def test_unknown_consumption_blocks_reallocation(tmp_path):
    path=ledger(tmp_path)
    reserve(path,'a','run_a','source',10)
    settle(path,'a',None,'missing-event',unknown=True)
    with pytest.raises(ValueError, match='Insufficient'):
        reserve(path,'b','run_b','source',1)
