"""Precheck thresholds and the application stop-on-regression boundary."""
import copy
import json

import pytest

from wheel_legged_gym.evaluation.closed_precheck import yaw_precheck_passed
from wheel_legged_gym.workflows.closed_review_report import render_candidate_review


def passing():
    return {'policy_sha256': 'sha', 'result': {'passed': True, 'completed_steps': 34000, 'failure': None},
            'measurements': {'samples': 20000, 'vx_mae': .05, 'yaw_mae': .1, 'height_mae': .03}}


@pytest.mark.parametrize('section,key,value', [
    ('result', 'passed', False), ('result', 'completed_steps', 33999),
    ('measurements', 'samples', 19999), ('measurements', 'vx_mae', .050001),
    ('measurements', 'yaw_mae', .100001), ('measurements', 'height_mae', .030001),
])
def test_each_gate_rejects(section, key, value):
    raw = passing()
    assert yaw_precheck_passed(raw, 'sha')
    raw[section][key] = value
    before = copy.deepcopy(raw)
    assert not yaw_precheck_passed(raw, 'sha')
    assert raw == before


def test_hash_mismatch_is_error():
    with pytest.raises(RuntimeError, match='Policy changed'):
        yaw_precheck_passed(passing(), 'other')


def test_regression_finishes_with_report_and_no_full_review(tmp_path, monkeypatch):
    from wheel_legged_gym.app import candidate_closed_review as app
    output = tmp_path/'plane/outputs'
    output.mkdir(parents=True)
    policy = tmp_path/'policy.onnx'
    policy.write_bytes(b'fake-policy-for-application-test')
    calls = []
    class Child:
        pid = 12345
        returncode = 0
        def __init__(self, cmd, **kwargs):
            calls.append(cmd)
            raw = passing()
            raw['policy_sha256'] = app.hashlib.sha256(policy.read_bytes()).hexdigest()
            raw['measurements']['yaw_mae'] = .2
            app.Path(cmd[cmd.index('--out')+1]).write_text(json.dumps(raw))
        def poll(self):
            return 0
    monkeypatch.setattr(app.subprocess, 'Popen', Child)
    monkeypatch.setattr(app, 'write_report', lambda *args: None)
    monkeypatch.setattr(app, 'wake_session', lambda *args: pytest.fail('Unexpected notification'))
    monkeypatch.delenv('HAPI_SESSION_ID', raising=False)
    app.main(tmp_path, ['--policy', str(policy), '--yaw-precheck'])
    job = next(output.glob('candidate_closed_review_*'))
    state = json.loads((job/'status.json').read_text())
    assert state['status'] == 'paused_on_regression'
    assert state['child_pid'] is None and len(calls) == 1
    assert len(state['reviews']) == 1
    assert state['reviews'][0]['result']['results'][0]['tracking'] is False
    assert (job/'completion_report.md').read_text() == render_candidate_review(policy, state)
    assert '不自动提升模型' in (job/'completion_report.md').read_text()
