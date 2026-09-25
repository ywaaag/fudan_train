"""State saves must preserve ordering and avoid duplicate/implicit notifications."""
import json

import pytest

from wheel_legged_gym.adapters.artifacts.job_files import JobFiles
from wheel_legged_gym.app.motion_status import save_motion_status
from wheel_legged_gym.workflows.completion import TERMINAL


def state(status):
    return dict(status=status, stage='basic_motion', round=2, accepted=None,
                history=[{'decision': '保留原模型'}])


@pytest.mark.parametrize('terminal', [False, True])
@pytest.mark.parametrize('session', ['', 'test-session'])
def test_save_order_and_wake_condition(tmp_path, terminal, session):
    current = state(sorted(TERMINAL)[0] if terminal else 'training')
    calls = []
    def report(directory, saved):
        assert json.loads((directory / 'status.json').read_text()) == current
        assert not (directory / 'status.tmp').exists()
        assert not (directory / 'progress.md').exists()
        calls.append('report')
    def wake(directory, identifier):
        assert identifier == session
        assert calls == ['report']
        calls.append('wake')
    save_motion_status(JobFiles(tmp_path), current,
                       environment={'HAPI_SESSION_ID': session}, report=report, wake=wake)
    assert calls == (['report', 'wake'] if terminal and session else ['report'])
    assert '保留原模型' in (tmp_path / 'progress.md').read_text()
    assert 'Goal remains incomplete' in (tmp_path / 'progress.md').read_text()


def test_wake_failure_is_recorded_and_progress_still_written(tmp_path):
    def wake(*args):
        raise RuntimeError('transport failed')
    save_motion_status(JobFiles(tmp_path), state(sorted(TERMINAL)[0]),
                       environment={'HAPI_SESSION_ID': 'test'}, report=lambda *args: None,
                       wake=wake)
    assert (tmp_path / 'hapi_hook_error.txt').read_text() == 'transport failed\n'
    assert (tmp_path / 'progress.md').exists()


def test_report_failure_propagates_without_wake_or_progress(tmp_path):
    def report(*args):
        raise RuntimeError('report failed')
    def wake(*args):
        pytest.fail('must not wake after report failure')
    with pytest.raises(RuntimeError, match='report failed'):
        save_motion_status(JobFiles(tmp_path), state(sorted(TERMINAL)[0]),
                           environment={'HAPI_SESSION_ID': 'test'}, report=report, wake=wake)
    assert (tmp_path / 'status.json').exists()
    assert not (tmp_path / 'progress.md').exists()
