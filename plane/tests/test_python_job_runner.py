"""Verify process lifecycle and failure/cancellation semantics without training."""
import os
import sys

import pytest

from wheel_legged_gym.adapters.processes.python_job import PythonJobRunner


def runner(tmp_path, events, executable=sys.executable):
    return PythonJobRunner(
        executable=executable, working_directory=tmp_path, job_directory=tmp_path,
        environment=os.environ,
        on_preparing=lambda tag, cmd: events.append(('prepare', tag, cmd)),
        on_started=lambda pid: events.append(('started', pid)),
        on_finished=lambda: events.append(('finished',)),
    )


def test_success_logs_output_and_orders_callbacks(tmp_path):
    events = []
    runner(tmp_path, events).run(
        ['-c', 'import os; print(os.environ["ARCH_TEST_VALUE"])'],
        'success', {'ARCH_TEST_VALUE': 'explicit override'},
    )
    assert [event[0] for event in events] == ['prepare', 'started', 'finished']
    assert events[0][1] == 'success'
    assert events[1][1] > 0
    assert (tmp_path / 'success.log').read_text().strip() == 'explicit override'


def test_nonzero_exit_finishes_then_raises(tmp_path):
    events = []
    with pytest.raises(RuntimeError, match='failed exited 7'):
        runner(tmp_path, events).run(['-c', 'raise SystemExit(7)'], 'failed')
    assert [event[0] for event in events] == ['prepare', 'started', 'finished']


def test_existing_stop_does_not_launch_or_modify_operation(tmp_path):
    (tmp_path / 'STOP').touch()
    events = []
    with pytest.raises(InterruptedError, match='User STOP file'):
        runner(tmp_path, events).run(['-c', 'raise AssertionError'], 'stopped')
    assert events == []
    assert not (tmp_path / 'stopped.log').exists()


def test_spawn_failure_records_attempt_but_no_child_completion(tmp_path):
    events = []
    with pytest.raises(FileNotFoundError):
        runner(tmp_path, events, '/no-such-architecture-test-python').run([], 'missing')
    assert [event[0] for event in events] == ['prepare']


def test_stop_while_child_is_alive_terminates_and_reaps(tmp_path):
    events = []
    job = runner(tmp_path, events)
    def started(pid):
        events.append(('started', pid))
        (tmp_path / 'STOP').touch()
    job.on_started = started
    with pytest.raises(InterruptedError, match='User STOP file'):
        job.run(['-c', 'import time; time.sleep(30)'], 'cancelled')
    assert events[-1] == ('finished',)
    with pytest.raises(ProcessLookupError):
        os.kill(events[1][1], 0)
