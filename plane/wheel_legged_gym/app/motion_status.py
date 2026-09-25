"""Compose status storage, completion report and optional session wakeup."""
from wheel_legged_gym.workflows.completion import TERMINAL
from wheel_legged_gym.workflows.motion_report import render_progress


def save_motion_status(files, state, *, environment, report, wake):
    """Keep legacy side-effect order; only wake errors are caught here.

    environment remains caller-owned and is read on each save, as in the
    original supervisor. Inject report/wake to test without sending messages.
    """
    files.save_status(state)
    report(files.directory, state)
    if state['status'] in TERMINAL and environment.get('HAPI_SESSION_ID'):
        try:
            wake(files.directory, environment['HAPI_SESSION_ID'])
        except Exception as exc:
            files.write_text('hapi_hook_error.txt', str(exc) + '\n')
    files.write_text('progress.md', render_progress(state))
