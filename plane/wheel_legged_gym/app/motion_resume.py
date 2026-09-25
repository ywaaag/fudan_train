"""Load a stopped motion evaluation at the application I/O boundary."""
import json
import os
from pathlib import Path


def restore_evaluation_options(options, parser):
    """Return (job, state), restoring saved limits only after resume checks pass.

    This function reads the journal and probes recorded PIDs. It never starts a
    process, removes STOP, acquires the training lock, or writes the journal.
    Permission errors from the PID probe propagate rather than imply a dead PID.
    """
    job = options.resume_evaluation_job.resolve()
    state = json.loads((job / 'status.json').read_text())
    stopped_evaluation = (
        state['status'] == 'stopped' and state.get('error') == 'User STOP file'
        and state.get('operation', '').startswith('r')
        and '_m' in state.get('operation', '') and state.get('run_dir')
    )
    if state['status'] != 'evaluating' and not stopped_evaluation:
        parser.error('Resume only supports completed-training evaluation boundary')
    if stopped_evaluation:
        source_iteration = int(Path(state['source']).stem.split('_')[-1])
        if not (Path(state['run_dir']) / f'model_{source_iteration + 500}.pt').exists():
            parser.error('Completed training checkpoint missing')
    for key in ['supervisor_pid', 'child_pid']:
        pid = state.get(key)
        if pid:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                pass
            else:
                parser.error('Recorded process is still alive; refusing duplicate supervisor')
    for key, value in state['limits'].items():
        if key != 'resume_evaluation_job':
            setattr(options, key, value)
    return job, state
