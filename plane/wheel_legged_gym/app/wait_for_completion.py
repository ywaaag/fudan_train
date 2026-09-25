"""Wait for a supervisor completion report at the application boundary."""
import argparse
import json
import os
import time
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('job', type=Path)
    parser.add_argument('--timeout', type=float, default=1800)
    args = parser.parse_args(argv)
    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        marker = args.job / 'completion_hook.json'
        if marker.exists():
            data = json.loads(marker.read_text())
            if data.get('status') == 'reported':
                print((args.job / 'completion_report.md').read_text(), flush=True)
                return
        state = json.loads((args.job / 'status.json').read_text())
        try:
            os.kill(state['supervisor_pid'], 0)
        except ProcessLookupError:
            time.sleep(1)
            if marker.exists():
                continue
            raise RuntimeError('Supervisor exited without completion report; inspect job logs')
        time.sleep(10)
    raise TimeoutError('Completion wait expired; this does not imply the trainer stopped')
