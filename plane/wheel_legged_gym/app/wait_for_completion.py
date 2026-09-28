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
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--json', action='store_true', dest='as_json')
    parser.add_argument('--no-report', action='store_true')
    args = parser.parse_args(argv)
    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        marker = args.job / 'completion_hook.json'
        if marker.exists():
            data = json.loads(marker.read_text())
            if data.get('status') == 'reported':
                if not args.quiet:
                    if args.as_json:
                        print(json.dumps({'job': str(args.job), 'status': data.get('status'),
                            'report': str(args.job / 'completion_report.md'),
                            'event_id': data.get('event_id', 'unknown')}), flush=True)
                    elif not args.no_report:
                        print((args.job / 'completion_report.md').read_text(), flush=True)
                    return
        state = json.loads((args.job / 'status.json').read_text())
        if args.once:
            if not args.quiet:
                payload = {'job': str(args.job), 'status': state.get('status', 'unknown')}
                print(json.dumps(payload) if args.as_json else payload['status'], flush=True)
            return
        try:
            os.kill(state['supervisor_pid'], 0)
        except ProcessLookupError:
            time.sleep(1)
            if marker.exists():
                continue
            raise RuntimeError('Supervisor exited without completion report; inspect job logs')
        time.sleep(10)
    raise TimeoutError('Completion wait expired; this does not imply the trainer stopped')
