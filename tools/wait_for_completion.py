"""Wait quietly for the supervisor's one-shot completion report, without reading metrics."""
import argparse,json,os,time
from pathlib import Path


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('job',type=Path);p.add_argument('--timeout',type=float,default=1800)
    a=p.parse_args();deadline=time.monotonic()+a.timeout
    while time.monotonic()<deadline:
        marker=a.job/'completion_hook.json'
        if marker.exists():
            data=json.loads(marker.read_text())
            if data.get('status')=='reported':
                print((a.job/'completion_report.md').read_text(),flush=True);return
        state=json.loads((a.job/'status.json').read_text())
        try:os.kill(state['supervisor_pid'],0)
        except ProcessLookupError:
            # Allow the final atomic report write, but do not wait forever after a crash.
            time.sleep(1)
            if marker.exists():continue
            raise RuntimeError('Supervisor exited without completion report; inspect job logs')
        time.sleep(10)
    raise TimeoutError('Completion wait expired; this does not imply the trainer stopped')


if __name__=='__main__':main()
