"""Run a child that may own a nested job with cooperative STOP propagation."""
import subprocess
import time


def run_child(arguments, tag, child_job=None, *, executable, root, job, environment,
              on_preparing, on_started, on_finished):
    """Keep nested-job cancellation distinct from the motion runner protocol."""
    if (job / 'STOP').exists():
        raise InterruptedError('User STOP')
    on_preparing(tag)
    with (job / (tag + '.log')).open('w') as log:
        child = subprocess.Popen(
            [executable] + list(map(str, arguments)), cwd=root, env=environment,
            stdout=log, stderr=subprocess.STDOUT,
        )
        on_started(child.pid)
        try:
            while child.poll() is None:
                if (job / 'STOP').exists():
                    if child_job and child_job.exists():
                        (child_job / 'STOP').touch()
                    else:
                        child.terminate()
                time.sleep(2)
            if (job / 'STOP').exists():
                raise InterruptedError('User STOP')
            if child.returncode:
                raise RuntimeError(f'{tag} exited {child.returncode}')
        finally:
            if child.poll() is None:
                if child_job and child_job.exists():
                    (child_job / 'STOP').touch()
                else:
                    child.terminate()
                child.wait()
            on_finished()
