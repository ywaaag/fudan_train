"""Run one logged Python child with cooperative STOP-file cancellation."""
from pathlib import Path
import subprocess
import time


class PythonJobRunner:
    """Own child lifetime; report transitions through explicit callbacks.

    No training decisions, status JSON writes or notifications occur here.
    Callbacks are synchronous and preserve the supervisor's original save order.
    """

    def __init__(self, *, executable, working_directory, job_directory, environment,
                 on_preparing, on_started, on_finished):
        self.executable = executable
        self.working_directory = working_directory
        self.job_directory = Path(job_directory)
        self.environment = dict(environment)
        self.on_preparing = on_preparing
        self.on_started = on_started
        self.on_finished = on_finished

    def run(self, arguments, tag, extra=None):
        if (self.job_directory / 'STOP').exists():
            raise InterruptedError('User STOP file')
        command = [self.executable] + [str(argument) for argument in arguments]
        self.on_preparing(tag, command)
        with (self.job_directory / (tag + '.log')).open('w') as log:
            child = subprocess.Popen(
                command, cwd=self.working_directory,
                env=dict(self.environment, **(extra or {})),
                stdout=log, stderr=subprocess.STDOUT,
            )
            self.on_started(child.pid)
            try:
                while child.poll() is None:
                    if (self.job_directory / 'STOP').exists():
                        raise InterruptedError('User STOP file')
                    time.sleep(2)
                if child.returncode:
                    raise RuntimeError(f'{tag} exited {child.returncode}')
            finally:
                if child.poll() is None:
                    child.terminate()
                    child.wait()
                self.on_finished()
