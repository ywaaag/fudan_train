"""Nested STOP propagation retains process ownership and callback ordering."""
from pathlib import Path
import pytest

from wheel_legged_gym.adapters.processes import cooperative_job


@pytest.mark.parametrize('mode', ['success', 'exit_error', 'early_stop', 'direct_stop',
                                  'nested_stop', 'poll_error', 'spawn_error'])
def test_child_lifecycle(tmp_path, monkeypatch, mode):
    job = tmp_path/'job'
    job.mkdir()
    nested = tmp_path/'nested'
    if mode in ['nested_stop', 'poll_error']:
        nested.mkdir()
    events = []
    command = []
    class Child:
        pid = 41
        returncode = None
        def __init__(self, args, **kwargs):
            command.extend(args)
            events.append('spawn')
            assert kwargs['cwd']==tmp_path and kwargs['env']=={'TASK': 'value'}
            if mode=='spawn_error':
                raise OSError('spawn failed')
        def poll(self):
            if mode=='success': self.returncode=0
            if mode=='exit_error': self.returncode=7
            if (nested/'STOP').exists(): self.returncode=0
            return self.returncode
        def terminate(self):
            events.append('terminate')
            self.returncode=-15
        def wait(self):
            events.append('wait')
            self.returncode=0
    def preparing(tag):
        events.append('preparing')
        assert tag=='case'
    def started(pid):
        events.append('started')
        assert pid==41
        if mode in ['direct_stop','nested_stop']:
            (job/'STOP').touch()
    def sleep(seconds):
        assert seconds==2
        if mode=='poll_error': raise RuntimeError('poll interrupted')
    if mode=='early_stop': (job/'STOP').touch()
    monkeypatch.setattr(cooperative_job.subprocess,'Popen',Child)
    monkeypatch.setattr(cooperative_job.time,'sleep',sleep)
    expected = {'early_stop':InterruptedError, 'direct_stop':InterruptedError,
                'nested_stop':InterruptedError, 'exit_error':RuntimeError,
                'poll_error':RuntimeError, 'spawn_error':OSError}.get(mode)
    def run():
        cooperative_job.run_child([Path('script.py'),'literal$(unchanged)'],'case',nested,
            executable='python',root=tmp_path,job=job,environment={'TASK':'value'},
            on_preparing=preparing,on_started=started,on_finished=lambda:events.append('finished'))
    if expected:
        with pytest.raises(expected): run()
    else:
        run()
    if mode=='early_stop':
        assert events==[] and not (job/'case.log').exists()
        return
    assert command==['python','script.py','literal$(unchanged)']
    if mode=='spawn_error':
        assert events==['preparing','spawn']
        return
    assert events[:3]==['preparing','spawn','started'] and events[-1]=='finished'
    if mode=='direct_stop': assert 'terminate' in events
    if mode in ['nested_stop','poll_error']:
        assert (nested/'STOP').exists() and 'terminate' not in events
    if mode=='poll_error': assert 'wait' in events
