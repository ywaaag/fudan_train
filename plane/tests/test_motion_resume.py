"""Resume checks must finish before options change or any job can be launched."""
import argparse
import json

import pytest

from wheel_legged_gym.app import motion_resume


def setup_job(tmp_path, status='evaluating'):
    run = tmp_path/'run'
    run.mkdir()
    state = {'status':status, 'error':'User STOP file', 'operation':'r01_m10200',
             'source':str(run/'model_10000.pt'), 'run_dir':str(run),
             'supervisor_pid':101, 'child_pid':102,
             'limits':{'max_rounds':7,'resume_evaluation_job':'old/job'}}
    (tmp_path/'status.json').write_text(json.dumps(state))
    (tmp_path/'STOP').touch()
    options = argparse.Namespace(resume_evaluation_job=tmp_path, max_rounds=1)
    return state, options, run


@pytest.mark.parametrize('status', ['evaluating', 'stopped'])
def test_restore_after_dead_processes_only(tmp_path, monkeypatch, status):
    state, options, run = setup_job(tmp_path, status)
    if status=='stopped': (run/'model_10500.pt').touch()
    probes = []
    def dead(pid, signal):
        probes.append((pid, signal))
        raise ProcessLookupError
    monkeypatch.setattr(motion_resume.os, 'kill', dead)
    before = (tmp_path/'status.json').read_bytes()
    job, restored = motion_resume.restore_evaluation_options(options, argparse.ArgumentParser())
    assert probes==[(101,0),(102,0)]
    assert restored==state and job==tmp_path.resolve()
    assert options.max_rounds==7 and options.resume_evaluation_job==tmp_path
    assert (tmp_path/'STOP').exists() and (tmp_path/'status.json').read_bytes()==before


@pytest.mark.parametrize('change', [
    {'status':'training'}, {'status':'stopped','error':'different'},
    {'status':'stopped','operation':'r01_train'}, {'status':'stopped','run_dir':None},
    {'status':'stopped'},
])
def test_invalid_boundary_or_missing_checkpoint_prevents_pid_probes(tmp_path, monkeypatch, change):
    state, options, _ = setup_job(tmp_path)
    state.update(change)
    (tmp_path/'status.json').write_text(json.dumps(state))
    monkeypatch.setattr(motion_resume.os, 'kill', lambda *args: pytest.fail('Early PID probe'))
    with pytest.raises(SystemExit) as error:
        motion_resume.restore_evaluation_options(options, argparse.ArgumentParser())
    assert error.value.code==2 and options.max_rounds==1


@pytest.mark.parametrize('pid', [101,102])
def test_live_process_blocks_restore(tmp_path, monkeypatch, pid):
    _, options, _ = setup_job(tmp_path)
    def probe(value, signal):
        if value!=pid: raise ProcessLookupError
    monkeypatch.setattr(motion_resume.os,'kill',probe)
    with pytest.raises(SystemExit):
        motion_resume.restore_evaluation_options(options,argparse.ArgumentParser())
    assert options.max_rounds==1


def test_permission_error_is_not_a_dead_process(tmp_path, monkeypatch):
    _, options, _ = setup_job(tmp_path)
    def probe(*args): raise PermissionError('Cannot inspect PID')
    monkeypatch.setattr(motion_resume.os,'kill',probe)
    with pytest.raises(PermissionError):
        motion_resume.restore_evaluation_options(options,argparse.ArgumentParser())
    assert options.max_rounds==1
