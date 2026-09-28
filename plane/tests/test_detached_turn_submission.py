import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest


@pytest.mark.parametrize(('trainer_returncode','failure_stage'),
    [(0,None),(7,None),(0,'trainer'),(0,'monitor'),(0,'hook')])
def test_dummy_orchestrator_survives_submit_parent_and_finalizes(tmp_path,
                                                                 trainer_returncode,
                                                                 failure_stage):
    root=tmp_path/'fake_root'
    (root/'plane/logs/wheel_legged').mkdir(parents=True)
    (root/'tools').mkdir()
    job=tmp_path/'job';job.mkdir()
    ledger=tmp_path/'budget_ledger.json'
    ledger.write_text(json.dumps({'formal_limit_iterations':1000,
        'formal_used_iterations':0,'source_checkpoint_sha256':'source',
        'attempts':[{'attempt_id':'attempt','run_name':'dummy','source_checkpoint_sha256':'source',
            'reserved_iterations':1000,'status':'reserved','actual_update_count':None}]}))
    spec=tmp_path/'spec.json';spec.write_text('{}')
    trainer=tmp_path/'dummy_trainer.py'
    trainer.write_text(('raise SystemExit(3)\n' if failure_stage=='trainer' else """import json,sys,time
from pathlib import Path
run=Path(sys.argv[1]);run.mkdir()
(run/'policy_experiment.json').write_text('{}')
time.sleep(1.5)
(run/'model_1100.pt').write_text('dummy checkpoint')
"""+f'\nraise SystemExit({trainer_returncode})\n'))
    monitor=root/'tools/monitor_turn_lean_long.py'
    monitor.write_text('raise SystemExit(4)\n' if failure_stage=='monitor' else """import json,sys,time
from pathlib import Path
job=Path(sys.argv[sys.argv.index('--job')+1]);run=Path(sys.argv[sys.argv.index('--run')+1])
(job/'status.json').write_text(json.dumps({'status':'running','source_iteration':100,
 'run':str(run),'new_iteration_limit':1000,'milestones':[]}))
while not (job/'trainer_exit.json').exists(): time.sleep(.05)
(job/'status.json').write_text(json.dumps({'status':'finished_pending_review',
 'source_iteration':100,'run':str(run),'new_iteration_limit':1000,
 'actual_update_count':1000,'checkpoint_iteration':1100,
 'last_checkpoint_sha256':'checkpoint','milestones':[]}))
""")
    hook=root/'tools/training_completion_hook.py'
    hook.write_text('raise SystemExit(5)\n' if failure_stage=='hook' else """import sys,time
from pathlib import Path
job=Path(sys.argv[sys.argv.index('--job')+1])
while not (job/'report_ready.json').exists(): time.sleep(.05)
""")
    run=root/'plane/logs/wheel_legged/Sep28_dummy'
    snapshot={'attempt_id':'attempt','event_id':'event-dummy','budget_ledger':str(ledger),
        'source_iteration':100,'target_new_iterations':1000,'spec':str(spec),
        'run_name':'dummy','actual_cli':[sys.executable,str(trainer),str(run)],
        'hapi_session':None}
    (job/'submit_snapshot.json').write_text(json.dumps(snapshot))
    package=Path(__file__).resolve().parents[1]
    env=dict(os.environ,PYTHONPATH=str(package))
    launcher="""import subprocess,sys
from pathlib import Path
command=[sys.executable,'-c',
 'import sys; from wheel_legged_gym.app.submit_turn_training import orchestrate; from pathlib import Path; orchestrate(Path(sys.argv[1]),Path(sys.argv[2]))',
 sys.argv[1],sys.argv[2]]
log=open(sys.argv[2]+'/dummy_orchestrator.log','w')
p=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,
 stderr=subprocess.STDOUT,start_new_session=True)
print(p.pid)
"""
    parent=subprocess.run([sys.executable,'-c',launcher,str(root),str(job)],
        env=env,text=True,capture_output=True,check=True)
    assert parent.returncode==0
    deadline=time.monotonic()+15
    while time.monotonic()<deadline:
        marker=job/'monitor/report_ready.json'
        if marker.exists():
            break
        submission=job/'submission.json'
        if failure_stage and submission.exists() and json.loads(
                submission.read_text()).get('status')=='error':
            break
        time.sleep(.1)
    if failure_stage:
        status=json.loads((job/'submission.json').read_text())
        assert status['status']=='error'
        assert not marker.exists()
        budget=json.loads(ledger.read_text())
        assert budget['formal_used_iterations']==0
        assert budget['attempts'][0]['status'] in ('finalized','unknown_consumption')
        return
    assert marker.exists(), (job/'submission.json').read_text() if (
        job/'submission.json').exists() else (job/'dummy_orchestrator.log').read_text()
    submission=json.loads((job/'submission.json').read_text())
    assert submission['status']=='running'
    assert submission['trainer_pid'] and submission['monitor_pid'] and submission['hook_pid']
    assert json.loads((job/'monitor/trainer_exit.json').read_text())[
        'returncode']==trainer_returncode
    summary=json.loads((job/'monitor/terminal_summary.json').read_text())
    assert summary['event_id']=='event-dummy'
    assert summary['last_checkpoint'].endswith('model_1100.pt')
    assert summary['actual_update_count']==1000
    budget=json.loads(ledger.read_text())
    if trainer_returncode==0:
        assert summary['status']=='finished_pending_review'
        assert summary['stop_reason']=='budget_exhausted'
        assert budget['formal_used_iterations']==1000
    else:
        assert summary['status']=='error'
        assert summary['stop_reason']=='runtime_error'
        assert budget['attempts'][0]['status']=='unknown_consumption'
