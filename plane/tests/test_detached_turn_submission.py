import copy
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest


def test_ablation_plan_rejects_non_coefficient_training_differences(tmp_path):
    from wheel_legged_gym.app.submit_turn_training import validate_ablation_plan
    source = tmp_path/'model_38250.pt'
    source.write_bytes(b'dummy source')
    spec = json.loads((Path(__file__).resolve().parents[1].parent/
        'plane/outputs/high_load_cornering_stageB_20260928/spec_stage_b.json').read_text())
    spec['experiment_id'] = 'roll_constraint_ablation_v1'
    spec['source_checkpoint'] = str(source)
    spec['source_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
    spec['source_iteration'] = 38250
    spec['course_plan'] = [{'phase':1,'budget_max':1}, {'phase':2,'budget_max':10000}]
    spec['stage']['turn_roll_cost_coefficient'] = 1.
    specs = {}
    for name,coefficient in (('A',1.),('B',.25)):
        item = copy.deepcopy(spec)
        item['stage']['turn_roll_cost_coefficient'] = coefficient
        path = tmp_path/(name+'.json')
        path.write_text(json.dumps(item))
        specs[name] = {'spec':str(path),'run_name':'ablation_'+name,
                       'new_iterations':5000}
    holdout = tmp_path/'holdout.json';holdout.write_text('{}')
    baseline = tmp_path/'baseline.json';baseline.write_text('{}')
    plan = {'output_root':str(tmp_path),'branches':specs,
        'holdout_commands':str(holdout),'baseline_sidecar':str(baseline)}
    assert validate_ablation_plan(plan)['source_sha256']==spec['source_sha256']
    altered = json.loads((tmp_path/'B.json').read_text())
    altered['stage']['learning_rate'] = 3e-6
    (tmp_path/'B.json').write_text(json.dumps(altered))
    with pytest.raises(ValueError,match='differ beyond'):
        validate_ablation_plan(plan)


def test_sampling_plan_keeps_same_source_and_only_sampling_changes(tmp_path):
    from wheel_legged_gym.app.submit_turn_training import validate_ablation_plan
    source = tmp_path/'model_43250.pt';source.write_bytes(b'source')
    old = (Path(__file__).resolve().parents[1].parent/
        'plane/outputs/roll_constraint_ablation_20260928_110822/spec_A.json')
    spec = json.loads(old.read_text())
    spec['experiment_id'] = 'cornering_sampling_ablation_v1'
    spec['source_checkpoint'] = str(source)
    spec['source_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
    spec['source_iteration'] = 43250
    sampling = {'version':1,'failed_centers':[
        [-4.,-1.],[-4.,1.],[-3.5,-.8],[-3.,-1.],[-3.,1.],
        [3.,-1.],[3.5,-.8],[4.,-1.]],
        'other_centers':[[3.,1.],[-3.5,.8],[3.5,.8],[4.,1.]],
        'vx_half_width':.15,'yaw_half_width':.05}
    branches = {}
    for name in ('A','B'):
        item = copy.deepcopy(spec)
        if name == 'B':
            item['stage']['sampling_ablation'] = sampling
        path = tmp_path/(name+'.json');path.write_text(json.dumps(item))
        branches[name] = {'spec':str(path),'run_name':'sampling_'+name,
                          'new_iterations':5000}
    holdout = tmp_path/'holdout.json';holdout.write_text('{}')
    baseline = tmp_path/'baseline.json';baseline.write_text('{}')
    plan = {'output_root':str(tmp_path),'branches':branches,
        'holdout_commands':str(holdout),'baseline_sidecar':str(baseline)}
    assert validate_ablation_plan(plan)['source_iteration'] == 43250
    altered = json.loads((tmp_path/'B.json').read_text())
    altered['stage']['learning_rate'] = 3e-6
    (tmp_path/'B.json').write_text(json.dumps(altered))
    with pytest.raises(ValueError,match='differ beyond'):
        validate_ablation_plan(plan)


def test_review_checkpoint_prefers_screened_candidate_over_last(tmp_path):
    from wheel_legged_gym.app.submit_turn_training import review_checkpoint
    early = tmp_path/'model_44250.pt';early.write_bytes(b'best')
    last = tmp_path/'model_48250.pt';last.write_bytes(b'last')
    summary = {'last_checkpoint':str(last),
        'best_candidate':{'checkpoint':str(early),
            'checkpoint_sha256':hashlib.sha256(early.read_bytes()).hexdigest()}}
    assert review_checkpoint(summary) == str(early)
    summary['best_candidate']['checkpoint_sha256'] = 'bad'
    with pytest.raises(ValueError,match='SHA'):
        review_checkpoint(summary)


def test_ablation_plan_submits_sequential_branches_and_one_terminal(tmp_path, monkeypatch):
    from wheel_legged_gym.app.submit_turn_training import orchestrate_ablation_plan
    output = tmp_path/'output';output.mkdir()
    specs = {}
    for name in ('A','B'):
        spec = output/(name+'.json');spec.write_text('{}')
        specs[name] = {'spec':str(spec),'run_name':'run_'+name,'new_iterations':5000}
    holdout = output/'holdout.json';holdout.write_text('{}')
    baseline = output/'baseline.json'
    result = {'protocol_id':'cornering_task_v2','evaluator_sha256':'evaluator',
        'gate_sha256':'gate','metric_definition_sha256':'metric',
        'retention_points':25,'points':[{'group':'turn_train','command':[2.,.5,.4],
                                        'new_passed':True}]}
    baseline.write_text(json.dumps(result))
    plan = {'output_root':str(output),'branches':specs,
        'holdout_commands':str(holdout),'baseline_sidecar':str(baseline)}
    plan_path = output/'plan.json';plan_path.write_text(json.dumps(plan))
    (output/'plan_submission.json').write_text(json.dumps({'event_id':'event-plan'}))
    monkeypatch.setattr('wheel_legged_gym.app.submit_turn_training.validate_ablation_plan',
        lambda _plan:{'source_checkpoint':'/source/model_38250.pt'})
    calls=[]
    def fake_run(argv, **_kwargs):
        if 'submit_turn_training.py' in argv[1]:
            name = argv[argv.index('--run-name')+1]
            calls.append(name)
            job = output/name
            monitor = job/'monitor';monitor.mkdir(parents=True)
            checkpoint = job/'model_43250.pt';checkpoint.write_bytes(b'model')
            (monitor/'report_ready.json').write_text('{}')
            (monitor/'status.json').write_text(json.dumps({'stop_reason':'budget_exhausted'}))
            (monitor/'terminal_summary.json').write_text(json.dumps({
                'last_checkpoint':str(checkpoint)}))
            return subprocess.CompletedProcess(argv,0,stdout=json.dumps({
                'orchestrator_pid':os.getpid(),
                'orchestrator_start_ticks':0,'trainer_pid':123,
                'monitor_pid':124,'hook_pid':125,'event_id':'branch'}))
        calls.append(Path(argv[1]).name)
        if '--cornering-task-v2-out' in argv:
            path=Path(argv[argv.index('--cornering-task-v2-out')+1])
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps(result))
        return subprocess.CompletedProcess(argv,0)
    monkeypatch.setattr('wheel_legged_gym.app.submit_turn_training.subprocess.run',fake_run)
    orchestrate_ablation_plan(tmp_path,plan_path,None)
    assert calls[:3] == ['run_A','review_turn_lean_long.py','run_B']
    assert (output/'ablation_comparison.json').is_file()
    terminal=json.loads((output/'plan_terminal/status.json').read_text())
    assert terminal['status']=='finished_pending_review'
    assert terminal['event_id']=='event-plan'


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
        assert budget['attempts'][0]['status']=='finalized'
        assert budget['formal_used_iterations']==1000
