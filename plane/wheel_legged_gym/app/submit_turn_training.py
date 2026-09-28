"""Submit one bounded turn training job to a detached process owner."""
import argparse
import copy
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

from wheel_legged_gym.app.training_budget import atomic_json, reserve, settle
from wheel_legged_gym.workflows.completion import write_report


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def process_start(pid):
    """Linux /proc start ticks distinguish a reused PID from the launched child."""
    return int(Path('/proc/{}/stat'.format(pid)).read_text().rsplit(')', 1)[1].split()[19])


def terminate(child):
    if child is None or child.poll() is not None:
        return
    os.killpg(child.pid, signal.SIGTERM)
    try:
        child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait(timeout=10)


def find_run(root, run_name, trainer, deadline):
    logs = root / 'plane/logs/wheel_legged'
    while time.monotonic() < deadline:
        matches = sorted(logs.glob('*_' + run_name))
        if matches and (matches[-1] / 'policy_experiment.json').is_file():
            return matches[-1]
        if trainer.poll() is not None:
            raise RuntimeError('Trainer exited before manifest; returncode={}'.format(trainer.returncode))
        time.sleep(.2)
    raise TimeoutError('Trainer did not create a run manifest before startup deadline')


def orchestrate(root, job):
    snapshot = json.loads((job/'submit_snapshot.json').read_text())
    attempt, event = snapshot['attempt_id'], snapshot['event_id']
    ledger = Path(snapshot['budget_ledger'])
    source = snapshot['source_iteration']
    target = snapshot['target_new_iterations']
    trainer = monitor = hook = None
    run = None
    def interrupted(_signal, _frame):
        raise KeyboardInterrupt('Orchestrator received termination request')
    signal.signal(signal.SIGTERM, interrupted)
    try:
        env = dict(os.environ)
        env['PYTHONPATH'] = str(root/'plane') + os.pathsep + env.get('PYTHONPATH','')
        env['FUDAN_TURN_LONG_SPEC'] = snapshot['spec']
        with (job/'trainer.log').open('w') as log:
            trainer = subprocess.Popen(snapshot['actual_cli'], cwd=root/'plane', env=env,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True)
        started = process_start(trainer.pid)
        atomic_json(job/'submission.json', {'job':str(job/'monitor'), 'status':'starting',
            'trainer_pid':trainer.pid, 'trainer_start_ticks':started,
            'orchestrator_pid':os.getpid(), 'orchestrator_start_ticks':process_start(os.getpid()),
            'monitor_pid':None, 'hook_pid':None, 'event_id':event,
            'attempt_id':attempt})
        run = find_run(root, snapshot['run_name'], trainer, time.monotonic()+120)
        monitor_job = job/'monitor'
        monitor_job.mkdir(exist_ok=False)
        command = [sys.executable, str(root/'tools/monitor_turn_lean_long.py'),
            '--job',str(monitor_job), '--run',str(run), '--spec',snapshot['spec'],
            '--trainer-pid',str(trainer.pid), '--trainer-start-ticks',str(started),
            '--new-iterations',str(target), '--external-finalizer']
        with (job/'monitor.log').open('w') as log:
            monitor = subprocess.Popen(command, cwd=root, env=env,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True)
        deadline = time.monotonic()+30
        while not (monitor_job/'status.json').is_file():
            if monitor.poll() is not None:
                raise RuntimeError('Monitor exited before startup acknowledgement')
            if trainer.poll() is not None:
                raise RuntimeError('Trainer exited during monitor startup')
            if time.monotonic() >= deadline:
                raise TimeoutError('Monitor did not acknowledge startup')
            time.sleep(.1)
        hook_command = [sys.executable,str(root/'tools/training_completion_hook.py'),
            '--job',str(monitor_job)]
        if snapshot.get('hapi_session'):
            hook_command += ['--hapi-session',snapshot['hapi_session']]
        with (job/'hook.log').open('w') as log:
            hook = subprocess.Popen(hook_command,cwd=root,env=env,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                start_new_session=True)
        time.sleep(.2)
        if hook.poll() is not None:
            raise RuntimeError('Completion hook exited before startup acknowledgement')
        atomic_json(job/'submission.json', {'job':str(monitor_job), 'status':'running',
            'trainer_pid':trainer.pid, 'trainer_start_ticks':started,
            'monitor_pid':monitor.pid, 'monitor_start_ticks':process_start(monitor.pid),
            'hook_pid':hook.pid, 'hook_start_ticks':process_start(hook.pid),
            'orchestrator_pid':os.getpid(), 'orchestrator_start_ticks':process_start(os.getpid()),
            'event_id':event, 'attempt_id':attempt, 'run':str(run)})
        while trainer.poll() is None:
            if monitor.poll() is not None:
                raise RuntimeError('Monitor exited while trainer was running')
            if hook.poll() is not None:
                raise RuntimeError('Completion hook exited while trainer was running')
            time.sleep(1)
        returncode = trainer.wait()
        atomic_json(monitor_job/'trainer_exit.json', {'attempt_id':attempt,
            'pid':trainer.pid, 'start_ticks':started, 'returncode':returncode,
            'exited_at':time.time(), 'stop_request':('STOP' if (monitor_job/'STOP').exists()
                else None)})
        monitor_returncode = monitor.wait()
        state = json.loads((monitor_job/'status.json').read_text())
        actual = state.get('actual_update_count')
        if not isinstance(actual,int):
            actual = None
        if (returncode == 0 and monitor_returncode == 0 and actual == target and
                state.get('stop_reason') in (None,'budget_exhausted')):
            state.update(status='finished_pending_review',stop_reason='budget_exhausted')
        elif state.get('stop_reason') in ('user_stop','retention_regression','no_progress'):
            pass
        else:
            state.update(status='error',stop_reason='runtime_error')
        state.update(trainer_returncode=returncode,monitor_returncode=monitor_returncode,
            event_id=event,attempt_id=attempt,child_pid=None)
        atomic_json(monitor_job/'status.json',state)
        settle(ledger,attempt,actual,str(monitor_job/'status.json'),
               unknown=actual is None)
        write_report(monitor_job,state,lambda _message:None)
        atomic_json(monitor_job/'report_ready.json',{'event_id':event,
            'report':str(monitor_job/'completion_report.md'),'ready_at':time.time()})
        hook.wait(timeout=120)
    except BaseException as exc:
        for child in (hook,monitor,trainer):
            terminate(child)
        failure = {'status':'error','stop_reason':'runtime_error','error':str(exc),
            'event_id':event,'attempt_id':attempt,'run':str(run) if run else None,
            'source_iteration':source,'new_iteration_limit':target,
            'trainer_returncode':trainer.returncode if trainer else None}
        atomic_json(job/'submission.json',failure)
        monitor_job=job/'monitor'
        if monitor_job.is_dir():
            atomic_json(monitor_job/'finalization_error.json',failure)
        try:
            known_zero = run is None and trainer is not None and trainer.returncode is not None
            settle(ledger,attempt,0 if known_zero else None,
                   str(job/'submission.json'),unknown=not known_zero)
        except Exception as ledger_exc:
            failure['budget_error']=str(ledger_exc)
            atomic_json(job/'submission.json',failure)
        raise


def validate_ablation_plan(plan):
    if set(plan) != {'output_root','branches','holdout_commands','baseline_sidecar'}:
        raise ValueError('Ablation plan has unreviewed fields')
    branches = plan['branches']
    if set(branches) != {'A','B'}:
        raise ValueError('Ablation requires exactly A and B')
    normalized = []
    for name in ('A','B'):
        branch = branches[name]
        if set(branch) != {'spec','run_name','new_iterations'} or branch['new_iterations'] != 5000:
            raise ValueError('Each ablation branch must request exactly 5000 updates')
        data = json.loads(Path(branch['spec']).read_text())
        from wheel_legged_gym.experiments.recipes.turn_lean_long import validate_spec
        validate_spec(data)
        if data.get('experiment_id') != 'roll_constraint_ablation_v1':
            raise ValueError('Ablation experiment identity differs')
        if data['stage']['turn_roll_cost_coefficient'] != (1. if name == 'A' else .25):
            raise ValueError('Branch roll coefficient differs')
        source = Path(data['source_checkpoint']).resolve(strict=True)
        if digest(source) != data['source_sha256']:
            raise ValueError('Ablation source SHA differs')
        comparable = copy.deepcopy(data)
        del comparable['stage']['turn_roll_cost_coefficient']
        normalized.append(comparable)
    if normalized[0] != normalized[1]:
        raise ValueError('A/B training specs differ beyond turn roll coefficient')
    if branches['A']['run_name'] == branches['B']['run_name']:
        raise ValueError('A/B run names must be distinct')
    if not Path(plan['holdout_commands']).is_file() or not Path(plan['baseline_sidecar']).is_file():
        raise ValueError('Frozen holdout or baseline result is missing')
    return normalized[0]


def orchestrate_ablation_plan(root, plan_path, session):
    plan = json.loads(plan_path.read_text())
    comparable = validate_ablation_plan(plan)
    output = Path(plan['output_root']).resolve(strict=True)
    event = json.loads((output/'plan_submission.json').read_text())['event_id']
    jobs = {}
    try:
        for name in ('A','B'):
            branch = plan['branches'][name]
            argv = [sys.executable,str(root/'tools/submit_turn_training.py'),
                '--spec',branch['spec'],'--run-name',branch['run_name'],
                '--new-iterations','5000','--output-root',str(output),
                '--num-envs','4096','--seed','23','--detach']
            submitted = subprocess.run(argv,cwd=root,text=True,capture_output=True,check=True)
            identity = json.loads(submitted.stdout)
            jobs[name] = identity
            atomic_json(output/'plan_submission.json', {'status':'running',
                'active_branch':name,'event_id':event,'jobs':jobs,
                'plan_owner_pid':os.getpid(),
                'plan_owner_start_ticks':process_start(os.getpid())})
            job = output/branch['run_name']
            marker = job/'monitor/report_ready.json'
            while not marker.is_file():
                status_path = job/'submission.json'
                if status_path.is_file() and json.loads(status_path.read_text()).get('status') == 'error':
                    raise RuntimeError('{} infrastructure failure: {}'.format(name,status_path))
                owner = identity['orchestrator_pid']
                try:
                    if process_start(owner) != identity['orchestrator_start_ticks']:
                        raise RuntimeError('{} orchestrator identity changed'.format(name))
                except FileNotFoundError:
                    raise RuntimeError('{} orchestrator exited without report'.format(name))
                time.sleep(30)
            status = json.loads((job/'monitor/status.json').read_text())
            if status.get('stop_reason') in ('runtime_error','exit_status_unknown'):
                raise RuntimeError('{} ended with infrastructure error'.format(name))
            endpoint = json.loads((job/'monitor/terminal_summary.json').read_text()).get('last_checkpoint')
            if not endpoint or not Path(endpoint).is_file():
                raise RuntimeError('{} has no endpoint for stage screening'.format(name))
            (job/'holdout_commands.json').write_bytes(Path(plan['holdout_commands']).read_bytes())
            stage_review = [sys.executable,str(root/'tools/review_turn_lean_long.py'),
                '--job',str(job),'--checkpoint',endpoint,'--spec',branch['spec'],
                '--review-name','stage_screen','--model-label',name,
                '--seeds','19','--envs-per-command','4',
                '--groups','retention','turn_train','entry_exit','slow_reverse',
                '--resume']
            with (job/'stage_screen.log').open('w') as log:
                subprocess.run(stage_review,cwd=root,check=True,
                    stdout=log,stderr=subprocess.STDOUT)

        review_results = {}
        holdout = Path(plan['holdout_commands']).read_bytes()
        for name in ('A','B'):
            branch = plan['branches'][name]
            job = output/branch['run_name']
            summary = json.loads((job/'monitor/terminal_summary.json').read_text())
            checkpoint = summary.get('last_checkpoint')
            if not checkpoint or not Path(checkpoint).is_file():
                raise RuntimeError('{} has no saved endpoint checkpoint for comparison'.format(name))
            (job/'holdout_commands.json').write_bytes(holdout)
            review = job/'phase_reviews'/'endpoint'
            commands = [
                [sys.executable,str(root/'tools/review_turn_lean_long.py'),
                    '--job',str(job),'--checkpoint',checkpoint,'--spec',branch['spec'],
                    '--review-name','endpoint','--model-label',name,
                    '--seeds','19','37','53','--envs-per-command','16','--resume'],
                [sys.executable,str(root/'tools/summarize_turn_envelope.py'),
                    '--job',str(review),'--long'],
                [sys.executable,str(root/'tools/summarize_turn_envelope.py'),
                    '--job',str(review),'--cornering-task-v2-out',
                    str(review/'cornering_task_v2.json')],
            ]
            for index,command in enumerate(commands):
                with (job/'review_{}.log'.format(index)).open('w') as log:
                    subprocess.run(command,cwd=root,check=True,stdout=log,stderr=subprocess.STDOUT)
            review_results[name] = str(review/'cornering_task_v2.json')
        results = {name:json.loads(Path(path).read_text())
                   for name,path in review_results.items()}
        baseline = json.loads(Path(plan['baseline_sidecar']).read_text())
        identities = [(row['protocol_id'],row['evaluator_sha256'],row['gate_sha256'],
                       row['metric_definition_sha256']) for row in (baseline,results['A'],results['B'])]
        if len(set(identities)) != 1:
            raise ValueError('Baseline and branch evaluation identities differ')
        comparison = {'event_id':event,'protocol_id':baseline['protocol_id'],
            'baseline':plan['baseline_sidecar'],'branches':review_results,
            'source_checkpoint':comparable['source_checkpoint'],
            'turn_train_passed':{name:sum(point['new_passed'] for point in row['points']
                if point['group']=='turn_train') for name,row in results.items()},
            'retention_passed':{name:row['retention_points'] for name,row in results.items()},
            'per_point':[{name:next(point for point in results[name]['points']
                if point['group']=='turn_train' and point['command']==source['command'])
                for name in ('A','B')} for source in baseline['points']
                if source['group']=='turn_train']}
        atomic_json(output/'ablation_comparison.json',comparison)
        state={'status':'finished_pending_review','event_id':event,
            'stop_reason':'bounded_ablation_complete','branches':jobs,
            'comparison':str(output/'ablation_comparison.json')}
    except BaseException as exc:
        state={'status':'error','event_id':event,'stop_reason':'runtime_error',
            'error':str(exc),'branches':jobs}
    terminal = output/'plan_terminal'
    terminal.mkdir(exist_ok=True)
    atomic_json(terminal/'status.json',state)
    atomic_json(terminal/'terminal_summary.json',state)
    lines=['# A/B roll constraint experiment','',
        'event_id: '+event,'status: '+state['status'],
        'stop_reason: '+state['stop_reason'],
        'A and B start from the same model_38250 checkpoint; neither starts from the other.',
        'A uses roll coefficient 1.0; B uses 0.25 on turn slots only.',
        'Report generation is not model acceptance.']
    if state.get('comparison'):
        lines += ['comparison: '+state['comparison'],
            'A/B endpoint reviews: '+str(review_results)]
    if state.get('error'):
        lines += ['error: '+state['error']]
    (terminal/'completion_report.md').write_text('\n'.join(lines)+'\n')
    atomic_json(terminal/'report_ready.json',{'event_id':event,
        'report':str(terminal/'completion_report.md'),'ready_at':time.time()})
    if session:
        from wheel_legged_gym.app.completion import wake_session
        wake_session(terminal,session)


def submit_ablation_plan(root, plan_path, session):
    plan_path = plan_path.resolve(strict=True)
    plan = json.loads(plan_path.read_text())
    comparable = validate_ablation_plan(plan)
    output = Path(plan['output_root']).resolve(strict=True)
    if any((output/plan['branches'][name]['run_name']).exists() for name in ('A','B')):
        raise FileExistsError('An ablation branch job already exists')
    ledger = json.loads((output/'budget_ledger.json').read_text())
    if (ledger.get('formal_limit_iterations') != 10000 or
            ledger.get('formal_used_iterations') != 0 or
            ledger.get('source_checkpoint_sha256') != comparable['source_sha256']):
        raise ValueError('Ablation budget ledger identity differs')
    event = hashlib.sha256(('roll-ablation-v1:'+uuid.uuid4().hex).encode()).hexdigest()[:24]
    atomic_json(output/'plan_submission.json',{'status':'starting','event_id':event,
        'plan':str(plan_path),'plan_sha256':digest(plan_path),'hapi_session':session})
    env=dict(os.environ,PYTHONPATH=str(root/'plane')+os.pathsep+os.environ.get('PYTHONPATH',''))
    with (output/'plan_orchestrator.log').open('w') as log:
        owner=subprocess.Popen([sys.executable,str(Path(__file__)),
            '--_orchestrate_plan',str(plan_path), *(['--hapi-session',session] if session else [])],
            cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
            start_new_session=True,close_fds=True,env=env)
    deadline=time.monotonic()+240
    while time.monotonic()<deadline:
        marker=json.loads((output/'plan_submission.json').read_text())
        if marker['status']=='running':
            if owner.poll() is not None:
                raise RuntimeError('Plan owner exited during startup handshake')
            print(json.dumps(marker))
            return
        if owner.poll() is not None:
            raise RuntimeError('Plan startup failed; inspect '+str(output/'plan_orchestrator.log'))
        time.sleep(.2)
    terminate(owner)
    raise TimeoutError('Ablation plan startup handshake timed out')


def main(root, argv=None):
    root=Path(root).resolve()
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--spec',type=Path)
    parser.add_argument('--run-name')
    parser.add_argument('--new-iterations',type=int)
    parser.add_argument('--output-root',type=Path)
    parser.add_argument('--hapi-session')
    parser.add_argument('--num-envs',type=int,default=4096)
    parser.add_argument('--seed',type=int,default=23)
    parser.add_argument('--detach',action='store_true')
    parser.add_argument('--plan',type=Path)
    parser.add_argument('--_orchestrate_plan',type=Path,help=argparse.SUPPRESS)
    parser.add_argument('--_orchestrate',type=Path,help=argparse.SUPPRESS)
    args=parser.parse_args(argv)
    if args._orchestrate:
        orchestrate(root,args._orchestrate.resolve(strict=True))
        return
    if args._orchestrate_plan:
        orchestrate_ablation_plan(root,args._orchestrate_plan.resolve(strict=True),args.hapi_session)
        return
    if args.plan:
        if not args.detach:
            parser.error('Ablation plan requires --detach')
        submit_ablation_plan(root,args.plan,args.hapi_session)
        return
    if not all((args.spec,args.run_name,args.new_iterations,args.output_root,args.detach)):
        parser.error('A detached bounded job requires spec, run name, iterations and output root')
    if (not args.run_name.replace('_','').isalnum() or
            not 1000<=args.new_iterations<=30000 or args.new_iterations % 1000):
        parser.error('Invalid run name or iteration budget')
    spec=args.spec.resolve(strict=True)
    data=json.loads(spec.read_text())
    source=Path(data['source_checkpoint']).resolve(strict=True)
    if digest(source)!=data['source_sha256']:
        parser.error('Source checkpoint SHA mismatch')
    output_root=args.output_root.resolve(strict=True)
    ledger=(output_root/'budget_ledger.json').resolve(strict=True)
    job=output_root/args.run_name
    job.mkdir(exist_ok=False)
    attempt=uuid.uuid4().hex
    event=hashlib.sha256(('training-completion-v2:'+attempt).encode()).hexdigest()[:24]
    reserve(ledger,attempt,args.run_name,data['source_sha256'],args.new_iterations)
    command=[sys.executable,'wheel_legged_gym/scripts/train.py','--task=wheel_legged',
        '--headless',f'--num_envs={args.num_envs}',f'--max_iterations={args.new_iterations}',
        f'--seed={args.seed}',f'--run_name={args.run_name}',
        '--policy_experiment=TURN_LEAN_LONG','--resume','--resume_mode=full',
        f'--load_run={source.parent.name}',f'--checkpoint={data["source_iteration"]}']
    snapshot={'job':str(job),'run_name':args.run_name,'spec':str(spec),
        'spec_sha256':digest(spec),'source_checkpoint':str(source),
        'source_sha256':digest(source),'source_iteration':data['source_iteration'],
        'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        'actual_cli':command,'budget_ledger':str(ledger),'attempt_id':attempt,
        'event_id':event,'target_new_iterations':args.new_iterations,
        'seed':args.seed,'num_envs':args.num_envs,'hapi_session':args.hapi_session,
        'submitted_at':time.time()}
    atomic_json(job/'job_spec.json',data)
    atomic_json(job/'submit_snapshot.json',snapshot)
    environment=dict(os.environ)
    environment['PYTHONPATH']=str(root/'plane')+os.pathsep+environment.get('PYTHONPATH','')
    with (job/'orchestrator.log').open('w') as log:
        owner=subprocess.Popen([sys.executable,str(Path(__file__)),
            '--_orchestrate',str(job)],cwd=root,stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
            close_fds=True,env=environment)
    deadline=time.monotonic()+180
    while time.monotonic()<deadline:
        submission=job/'submission.json'
        if submission.is_file():
            status=json.loads(submission.read_text())
            if status.get('status')=='running':
                if owner.poll() is not None:
                    raise RuntimeError('Orchestrator exited after writing startup marker')
                monitor_status=Path(status['job'])/'status.json'
                if not monitor_status.is_file() or json.loads(
                        monitor_status.read_text()).get('status')!='running':
                    raise RuntimeError('Monitor startup marker is not running')
                print(json.dumps({**status,'source_sha256':snapshot['source_sha256'],
                    'spec_sha256':snapshot['spec_sha256']}))
                return
            if status.get('status')=='error':
                raise RuntimeError('Detached startup failed: '+str(status.get('error')))
        if owner.poll() is not None:
            raise RuntimeError('Orchestrator exited before startup handshake')
        time.sleep(.1)
    terminate(owner)
    raise TimeoutError('Detached startup handshake timed out')


if __name__=='__main__':
    main(Path(__file__).resolve().parents[3])
