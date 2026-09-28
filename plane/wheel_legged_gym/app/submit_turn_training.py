"""Submit one bounded turn training job to a detached process owner."""
import argparse
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
        exact = returncode == 0 and actual == target
        settle(ledger,attempt,actual,str(monitor_job/'status.json'),unknown=not exact)
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
    parser.add_argument('--_orchestrate',type=Path,help=argparse.SUPPRESS)
    args=parser.parse_args(argv)
    if args._orchestrate:
        orchestrate(root,args._orchestrate.resolve(strict=True))
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
