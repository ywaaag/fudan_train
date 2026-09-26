"""Bounded checkpoint summaries and small probes for one turn training run."""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from wheel_legged_gym.adapters.artifacts.job_files import JobFiles
from wheel_legged_gym.adapters.artifacts.tensorboard_scalars import read_scalars
from wheel_legged_gym.app.completion import wake_session, write_report
from wheel_legged_gym.evaluation.gates import gate
from wheel_legged_gym.evaluation.training_summary import (
    METRIC_KEYS, summarize_scalars, through_iteration,
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def screen(data, stage=None):
    if stage is not None and stage.get('cohort_plan') is not None:
        rows = data['results']
        if len(rows) != 23:
            raise ValueError('Cornering-height probe requires 23 fixed commands')
        def summarize(group):
            return [{'command':row['command'], 'original_gate':gate(row),
                     'height_mae':row['metrics']['height_mae'],
                     'root_height':row['metrics']['height'],
                     'com_height':row['metrics']['com_height'],
                     'yaw_mae':row['metrics']['yaw_mae'],
                     'vx_mae':row['metrics']['vx_mae'],
                     'roll':row['metrics']['roll'],
                     'slip_rms':row['metrics']['slip_rms'],
                     'min_wheel_contact':min(row['metrics']['left_contact'],row['metrics']['right_contact']),
                     'failure_count':row['failure_count']} for row in group]
        retention, height, mid, high = map(summarize,
            (rows[:5], rows[5:11], rows[11:19], rows[19:]))
        return {'retention_passed':sum(r['original_gate']['passed'] for r in retention),
                'retention_total':5, 'height_skill_passed':sum(
                    r['original_gate']['passed'] and r['height_mae']<=.015
                    for r in height), 'height_skill_total':6,
                'mid_turn_passed':sum(r['original_gate']['passed'] for r in mid),
                'mid_turn_total':8,
                'high_turn_passed':sum(r['original_gate']['passed'] for r in high),
                'high_turn_total':4,
                'retention_rows':retention,'height_rows':height,
                'mid_rows':mid,'high_rows':high}
    retention, turns = data['results'][:5], data['results'][5:]
    checked = []
    for row in turns:
        m = row['metrics']
        posture = (m['height_mae'] <= .015 and
                   abs(m['roll']-m['roll_target']) <= .035)
        checked.append({'command':row['command'], 'original_gate':gate(row),
                        'height_and_roll_learned':posture,
                        'yaw_mae':m['yaw_mae'], 'vx_mae':m['vx_mae'],
                        'height':m['height'], 'height_mae':m['height_mae'],
                        'roll':m['roll'], 'roll_target':m['roll_target'],
                        'slip_rms':m['slip_rms'], 'failure_count':row['failure_count']})
    return {'retention_passed':sum(gate(row)['passed'] for row in retention),
            'retention_total':len(retention),
            'turn_posture_learned':sum(row['height_and_roll_learned'] for row in checked),
            'turn_total':len(checked), 'turn_rows':checked,
            'mean_turn_yaw_mae':sum(row['yaw_mae'] for row in checked)/len(checked),
            'mean_turn_height_mae':sum(row['height_mae'] for row in checked)/len(checked)}


def probe_command(checkpoint, out, stage):
    if stage.get('cohort_plan') is not None:
        height_pairs = [(v,0.,h) for h in (.38,.36) for v in (0.,-.5,.5)]
        mid_pairs = [(sv*v,sw*w,.38) for v,w in ((2.5,.6),(3.,.7))
                     for sv in (-1.,1.) for sw in (-1.,1.)]
        high_pairs = [(sv*3.,sw*1.,.36) for sv in (-1.,1.) for sw in (-1.,1.)]
        pairs = [(0.,0.,.4),(-4.,0.,.4),(4.,0.,.4),(0.,-4.,.4),(0.,4.,.4)] + height_pairs + mid_pairs + high_pairs
        speeds,yaw,height = zip(*pairs)
    else:
        turn_height = stage['turn_height']
        turn_pairs = [(-1.,-.5),(-1.,.5),(1.,-.5),(1.,.5),(-2.,.5),(2.,.5)]
        speeds = [0.,-4.,4.,0.,0.] + [pair[0] for pair in turn_pairs]
        yaw = [0.,0.,0.,-4.,4.] + [pair[1] for pair in turn_pairs]
        height = [.4]*5 + [turn_height]*len(turn_pairs)
    initial = [0.]*len(speeds)
    initial_height = [.4]*len(speeds)
    return [sys.executable, 'wheel_legged_gym/scripts/evaluate_policy_comparison.py',
            '--checkpoint', str(checkpoint), '--out', str(out),
            '--commands', *map(str,speeds), '--yaw-commands', *map(str,yaw),
            '--height-commands', *map(str,height),
            '--initial-commands', *map(str,initial),
            '--initial-height-commands', *map(str,initial_height),
            '--switch-at', '1', '--transition-ramp-seconds', str(stage['speed_ramp_seconds']),
            '--yaw-delay-seconds', str(stage['speed_ramp_seconds']),
            '--height-delay-seconds', str(stage['speed_ramp_seconds']),
            '--lean-reference-max-deg', str(stage['lean_max_deg']),
            '--envs-per-command', '4', '--seed', '19', '--seconds',
            '18' if stage.get('cohort_plan') is not None else '12',
            '--warmup', '8' if stage.get('cohort_plan') is not None else '7',
            '--trace-stride', '100', '--profile', 'method_v1',
            '--randomization-level', '1']


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def main(root):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--trainer-pid', type=int, required=True)
    parser.add_argument('--new-iterations', type=int, required=True)
    parser.add_argument('--poll-seconds', type=int, default=20)
    args = parser.parse_args()
    job, run = args.job.resolve(strict=True), args.run.resolve(strict=True)
    lock = (job / 'turn_lean_monitor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    spec = json.loads(args.spec.read_text())
    source = spec['source_iteration']
    if args.new_iterations < 1000 or args.new_iterations > 30000 or args.new_iterations % 1000:
        parser.error('new iterations must be a bounded multiple of 1000')
    files = JobFiles(job)
    probes = job / 'probes'
    probes.mkdir(exist_ok=True)
    summaries = job / 'summaries'
    summaries.mkdir(exist_ok=True)
    state = {'status':'running', 'supervisor_pid':os.getpid(),
             'child_pid':args.trainer_pid, 'source':spec['source_checkpoint'],
             'stage':spec['stage']['phase'], 'run':str(run),
             'new_iteration_limit':args.new_iterations,
             'milestones':[], 'accepted':None}
    previous_status = job / 'status.json'
    if previous_status.exists():
        previous = json.loads(previous_status.read_text())
        if (previous.get('status') != 'running' or previous.get('run') != str(run)
                or previous.get('child_pid') != args.trainer_pid
                or previous.get('new_iteration_limit') != args.new_iterations):
            raise ValueError('Monitor restart does not match the active job')
        prior_pid = previous.get('supervisor_pid')
        if prior_pid and prior_pid != os.getpid() and alive(prior_pid):
            raise RuntimeError('Another monitor still owns this job')
        state = previous
        state['supervisor_pid'] = os.getpid()
    files.save_status(state)
    collapsed = 0
    for entry in state['milestones'][-2:]:
        prior = json.loads(Path(entry['probe']).read_text())
        collapsed = collapsed+1 if prior['retention_passed'] < 4 else 0
    try:
        for increment in range(1000, args.new_iterations+1, 1000):
            target_iteration = source + increment
            iteration = (target_iteration if increment == args.new_iterations else
                         ((target_iteration + 249) // 250) * 250)
            if any(entry['target_iteration'] == target_iteration
                   for entry in state['milestones']):
                continue
            checkpoint = run / ('model_{}.pt'.format(iteration))
            while not checkpoint.is_file():
                if (job / 'STOP').exists():
                    os.kill(args.trainer_pid, signal.SIGINT)
                    raise InterruptedError('STOP requested')
                if not alive(args.trainer_pid):
                    raise RuntimeError('Trainer exited before checkpoint {}'.format(iteration))
                time.sleep(args.poll_seconds)
            # torch.save writes directly to the final path; wait for stable bytes.
            previous = -1
            while checkpoint.stat().st_size != previous:
                previous = checkpoint.stat().st_size
                time.sleep(2)
            checkpoint_sha = digest(checkpoint)
            scalar = summarize_scalars(run,
                through_iteration(read_scalars(run, METRIC_KEYS), iteration), 50)
            values = [field['tail_mean'] for field in scalar['metrics'].values()]
            if not values or not all(math.isfinite(value) for value in values):
                os.kill(args.trainer_pid, signal.SIGINT)
                raise ValueError('Nonfinite training scalar at {}'.format(iteration))
            summary_path = summaries / ('iter_{}.json'.format(iteration))
            if not summary_path.exists():
                with summary_path.open('x') as output:
                    output.write(json.dumps({'target_iteration':target_iteration,
                        'checkpoint':str(checkpoint),'checkpoint_sha256':checkpoint_sha,
                        'summary':scalar}, indent=2)+'\n')
            elif json.loads(summary_path.read_text())['checkpoint_sha256'] != checkpoint_sha:
                raise ValueError('Prior milestone summary checksum differs')
            out = probes / ('iter_{}.json'.format(iteration))
            command = probe_command(checkpoint, out, spec['stage'])
            if not out.exists():
                with (probes / ('iter_{}.log'.format(iteration))).open('w') as log:
                    subprocess.run(command, cwd=Path(root)/'plane',
                        stdout=log, stderr=subprocess.STDOUT, check=True)
            data = json.loads(out.read_text())
            if data['checkpoint_sha256'] != checkpoint_sha:
                raise ValueError('Probe checksum differs from checkpoint')
            result = screen(data, spec['stage'])
            probe_summary_path = probes / ('iter_{}_summary.json'.format(iteration))
            if not probe_summary_path.exists():
                with probe_summary_path.open('x') as output:
                    output.write(json.dumps({'target_iteration':target_iteration,
                        'checkpoint':str(checkpoint),'checkpoint_sha256':checkpoint_sha,
                        **result},indent=2)+'\n')
            elif json.loads(probe_summary_path.read_text())['checkpoint_sha256'] != checkpoint_sha:
                raise ValueError('Prior probe summary checksum differs')
            collapsed = collapsed+1 if result['retention_passed'] < 4 else 0
            state['milestones'].append({'target_iteration':target_iteration,
                'checkpoint_iteration':iteration,
                'checkpoint_sha256':checkpoint_sha,
                'summary':str(summary_path), 'probe':str(probe_summary_path)})
            state['latest_iteration'] = iteration
            files.save_status(state)
            if collapsed >= 3:
                os.kill(args.trainer_pid, signal.SIGINT)
                raise RuntimeError('Retention probe failed three consecutive milestones')
        while alive(args.trainer_pid):
            time.sleep(args.poll_seconds)
        state.update(status='finished_pending_review', child_pid=None)
    except InterruptedError as exc:
        state.update(status='stopped', error=str(exc), child_pid=None)
    except Exception as exc:
        if alive(args.trainer_pid):
            os.kill(args.trainer_pid, signal.SIGINT)
        state.update(status='error', error=str(exc), child_pid=None)
    files.save_status(state)
    write_report(job, state)
    session_id = os.environ.get('HAPI_SESSION_ID')
    if session_id:
        wake_session(job, session_id)
    print(json.dumps({'status':state['status'],'latest_iteration':state.get('latest_iteration'),
                      'error':state.get('error')}))
