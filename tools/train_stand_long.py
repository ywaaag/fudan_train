"""Four full-resume standing segments, compact summaries and audit gates.

Run with fudan_leg Python. Each 500-iteration segment preserves optimizer
state; independent audits run between segments to avoid GPU contention.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import shutil
from datetime import datetime


def main():
    root = Path(__file__).resolve().parents[1]
    plane = root / 'plane'
    logs = plane / 'logs/wheel_legged'
    job = plane / 'outputs' / ('stand_long_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    job.mkdir(parents=True)
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES='0', FUDAN_STAND_RANDOMIZATION_LEVEL='1',
               PYTHONPATH=str(plane), LD_LIBRARY_PATH=str(Path(sys.executable).parent.parent / 'lib') + ':' + env.get('LD_LIBRARY_PATH', ''))
    run = 'Sep17_17-41-11_stand_rand1_entropyfix_probe_sep17'
    checkpoint = 200
    state = {'job': str(job), 'status': 'starting', 'supervisor_pid': os.getpid(), 'segments': []}

    def save():
        temporary = job / 'status.tmp'
        temporary.write_text(json.dumps(state, indent=2))
        temporary.replace(job / 'status.json')

    def execute(command, output):
        with output.open('w') as stream:
            process = subprocess.Popen(command, cwd=str(root), env=env, stdout=stream, stderr=subprocess.STDOUT)
            state['child_pid'] = process.pid
            state['command'] = command
            save()
            code = process.wait()
        if code:
            raise RuntimeError(f'exit={code}: see {output}')

    try:
        save()
        codex = shutil.which('codex')
        if codex:
            with (job / 'completion_watcher.log').open('a') as hook_log:
                watcher = subprocess.Popen([sys.executable, str(root/'tools/training_completion_hook.py'),
                    '--job', str(job), '--codex', codex], cwd=str(root), env=env,
                    stdin=subprocess.DEVNULL, stdout=hook_log, stderr=subprocess.STDOUT,
                    start_new_session=True)
            state['completion_watcher_pid'] = watcher.pid
            save()
        for segment in range(1, 5):
            name = job.name + f'_segment{segment}'
            state.update(status='training', segment=segment)
            execute([sys.executable, str(plane/'wheel_legged_gym/scripts/train.py'),
                '--task=wheel_legged', '--headless', '--num_envs=4096', '--resume',
                '--resume_mode=full', '--load_run='+run, '--checkpoint='+str(checkpoint),
                '--policy_experiment=method_v1', '--phase=stand', '--command_level=0',
                '--max_iterations=500', '--seed=23', '--run_name='+name], job/f'train_{segment}.log')
            matches = list(logs.glob('*_'+name))
            if len(matches) != 1:
                raise RuntimeError(f'ambiguous run: {matches}')
            run = matches[0].name
            checkpoint += 500
            model = logs/run/f'model_{checkpoint}.pt'
            if not model.is_file():
                raise FileNotFoundError(model)
            execute([sys.executable, str(root/'tools/summarize_training.py'), str(logs/run)], job/f'summary_{segment}.json')
            state['status'] = 'evaluating'
            audit = job/f'audit_{segment}.json'
            execute([sys.executable, str(plane/'wheel_legged_gym/scripts/evaluate_standing.py'),
                '--checkpoint', str(model), '--randomization-level', '1', '--num-envs', '32',
                '--seed', '19', '--seconds', '25', '--warmup', '5', '--out', str(audit)], job/f'eval_{segment}.log')
            report = json.loads(audit.read_text())
            metrics = report['metrics']
            # Guard against regression. These are stop thresholds, not a
            # declaration that the wider locomotion goal has been achieved.
            passed = (report['failure_count'] == 0 and report['timeout_count'] == 0
                and metrics['vx_m_s']['mean_abs'] <= 0.08
                and metrics['yaw_rad_s']['mean_abs'] <= 0.05
                and abs(metrics['height_m']['mean'] - 0.40) <= 0.025
                and min(metrics[k]['mean'] for k in ('left_contact','right_contact')) >= 0.99
                and max(metrics[k]['mean_abs'] for k in ('roll_rad','pitch_rad')) <= 0.10
                and max(metrics[k]['mean_abs'] for k in ('leg0_difference_rad','leg1_difference_rad')) <= 0.30)
            state['segments'].append({'run': run, 'checkpoint': checkpoint, 'audit': str(audit), 'guard_passed': passed})
            save()
            if not passed:
                state['status'] = 'paused_on_regression'
                save()
                return
        state['status'] = 'finished_pending_review'
    except Exception as exc:
        state.update(status='error', error=str(exc))
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
