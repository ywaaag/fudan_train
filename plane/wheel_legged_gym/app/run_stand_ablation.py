"""Matched control/treatment standing fine-tunes and compact deterministic audits."""
import json
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime



def main(root):
    root = Path(root)
    plane = root/'plane'
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--job', type=Path, help='Recover an interrupted job without replaying completed stages')
    args = parser.parse_args()
    job = args.job.resolve() if args.job else plane/'outputs'/('stand_ablation_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    job.mkdir(parents=True, exist_ok=bool(args.job))
    lock = (job/'supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    source = 'Sep17_17-41-11_stand_rand1_entropyfix_probe_sep17'
    env = dict(os.environ, PYTHONPATH=str(plane), CUDA_VISIBLE_DEVICES='0',
               LD_LIBRARY_PATH=str(Path(sys.executable).parent.parent/'lib')+':'+os.environ.get('LD_LIBRARY_PATH',''))
    state = {'status':'starting', 'source_run':source, 'source_checkpoint':200, 'probe_iterations':500, 'runs':{}}
    if args.job:
        state = json.loads((job/'status.json').read_text())
    # Historical jobs predate this field and used 200 iterations.
    probe_iterations = state.get('probe_iterations', 200)
    state['supervisor_pid'] = os.getpid()

    def save():
        tmp = job/'status.tmp'
        tmp.write_text(json.dumps(state, indent=2))
        tmp.replace(job/'status.json')

    def run(cmd, tag):
        state['stage'] = tag
        with (job/(tag+'.log')).open('w') as log:
            p = subprocess.Popen([sys.executable]+cmd, env=env, cwd=str(root), stdout=log, stderr=subprocess.STDOUT)
            state['child_pid'] = p.pid
            save()
            if p.wait() != 0:
                raise RuntimeError('Failed stage: '+tag)

    try:
        for profile in ['STAND_CONTROL', 'STAND_SYMMETRIC']:
            for label, count, iterations in [('smoke',64,1),('probe',4096,probe_iterations)]:
                name = job.name+'_'+profile.lower()+'_'+label
                state['status']='training'
                paths=list((plane/'logs/wheel_legged').glob('*_'+name))
                if len(paths)>1:
                    raise RuntimeError('Multiple existing runs for '+name)
                if paths:
                    pid = state.get('child_pid', 0)
                    proc = Path('/proc')/str(pid)/'cmdline'
                    while proc.exists() and ('--run_name='+name).encode() in proc.read_bytes().split(b'\0'):
                        time.sleep(5)
                    if not (paths[0]/f'model_{200+iterations}.pt').exists():
                        raise RuntimeError('Existing incomplete run needs inspection: '+str(paths[0]))
                else:
                    run([str(plane/'wheel_legged_gym/scripts/train.py'), '--task=wheel_legged', '--headless',
                     '--num_envs='+str(count), '--resume', '--resume_mode=full', '--load_run='+source,
                     '--checkpoint=200','--policy_experiment='+profile, '--max_iterations='+str(iterations),
                     '--seed=23','--run_name='+name],profile+'_'+label)
                paths=list((plane/'logs/wheel_legged').glob('*_'+name))
                if len(paths)!=1:
                    raise RuntimeError('Run identification ambiguous')
                if label=='probe':
                    state['runs'][profile]=str(paths[0])
                    run([str(root/'tools/summarize_training.py'),str(paths[0])],profile+'_summary')
            for seed in [19,37]:
                if (job/(profile+f'_seed{seed}.json')).exists():
                    continue
                state['status']='evaluating'
                run([str(plane/'wheel_legged_gym/scripts/evaluate_standing.py'), '--profile='+profile,
                     '--checkpoint='+str(paths[0]/f'model_{200+probe_iterations}.pt'), '--num-envs=32', '--randomization-level=1',
                     '--seed='+str(seed),'--out='+str(job/(profile+f'_seed{seed}.json'))],profile+f'_eval{seed}')
        state['status']='finished_pending_review'
    except Exception as exc:
        state.update(status='error',error=str(exc))
        raise
    finally:
        save()
