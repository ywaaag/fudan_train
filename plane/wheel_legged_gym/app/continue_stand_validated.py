"""One bounded 500-iteration continuation after the matched small-push audit."""

import json
import argparse
import os
from pathlib import Path
import subprocess
import sys
import shutil
from datetime import datetime


def main(root):
    root = Path(root)
    plane = root/'plane'
    additional_iterations = 500
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--source-job', type=Path, required=True, help='Completed three-seed audit job of the exact checkpoint to resume')
    args = parser.parse_args()
    preflight = json.loads((args.source_job/'status.json').read_text())
    if preflight['status'] != 'finished_pending_review':
        raise RuntimeError('Source job has not finished')
    source_model = Path(preflight['checkpoint']).resolve(strict=True)
    audits = [json.loads(Path(p).read_text()) for p in preflight['completed_audits']]
    if len(audits)!=3 or {a['seed'] for a in audits}!={19,37,53}:
        raise RuntimeError('Require three distinct audit seeds')
    for audit in audits:
        recoveries = [t for push in audit['push_results'] for t in push['recovery_seconds']]
        if (Path(audit['checkpoint']).resolve()!=source_model or audit['failure_count']!=0
            or audit['timeout_count']!=0 or audit['metrics']['nonwheel_contact_fraction']['mean']!=0
            or len(recoveries)!=160 or not all(t is not None for t in recoveries)):
            raise RuntimeError('Source checkpoint has not passed all impulse trials')
    source_iteration = int(source_model.stem.split('_')[-1])
    job=plane/'outputs'/('stand_validated_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    job.mkdir()
    env=dict(os.environ, CUDA_VISIBLE_DEVICES='0', PYTHONPATH=str(plane),
             LD_LIBRARY_PATH=str(Path(sys.executable).parent.parent/'lib')+':'+os.environ.get('LD_LIBRARY_PATH',''))
    state={'status':'training','completed_audits':[], 'additional_iterations':additional_iterations,
           'source_checkpoint':str(source_model), 'source_job':str(args.source_job.resolve()),
           'supervisor_pid':os.getpid()}
    def save():
        tmp=job/'status.tmp';tmp.write_text(json.dumps(state,indent=2));tmp.replace(job/'status.json')
    def run(cmd,tag):
        with (job/(tag+'.log')).open('w') as log:
            p=subprocess.Popen([sys.executable]+cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
            state.update(child_pid=p.pid,stage=tag);save()
            if p.wait():raise RuntimeError('Stage failed: '+tag)
    try:
        save()
        codex = shutil.which('codex')
        if codex:
            with (job/'completion_watcher.log').open('w') as log:
                watcher = subprocess.Popen([sys.executable,str(root/'tools/training_completion_hook.py'),
                    '--job',str(job),'--codex',codex],stdin=subprocess.DEVNULL,stdout=log,
                    stderr=subprocess.STDOUT,start_new_session=True)
            state['completion_watcher_pid']=watcher.pid
        name=job.name
        run([str(plane/'wheel_legged_gym/scripts/train.py'),'--task=wheel_legged','--headless','--num_envs=4096',
             '--resume','--resume_mode=full',
             '--load_run='+source_model.parent.name,'--checkpoint='+str(source_iteration),
             '--policy_experiment=STAND_SYMMETRIC','--max_iterations='+str(additional_iterations),'--seed=23','--run_name='+name],'train')
        paths=list((plane/'logs/wheel_legged').glob('*_'+name))
        if len(paths)!=1:raise RuntimeError('Ambiguous run')
        state['checkpoint']=str(paths[0]/f'model_{source_iteration+additional_iterations}.pt')
        run([str(root/'tools/summarize_training.py'),str(paths[0])],'summary')
        state['status']='evaluating'
        for seed in [19,37,53]:
            out=job/f'audit_{seed}.json'
            run([str(plane/'wheel_legged_gym/scripts/evaluate_standing.py'),'--profile=STAND_SYMMETRIC',
                 '--checkpoint='+state['checkpoint'],'--num-envs=32','--randomization-level=1',
                 '--seconds=60','--warmup=5','--push-delta-v=0.10','--seed='+str(seed),'--out='+str(out)],f'eval_{seed}')
            state['completed_audits'].append(str(out));save()
        state['status']='finished_pending_review'
    except Exception as exc:
        state.update(status='error',error=str(exc));raise
    finally:save()
