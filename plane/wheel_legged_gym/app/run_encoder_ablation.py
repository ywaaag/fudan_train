"""Matched 500-iteration frozen/updating encoder experiment; no auto-promotion."""
import sys
from pathlib import Path
import fcntl
import json
import os
import subprocess
from datetime import datetime
from wheel_legged_gym.evaluation.gates import gate



def main(root):
    ROOT = Path(root)
    PLANE=ROOT/'plane'
    SOURCE='Sep19_09-17-38_h3_low_speed_20260919_091730'
    lock=(PLANE/'outputs/h3_low_speed_training.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    job=PLANE/'outputs'/('encoder_ablation_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    job.mkdir()
    state={'status':'training','supervisor_pid':os.getpid(),'source_run':SOURCE,
           'source_iteration':100,'seed':23,'num_envs':4096,'additional_iterations_per_branch':500,
           'runs':{},'completed_audits':[]}
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',PYTHONPATH=str(PLANE),
             LD_LIBRARY_PATH=str(Path(sys.executable).parent.parent/'lib')+':'+os.environ.get('LD_LIBRARY_PATH',''))
    def save():
        t=job/'status.tmp';t.write_text(json.dumps(state,indent=2)+'\n');t.replace(job/'status.json')
    def run(args,tag):
        cmd=[sys.executable]+args
        with (job/(tag+'.log')).open('w') as log:
            child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            state.update(stage=tag,child_pid=child.pid,command=cmd);save()
            try:code=child.wait()
            except BaseException:
                child.terminate();child.wait();raise
            if code:raise RuntimeError(f'{tag} exited {code}')
    print(job,flush=True)
    summary={}
    try:
        for branch in ['frozen','updating']:
            name=job.name+'_'+branch
            state['status']='training'
            run([str(PLANE/'wheel_legged_gym/scripts/train.py'),'--task=wheel_legged','--headless',
                 '--num_envs=4096','--resume','--resume_mode=full','--load_run='+SOURCE,'--checkpoint=100',
                 '--policy_experiment=ENCODER_'+branch.upper(),'--max_iterations=500','--seed=23',
                 '--run_name='+name],branch+'_train')
            folders=list((PLANE/'logs/wheel_legged').glob('*_'+name))
            if len(folders)!=1:raise RuntimeError('Ambiguous run')
            folder=folders[0];state['runs'][branch]=str(folder);state['status']='evaluating';save()
            for iteration in [200,600]:
                rows=[]
                for seed in [19,37,53]:
                    tag=f'{branch}_{iteration}_seed{seed}';out=job/(tag+'.json')
                    run([str(PLANE/'wheel_legged_gym/scripts/evaluate_policy_comparison.py'),
                         '--checkpoint='+str(folder/f'model_{iteration}.pt'),'--out='+str(out),
                         '--randomization-level=1','--seed='+str(seed),'--commands','0','-0.5','0.5','-1','1'],tag)
                    d=json.loads(out.read_text())
                    rows.extend(dict(seed=seed,**r,gate=gate(r)) for r in d['results'])
                    state['completed_audits'].append(tag);save()
                summary[f'{branch}_{iteration}']={'passed':all(r['gate']['passed'] for r in rows),
                    'low_speed_passed':all(r['gate']['passed'] for r in rows if abs(r['command'][0])<=.5),
                    'rows':rows}
                (job/'acceptance.json').write_text(json.dumps(summary,indent=2)+'\n')
                print(branch,iteration,summary[f'{branch}_{iteration}']['passed'],flush=True)
        state.update(status='finished_pending_review',child_pid=None)
    except BaseException as exc:
        state.update(status='error',error=str(exc),child_pid=None);raise
    finally:save()
