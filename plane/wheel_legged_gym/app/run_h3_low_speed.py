"""Bounded H3 migration training with recorded independent checkpoint audits."""
import sys
from pathlib import Path
import fcntl
import argparse
import json
import os
import subprocess
from datetime import datetime
from wheel_legged_gym.evaluation.gates import gate

def audit_grid(profile):
    vx = [0., -.5, .5, -1., 1.]
    if profile in {'LEGACY_SPEED2', 'LEGACY_SPEED2_STOP', 'LEGACY_YAW'}:
        vx += [-1.5, 1.5, -2., 2.]
    yaw = [0.] * len(vx)
    if profile == 'LEGACY_YAW':
        vx += [0., 0.]
        yaw += [-.5, .5]
    return vx, yaw

def main(root):
    ROOT = Path(root)
    PLANE = ROOT/'plane'
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--profile',choices=['H3_LOW_SPEED','H3_SPEED1','ENCODER_ANCHORED','ANCHORED_WHEEL_EXPLORE','EXPLORE_STOP_RETENTION','EXPLORE_CLEAN_OBS','LEGACY_ANCHORS','LEGACY_SPEED2','LEGACY_SPEED2_STOP','LEGACY_YAW'],default='H3_LOW_SPEED')
    opts=parser.parse_args()
    speed1=opts.profile!='H3_LOW_SPEED'
    source_run='Sep19_09-17-38_h3_low_speed_20260919_091730' if speed1 else 'Sep05_17-41-43_H3_from_H2_best_v1'
    source_iteration=100 if speed1 else 15800
    audit_iterations=[200,600] if speed1 else [100,500]
    if opts.profile in {'EXPLORE_STOP_RETENTION','EXPLORE_CLEAN_OBS'}:
        source_run='Sep19_12-13-14_anchored_wheel_explore_20260919_121307'
        source_iteration=500
        audit_iterations=[600,1000]
    if opts.profile=='LEGACY_ANCHORS':
        source_run='Sep19_11-22-46_legacy_urdf_20260919'
        source_iteration=500
        audit_iterations=[600,1000]
    if opts.profile=='LEGACY_SPEED2':
        source_run='Sep19_12-59-33_legacy_anchors_20260919_125925'
        source_iteration=700
        audit_iterations=[800,1200]
    if opts.profile=='LEGACY_SPEED2_STOP':
        source_run='Sep19_22-05-02_legacy_speed2_20260919_220453'; source_iteration=900
        audit_iterations=[1000,1400]
    if opts.profile=='LEGACY_YAW':
        source_run='Sep19_23-08-55_legacy_speed2_stop_20260919_230848'; source_iteration=1400
        audit_iterations=[1500,1900]
    audit_commands, audit_yaws = audit_grid(opts.profile)
    job = PLANE/'outputs'/((opts.profile.lower()+'_')+datetime.now().strftime('%Y%m%d_%H%M%S'))
    job.mkdir()
    lock = (PLANE/'outputs/h3_low_speed_training.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = dict(status='training', supervisor_pid=os.getpid(), run_name=job.name,
                 profile=opts.profile,seed=23,num_envs=4096,iterations=500,completed_audits=[])
    env = dict(os.environ,CUDA_VISIBLE_DEVICES='0',PYTHONPATH=str(PLANE),
               LD_LIBRARY_PATH=str(Path(sys.executable).parent.parent/'lib')+':'+os.environ.get('LD_LIBRARY_PATH',''))
    def save():
        tmp=job/'status.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(job/'status.json')
    def run(args, tag):
        cmd=[sys.executable]+args
        state['command']=cmd
        with (job/(tag+'.log')).open('w') as log:
            child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            state.update(stage=tag,child_pid=child.pid);save()
            try:
                result=child.wait()
            except BaseException:
                child.terminate();child.wait();raise
            if result:raise RuntimeError(f'{tag} exited {result}')
    print(job,flush=True)
    try:
        run([str(PLANE/'wheel_legged_gym/scripts/train.py'),'--task=wheel_legged','--headless',
             '--num_envs=4096','--resume','--resume_mode='+('full' if speed1 else 'policy'),
             '--load_run='+source_run,'--checkpoint='+str(source_iteration),
             '--policy_experiment='+opts.profile,'--max_iterations=500','--seed=23',
             '--run_name='+job.name],'train')
        folders=list((PLANE/'logs/wheel_legged').glob('*_'+job.name))
        if len(folders)!=1:raise RuntimeError('Ambiguous training run')
        folder=folders[0];state.update(status='evaluating',run_dir=str(folder));save()
        run([str(ROOT/'tools/summarize_training.py'),str(folder),'--window=100'],'training_summary')
        # Check both early retention and final policy; never select latest on reward alone.
        summary={}
        for iteration in audit_iterations:
            records=[]
            for seed in [19,37,53]:
                tag=f'model{iteration}_seed{seed}';out=job/(tag+'.json')
                run([str(PLANE/'wheel_legged_gym/scripts/evaluate_policy_comparison.py'),
                     '--checkpoint='+str(folder/f'model_{iteration}.pt'),'--out='+str(out),
                     '--randomization-level=1','--seed='+str(seed),
                     '--commands']+[str(v) for v in audit_commands]+
                     ['--yaw-commands']+[str(w) for w in audit_yaws],tag)
                data=json.loads(out.read_text());records.append(data)
                state['completed_audits'].append(tag);save()
            low=[r for d in records for r in d['results'] if abs(r['command'][0])<=.5]
            summary[str(iteration)]={'low_speed_passed':all(gate(r)['passed'] for r in low),
                'full_grid_passed':all(gate(r)['passed'] for d in records for r in d['results']),
                'records':[{**r,'seed':d['seed'],'gate':gate(r)} for d in records for r in d['results']]}
            (job/'acceptance.json').write_text(json.dumps(summary,indent=2)+'\n')
        last=audit_iterations[-1]
        state.update(status='finished_pending_review',
                     gate_passed=summary[str(last)]['full_grid_passed' if speed1 else 'low_speed_passed'],
                     checkpoint=str(folder/f'model_{last}.pt'),child_pid=None)
        if not state['gate_passed']:state['status']='paused_on_regression'
    except BaseException as exc:
        state.update(status='error',error=str(exc),child_pid=None);raise
    finally:save()
