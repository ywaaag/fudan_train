"""Two isolated same-source parking experiments, followed by checkpoint screening."""
from wheel_legged_gym.adapters.processes.cooperative_job import run_child
import json
import os
import sys
import time
import fcntl
import subprocess
from pathlib import Path
from datetime import datetime
from wheel_legged_gym.evaluation.gates import gate



def main(root):
    ROOT = Path(root)
    OUTPUT = ROOT/'plane/outputs'
    lock=(OUTPUT/'fixed_height_diagnosis.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    job=OUTPUT/('fixed_height_diagnosis_'+datetime.now().strftime('%Y%m%d_%H%M%S'));job.mkdir()
    source=ROOT/'plane/logs/wheel_legged/Sep20_03-42-29_motion_goal_20260920_002043_r24_yaw4/model_6900.pt'
    state={'status':'running','supervisor_pid':os.getpid(),'child_pid':None,'source':str(source),
           'results':{},'scope':'Fixed parking diagnostic only; original motion baseline retained'}
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',PYTHONPATH=str(ROOT/'plane'),
             LD_LIBRARY_PATH=str(Path(sys.executable).parent.parent/'lib'))
    def save():
        tmp=job/'status.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(job/'status.json')
        (job/'progress.md').write_text('# 固定高度隔离实验\n\n'+json.dumps(state,ensure_ascii=False,indent=2)+'\n')
    def child_preparing(tag):
        state['operation'] = tag

    def child_started(pid):
        state['child_pid'] = pid
        save()

    def child_finished():
        state['child_pid'] = None
        save()

    def run(args, tag, childjob=None):
        run_child(args, tag, childjob, executable=sys.executable, root=ROOT,
                  job=job, environment=env, on_preparing=child_preparing,
                  on_started=child_started, on_finished=child_finished)
    print(job,flush=True);save()
    try:
        for stage,height in [('fixed35',.35),('fixed45',.45)]:
            childjob=job/stage;state.update(stage=stage,child_job=str(childjob));save()
            run([ROOT/'tools/run_height_course.py','--stage='+stage,'--source='+str(source),
                 '--job='+str(childjob),'--variable-repeats=1','--height-reward-gain=1','--seed=23'],stage,childjob)
            result=json.loads((childjob/(stage+'_acceptance.json')).read_text())
            state['results'][stage]={'final':{k:v for k,v in result.items() if k!='records'},
                                     'candidates':[]};save()
        # Run both full experiments even if the first fails its gate.
        for stage,height in [('fixed35',.35),('fixed45',.45)]:
            childjob=job/stage
            folder=Path(json.loads((childjob/'status.json').read_text())['run_dir'])
            for iteration in [7000,7100,7200,7300]:
                records=[]
                for seed in [19,37,53]:
                    tag=f'{stage}_m{iteration}_seed{seed}';out=job/(tag+'.json')
                    run([ROOT/'plane/wheel_legged_gym/scripts/evaluate_policy_comparison.py',
                         '--checkpoint='+str(folder/f'model_{iteration}.pt'),'--out='+str(out),
                         '--commands','0','--height-commands',str(height),'--seed='+str(seed)],tag)
                    row=json.loads(out.read_text())['results'][0]
                    passed=gate(row)['passed'] and row['metrics']['height_mae']<=.015 and row['nonwheel_contact_full_fraction']==0
                    records.append({'seed':seed,'passed':passed,'metrics':row['metrics'],
                                    'failure_count':row['failure_count'],'joint_diagnostics':row['joint_diagnostics']})
                    if not passed:break
                state['results'][stage]['candidates'].append({'checkpoint':str(folder/f'model_{iteration}.pt'),
                    'passed':len(records)==3 and all(r['passed'] for r in records),'records':records});save()
        both=all(v['final']['passed'] or any(c['passed'] for c in v['candidates']) for v in state['results'].values())
        state['status']='fixed_heights_passed_ready_for_conditional_design' if both else 'fixed_height_limit_found_needs_diagnosis'
        save()
    except BaseException as exc:
        state.update(status='stopped' if isinstance(exc,InterruptedError) else 'error',error=str(exc));save();raise

