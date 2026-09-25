"""One-policy two-height parking, then no-reset transition checks if steady gates pass."""
import sys
from pathlib import Path
import json
import os
import time
import subprocess
import fcntl
from datetime import datetime
from wheel_legged_gym.evaluation.gates import gate
from wheel_legged_gym.evaluation.transitions import summarize

def main(root):
    ROOT = Path(root)
    lock=(ROOT/'plane/outputs/dual_height.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    job=ROOT/'plane/outputs'/('dual_height_'+datetime.now().strftime('%Y%m%d_%H%M%S'));job.mkdir()
    childjob=job/'training'
    state={'status':'training','supervisor_pid':os.getpid(),'child_pid':None,'child_job':str(childjob),
           'scope':'Single policy dual-height parking only; original motion baseline retained'}
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',PYTHONPATH=str(ROOT/'plane'),
             LD_LIBRARY_PATH=str(Path(sys.executable).parent.parent/'lib'))
    def save():
        tmp=job/'status.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(job/'status.json')
    def run(args,tag,training=False):
        if (job/'STOP').exists():raise InterruptedError('User STOP')
        with (job/(tag+'.log')).open('w') as log:
            child=subprocess.Popen([sys.executable]+list(map(str,args)),cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            state.update(operation=tag,child_pid=child.pid);save()
            try:
                while child.poll() is None:
                    if (job/'STOP').exists():
                        if training and childjob.exists():(childjob/'STOP').touch()
                        else:child.terminate()
                    time.sleep(2)
                if (job/'STOP').exists():raise InterruptedError('User STOP')
                if child.returncode:raise RuntimeError(f'{tag} exited {child.returncode}')
            finally:
                if child.poll() is None:
                    if training and childjob.exists():(childjob/'STOP').touch()
                    else:child.terminate()
                    child.wait()
                state['child_pid']=None;save()
    print(job,flush=True);save()
    try:
        run([ROOT/'tools/run_height_course.py','--stage=dual','--job='+str(childjob)],'train_dual',True)
        result=json.loads((childjob/'dual_acceptance.json').read_text())
        state['steady_result']={k:v for k,v in result.items() if k!='records'}
        if not result['passed']:
            state['status']='paused_dual_height_steady_failed';save();return
        state['status']='evaluating_transitions';save()
        records=[];responses=[]
        for seed in [19,37,53]:
            out=job/f'transition_seed{seed}.json'
            run([ROOT/'plane/wheel_legged_gym/scripts/evaluate_policy_comparison.py',
                 '--checkpoint='+result['checkpoint'],'--out='+str(out),'--seed='+str(seed),
                 '--commands','0','0','--height-commands','.45','.35','--initial-commands','0','0',
                 '--initial-height-commands','.35','.45','--switch-at=5','--warmup=10'],f'transition_seed{seed}')
            d=json.loads(out.read_text());responses.append(summarize(d))
            records += [dict(r,seed=seed) for r in d['results']]
        passed=all(gate(r)['passed'] and r['metrics']['height_mae']<=.015 and
                   r['nonwheel_contact_full_fraction']==0 for r in records)
        settled=all(axis['settled_envs']==axis['total_envs'] and axis['worst_settling_s'] is not None
                    and axis['worst_settling_s']<=3 for d in responses for r in d['rows'] for axis in r['response'].values())
        (job/'transition_acceptance.json').write_text(json.dumps({'passed':passed and settled,
            'steady_passed':passed,'response_passed':settled,'records':records,'response':responses},indent=2)+'\n')
        state.update(status='dual_parking_passed_pending_motion_reintroduction' if passed and settled else
                     'paused_transition_failed',transition_passed=passed and settled);save()
    except BaseException as exc:
        state.update(status='stopped' if isinstance(exc,InterruptedError) else 'error',error=str(exc));save();raise
