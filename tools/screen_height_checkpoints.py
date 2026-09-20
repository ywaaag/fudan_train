"""Screen all missed micro checkpoints using the unchanged full command grid."""
import json
import os
import sys
import subprocess
import fcntl
from pathlib import Path
from datetime import datetime
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'plane'))
import isaacgym
from wheel_legged_gym.envs.wheel_legged.height_course import height_bank
from compare_policy_versions import gate


def main():
    lock=(ROOT/'plane/outputs/h3_low_speed_training.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    job=ROOT/'plane/outputs'/('height_screen_'+datetime.now().strftime('%Y%m%d_%H%M%S'));job.mkdir()
    previous=ROOT/'plane/outputs/height_continuation_20260920_110521'
    state={'status':'screening','pid':os.getpid(),'child_pid':None,'results':[]}
    def save():
        tmp=job/'status.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(job/'status.json')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',PYTHONPATH=str(ROOT/'plane'),
             LD_LIBRARY_PATH=str(Path(sys.executable).parent.parent/'lib'))
    bank=sorted(set(height_bank('micro',4)))
    print(job,flush=True);save()
    try:
        for n in [1,2,3]:
            cs=json.loads((previous/f'round{n:02d}_micro/status.json').read_text())
            folder=Path(cs['run_dir'])
            for it in [7000,7100,7200,7300]:
                tag=f'r{n}_m{it}';records=[]
                for seed in [19,37,53]:
                    out=job/f'{tag}_seed{seed}.json'
                    cmd=([sys.executable,str(ROOT/'plane/wheel_legged_gym/scripts/evaluate_policy_comparison.py'),
                         '--checkpoint='+str(folder/f'model_{it}.pt'),'--out='+str(out),'--seed='+str(seed),
                         '--commands']+[str(v) for v,w,h in bank]+['--yaw-commands']+[str(w) for v,w,h in bank]+
                         ['--height-commands']+[str(h) for v,w,h in bank])
                    with (job/f'{tag}_seed{seed}.log').open('w') as log:
                        child=subprocess.Popen(cmd,env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                        state.update(operation=tag,seed=seed,child_pid=child.pid);save()
                        try:
                            code=child.wait()
                            if code:raise RuntimeError(f'{tag} exited {code}')
                        finally:
                            if child.poll() is None:child.terminate();child.wait()
                    d=json.loads(out.read_text())
                    for r in d['results']:
                        r['passed']=gate(r)['passed'] and r['metrics']['height_mae']<=.005 and r['nonwheel_contact_full_fraction']==0
                    records+=d['results']
                    if not all(r['passed'] for r in d['results']):break
                state['results'].append({'checkpoint':str(folder/f'model_{it}.pt'),
                    'seeds_evaluated':seed,'passed':len(records)==135 and all(r['passed'] for r in records),
                    'passed_count':sum(r['passed'] for r in records),'total':len(records),
                    'height_means':{str(h):sum(r['metrics']['height'] for r in records if r['command']==[0,0,h]) /
                                    sum(r['command']==[0,0,h] for r in records) for h in [.39,.4,.41]}})
                state['child_pid']=None;save()
        state['status']='complete';save()
    except BaseException as exc:
        state.update(status='error',error=str(exc),child_pid=None);save();raise


if __name__=='__main__':main()
