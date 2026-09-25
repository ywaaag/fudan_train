"""Independent seeds: steady retention, starts/stops, spin and low-speed reversals."""

import sys
from pathlib import Path
_cli_package_root = str(Path(__file__).resolve().parents[1] / "plane")
if _cli_package_root not in sys.path:
    sys.path.insert(0, _cli_package_root)
import json,os,sys,subprocess,time,hashlib,fcntl
from pathlib import Path
from datetime import datetime
from wheel_legged_gym.evaluation.transitions import summarize
from wheel_legged_gym.evaluation.gates import gate
ROOT=Path(__file__).resolve().parents[1]
CP=ROOT/'plane/logs/wheel_legged/Sep21_14-37-19_motion_goal_20260921_143444_r01_basic_motion/model_9500.pt'


def main():
    lock=(ROOT/'plane/outputs/h3_low_speed_training.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    job=ROOT/'plane/outputs'/('basic_dynamics_'+datetime.now().strftime('%Y%m%d_%H%M%S'));job.mkdir()
    state={'status':'evaluating','supervisor_pid':os.getpid(),'child_pid':None,'completed':[],
           'checkpoint':str(CP),'sha256':hashlib.sha256(CP.read_bytes()).hexdigest()}
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',PYTHONPATH=str(ROOT/'plane'),LD_LIBRARY_PATH=str(Path(sys.executable).parent.parent/'lib'))
    def save():
        tmp=job/'status.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(job/'status.json')
    print(job,flush=True);save()
    groups={'linear':[((0,0),(v,0)) for v in [-1,1,-2,2,-4,4]]+
                     [((v,0),(0,0)) for v in [-1,1,-2,2,-4,4]],
            'spin':[((0,0),(0,w)) for w in [-3,3,-4,4]]+[((0,w),(0,0)) for w in [-3,3,-4,4]],
            'reverse':[((v,0),(-v,0)) for v in [-.5,.5,-1,1]]+
                      [((0,w),(0,-w)) for w in [-1,1]]}
    try:
        for group in ['steady']+list(groups):
            for seed in [71,89,107]:
                tag=f'{group}_seed{seed}';out=job/(tag+'.json')
                if group=='steady':
                    targets=[(0,0)]+[(s*v,0) for v in [.5,1,1.5,2,2.5,3,3.5,4] for s in [-1,1]]+[(0,s*w) for w in [.5,1,2,4] for s in [-1,1]]
                else:initials,targets=zip(*groups[group])
                cmd=[sys.executable,str(ROOT/'plane/wheel_legged_gym/scripts/evaluate_policy_comparison.py'),
                     '--checkpoint='+str(CP),'--out='+str(out),'--seed='+str(seed),'--seconds=60',
                     '--commands']+[str(v) for v,w in targets]+['--yaw-commands']+[str(w) for v,w in targets]
                if group!='steady':
                    cmd+=['--seconds=20','--warmup=10','--switch-at=5','--trace-stride=1',
                          '--initial-commands']+[str(v) for v,w in initials]+['--initial-yaw-commands']+[str(w) for v,w in initials]
                with (job/(tag+'.log')).open('w') as log:
                    child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                    state.update(operation=tag,child_pid=child.pid);save()
                    try:
                        while child.poll() is None:
                            if (job/'STOP').exists():raise InterruptedError('User STOP')
                            time.sleep(2)
                        if child.returncode:raise RuntimeError(f'{tag} exited {child.returncode}')
                    finally:
                        if child.poll() is None:child.terminate();child.wait()
                        state['child_pid']=None;save()
                d=json.loads(out.read_text())
                if d['checkpoint_sha256']!=state['sha256']:raise RuntimeError('Checkpoint changed')
                report={'steady_checks':[dict(command=r['command'],gate=gate(r),
                    geometry_passed=(r['command'][1]!=0 or max(r['metrics']['knee_mirror_m'],r['metrics']['wheel_mirror_m'])<=.03),
                    full_contact_passed=r['nonwheel_contact_full_fraction']==0) for r in d['results']]}
                if group!='steady':report['response']=summarize(d)
                (job/(tag+'_summary.json')).write_text(json.dumps(report,indent=2)+'\n')
                state['completed'].append(tag);save()
        state['status']='completed_pending_dynamic_review';save()
    except BaseException as exc:
        state.update(status='error',error=str(exc));save();raise


if __name__=='__main__':main()
