"""Audited ramp-and-hold validation, never changes closed-chain safety gates."""
import sys
from pathlib import Path
import json,os,sys,subprocess,time,fcntl,argparse,hashlib,threading
from wheel_legged_gym.workflows.validation_schedule import run_grid
from datetime import datetime



def main(root):
    ROOT = Path(root)
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--policy',type=Path,default=ROOT/'plane/outputs/motion_goal_20260921_143444/accepted_basic_motion.onnx')
    parser.add_argument('--workers',type=int,choices=[1,2,3,4],default=1)
    args=parser.parse_args();policy=args.policy.resolve()
    policy_sha=hashlib.sha256(policy.read_bytes()).hexdigest()
    lock=(ROOT/'plane/outputs/closed_ramp.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    job=ROOT/'plane/outputs'/('closed_ramp_v2_'+datetime.now().strftime('%Y%m%d_%H%M%S'));job.mkdir()
    state={'status':'evaluating','supervisor_pid':os.getpid(),'child_pid':None,'results':[],
           'policy':str(policy),'policy_sha256':policy_sha,'workers':args.workers,'child_pids':{},
           'protocol':'2s zero hold; 10s ramp; 2s settling; 20s target measurement', 'skipped':[]}
    state_lock=threading.RLock();cancelled=threading.Event()
    def save():
        with state_lock:
            tmp=job/'status.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(job/'status.json')
    def child_state(tag,pid):
        with state_lock:
            if pid is None:state['child_pids'].pop(tag,None)
            else:state['child_pids'][tag]=pid
            state['child_pid']=next(iter(state['child_pids'].values())) if len(state['child_pids'])==1 else None
            state['operation']=list(state['child_pids']);save()
    def record(result):
        with state_lock:state['results'].append(result);save()
    def skip(tag):
        with state_lock:state['skipped'].append(tag);save()
    print(job,flush=True);save()
    def evaluate(case):
            tag,v,w=case
            if (job/'STOP').exists() or cancelled.is_set():raise InterruptedError('Validation cancelled')
            out=job/(tag+'.json')
            command=[sys.executable,str(ROOT/'tools/probe_closed_initialization.py'),
                  '--policy',str(policy),
                  '--out',str(out),'--initialization','tree_zero','--steps','34000',
                  '--settle-seconds','2','--ramp-seconds','10','--metrics-warmup-seconds','14',
                  '--forward',str(v),'--yaw',str(w)]
            with (job/(tag+'.log')).open('w') as log:
                child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                child_state(tag,child.pid)
                try:
                    while child.poll() is None:
                        if (job/'STOP').exists() or cancelled.is_set():raise InterruptedError('Validation cancelled')
                        time.sleep(2)
                    if child.returncode:raise RuntimeError(f'{tag} failed with exit {child.returncode}')
                finally:
                    if child.poll() is None:child.terminate();child.wait()
                    child_state(tag,None)
            d=json.loads(out.read_text());r=d['result'];m=d['measurements']
            if d['policy_sha256']!=policy_sha:raise RuntimeError('Policy changed during validation')
            physical=r['passed'] and r['completed_steps']==34000
            if physical and m['samples']!=20000:raise RuntimeError('Unexpected steady sample count')
            tracking=physical and m['vx_mae']<=(.1 if v else .05) and m['yaw_mae']<=.1 and m['height_mae']<=.03
            return {'tag':tag,'command':[v,w,.4],'physical':physical,'tracking':tracking,
                'samples':m['samples'],'vx_mae':m['vx_mae'],'yaw_mae':m['yaw_mae'],
                'vx_mean':r['mean_forward_speed_m_s'] if physical else None,
                'yaw_mean':r['mean_yaw_rate_rad_s'] if physical else None,
                'max_closure':r['max_closure_residual_m'],'failure':r['failure']}
    try:
        run_grid(evaluate,record,skip,workers=args.workers,cancel=cancelled.set)
        state['status']='completed_pending_review';save()
    except BaseException as exc:
        state.update(status='error',error=str(exc));save();raise
