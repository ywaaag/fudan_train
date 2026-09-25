"""Progress each pure-axis direction separately; retain physical and tracking failures."""
import json,os,sys,subprocess,time,fcntl,argparse,hashlib
from pathlib import Path
from datetime import datetime
ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--policy',type=Path,default=ROOT/'plane/outputs/motion_goal_20260921_143444/accepted_basic_motion.onnx')
    parser.add_argument('--target-only',action='store_true',help='Report direct +/-4 steps without lower-speed screening')
    args=parser.parse_args();policy=args.policy.resolve()
    policy_sha=hashlib.sha256(policy.read_bytes()).hexdigest()
    speeds=[4] if args.target_only else [1,2,4]
    lock=(ROOT/'plane/outputs/closed_speed_validation.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    job=ROOT/'plane/outputs'/('closed_speed_'+datetime.now().strftime('%Y%m%d_%H%M%S'));job.mkdir()
    state={'status':'evaluating','supervisor_pid':os.getpid(),'child_pid':None,'results':[],'skipped':[],
           'policy':str(policy),'policy_sha256':policy_sha,'protocol':'Direct command from reset, 20 seconds, 2 second metrics warmup','speeds':speeds}
    def save():
        tmp=job/'status.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(job/'status.json')
    print(job,flush=True);save()
    try:
        for axis,sign in [('vx',1),('vx',-1),('yaw',1),('yaw',-1)]:
            for speed in speeds:
                tag=f'{axis}_{"pos" if sign>0 else "neg"}{speed}';out=job/(tag+'.json')
                cmd=[sys.executable,str(ROOT/'tools/probe_closed_initialization.py'),
                     '--policy',str(policy),
                     '--out',str(out),'--initialization','tree_zero','--steps','20000',
                     '--forward' if axis=='vx' else '--yaw',str(sign*speed)]
                with (job/(tag+'.log')).open('w') as log:
                    child=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                    state.update(operation=tag,child_pid=child.pid);save()
                    try:
                        while child.poll() is None:
                            if (job/'STOP').exists():raise InterruptedError('User STOP')
                            time.sleep(2)
                        if child.returncode:raise RuntimeError(f'{tag} exited {child.returncode}')
                    finally:
                        if child.poll() is None:child.terminate();child.wait()
                        state['child_pid']=None;save()
                d=json.loads(out.read_text());r=d['result'];m=d['measurements']
                if d['policy_sha256']!=policy_sha:raise RuntimeError('Policy changed during validation')
                physical=r['passed'] and r['completed_steps']==20000
                tracking=physical and m['vx_mae']<=(.1 if axis=='vx' else .05) and m['yaw_mae']<=.1 and m['height_mae']<=.03
                state['results'].append({'tag':tag,'physical_passed':physical,'tracking_passed':tracking,
                    'completed_steps':r['completed_steps'],'failure':r['failure'],
                    'vx_mean':r.get('mean_forward_speed_m_s'),'yaw_mean':r.get('mean_yaw_rate_rad_s'),
                    'measurements':m,'closure_max':r['max_closure_residual_m'],'policy_sha256':d['policy_sha256']});save()
                if not physical:
                    state['skipped'] += [f'{axis}_{sign}_{v}' for v in speeds if v>speed]
                    save();break
        state['status']='completed_pending_review';save()
    except BaseException as exc:
        state.update(status='stopped' if isinstance(exc,InterruptedError) else 'error',error=str(exc));save();raise


if __name__=='__main__':main()
