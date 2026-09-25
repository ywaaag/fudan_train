"""Assemble candidate closed-chain checks, process supervision and completion notification."""
from wheel_legged_gym.evaluation.closed_precheck import yaw_precheck_passed
from wheel_legged_gym.workflows.closed_review_report import render_candidate_review
import argparse,hashlib,json,os,subprocess,sys,time,fcntl
from pathlib import Path
from datetime import datetime
from wheel_legged_gym.app.completion import write_report,wake_session


def main(root, argv=None):
    ROOT = Path(root)
    parser=argparse.ArgumentParser("Finish candidate steady and dynamic closed-chain checks, then notify once.")
    parser.add_argument('--policy',type=Path,default=ROOT/'plane/outputs/motion_goal_20260922_111611/diagnostic_10300.onnx')
    parser.add_argument('--yaw-precheck',action='store_true',help='Check the known +4 yaw regression first; stop on failure')
    args=parser.parse_args(argv)
    lock=(ROOT/'plane/outputs/candidate_closed_review.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    job=ROOT/'plane/outputs'/('candidate_closed_review_'+datetime.now().strftime('%Y%m%d_%H%M%S'));job.mkdir()
    policy=args.policy.resolve()
    state={'status':'evaluating','supervisor_pid':os.getpid(),'child_pid':None,'policy':str(policy),
           'policy_sha256':hashlib.sha256(policy.read_bytes()).hexdigest(),'reviews':[]}
    def save():
        tmp=job/'status.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(job/'status.json')
    print(job,flush=True);save()
    jobs=[('steady','validate_closed_ramp.py',['--workers','4']),
          ('dynamic','run_dynamic_boundary.py',['--durations','2','1','--workers','3'])]
    try:
        if args.yaw_precheck:
            out=job/'yaw_precheck.json'
            cmd=[sys.executable,str(ROOT/'tools/probe_closed_initialization.py'),'--policy',str(policy),
                 '--out',str(out),'--initialization','tree_zero','--steps','34000',
                 '--settle-seconds','2','--ramp-seconds','10','--metrics-warmup-seconds','14','--yaw','4','--trace']
            with (job/'yaw_precheck.log').open('w') as log:
                child=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                state.update(operation='yaw_precheck',child_pid=child.pid);save()
                try:
                    while child.poll() is None:
                        if (job/'STOP').exists():raise InterruptedError('User STOP')
                        time.sleep(2)
                    if child.returncode:raise RuntimeError('Yaw precheck process failed')
                finally:
                    if child.poll() is None:child.terminate();child.wait()
                    state['child_pid']=None;save()
            raw=json.loads(out.read_text());r=raw['result'];m=raw['measurements']
            passed=yaw_precheck_passed(raw,state['policy_sha256'])
            state['reviews'].append({'name':'yaw_precheck','job':str(job),'result':{'results':[
                {'tag':'yaw_+4','tracking':passed,'failure':r['failure'],'completed_steps':r['completed_steps']}]}})
            save()
            if not passed:
                state.update(status='paused_on_regression',reason='Known yaw regression precheck failed; full review not run')
                return
        for name,script,args in jobs:
            cmd=[sys.executable,str(ROOT/'tools'/script),'--policy',str(policy)]+args
            logfile=job/(name+'.log');childjob=None
            if (job/'STOP').exists():raise InterruptedError('User STOP')
            with logfile.open('w') as log:
                child=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                state.update(operation=name,child_pid=child.pid);save()
                try:
                    while child.poll() is None:
                        if childjob is None:
                            for line in logfile.read_text().splitlines():
                                if line.startswith(str(ROOT/'plane/outputs')+'/'):
                                    childjob=Path(line.strip());state['child_job']=str(childjob);save();break
                        if (job/'STOP').exists():
                            if childjob:(childjob/'STOP').touch()
                            else:child.terminate()
                        time.sleep(2)
                    if (job/'STOP').exists():raise InterruptedError('User STOP')
                    if child.returncode:raise RuntimeError(name+' process failed: '+str(child.returncode))
                finally:
                    if child.poll() is None:
                        if childjob:(childjob/'STOP').touch()
                        else:child.terminate()
                        child.wait()
                    state['child_pid']=None;save()
            if childjob is None:
                childjob=Path(next(x for x in logfile.read_text().splitlines() if x.startswith(str(ROOT/'plane/outputs')+'/')))
            result=json.loads((childjob/'status.json').read_text())
            state['reviews'].append({'name':name,'job':str(childjob),'result':result});save()
        state['status']='finished_pending_review'
    except BaseException as exc:
        state.update(status='stopped' if isinstance(exc,InterruptedError) else 'error',error=str(exc))
    finally:
        save();write_report(job,state)
        (job/'completion_report.md').write_text(render_candidate_review(policy,state))
        if os.environ.get('HAPI_SESSION_ID'):wake_session(job,os.environ['HAPI_SESSION_ID'])

