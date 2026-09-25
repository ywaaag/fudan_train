"""Bounded independent closed-chain timing sweep; report tested points, never a global limit."""
import sys
from pathlib import Path
import argparse,concurrent.futures,hashlib,json,os,subprocess,sys,time
from datetime import datetime
from wheel_legged_gym.evaluation.sequences import summarize

def sequence(initial,target,duration):
    # Prepare moving initial state using the already validated slow protocol.
    if initial:
        points=[(0,0),(2,0),(12,initial),(14,initial)]
        start=14.
    else:
        points=[(0,0),(2,0)];start=2.
    points += [(start+duration,target)]
    return {'interpolation':'linear','keyframes':[{'time':t,'command':[v,0,.4]} for t,v in points]},start

def main(root):
    ROOT = Path(root)
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--policy',type=Path,required=True)
    p.add_argument('--durations',nargs='+',type=float,default=[4.,2.,1.,.5])
    p.add_argument('--workers',type=int,choices=[1,2,3,4],default=3)
    p.add_argument('--cases',nargs='+',choices=['start_pos','start_neg','stop_pos','stop_neg','reverse_pos','reverse_neg'])
    a=p.parse_args()
    if any(d<=0 or d>10 for d in a.durations) or len(set(a.durations))!=len(a.durations):p.error('Unique durations in (0,10] required')
    policy=a.policy.resolve();sha=hashlib.sha256(policy.read_bytes()).hexdigest()
    job=ROOT/'plane/outputs'/('dynamic_boundary_'+datetime.now().strftime('%Y%m%d_%H%M%S'));job.mkdir()
    state={'status':'running','supervisor_pid':os.getpid(),'policy':str(policy),'policy_sha256':sha,
           'durations':a.durations,'results':[],'scope':'Independent cold-start trials, fixed dt and original guards; a failed faster point does not prove a monotonic or global boundary.'}
    def save():
        tmp=job/'status.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(job/'status.json')
    print(job,flush=True);save()

    def trial(name,initial,target,duration):
        tag=name+'_ramp'+str(duration).replace('.','p');data,start=sequence(initial,target,duration)
        seq=job/(tag+'_sequence.json');seq.write_text(json.dumps(data,indent=2)+'\n')
        end=start+duration;warmup=end+2;steps=round((warmup+20)*1000)
        out=job/(tag+'.json')
        cmd=[sys.executable,str(ROOT/'tools/probe_closed_initialization.py'),'--policy',str(policy),
             '--out',str(out),'--initialization','tree_zero','--sequence-json',str(seq),
             '--steps',str(steps),'--metrics-warmup-seconds',str(warmup),'--trace']
        if (job/'STOP').exists():return {'tag':tag,'stopped':True}
        with (job/(tag+'.log')).open('w') as log:
            child=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            (job/(tag+'_process.json')).write_text(json.dumps({'pid':child.pid,'command':cmd})+'\n')
            try:
                while child.poll() is None:
                    if (job/'STOP').exists():child.terminate();child.wait();return {'tag':tag,'stopped':True}
                    time.sleep(1)
                if child.returncode:raise RuntimeError(tag+' exited '+str(child.returncode))
            finally:
                if child.poll() is None:child.terminate();child.wait()
        raw=json.loads(out.read_text())
        if raw['policy_sha256']!=sha:raise RuntimeError('Policy hash changed')
        report=summarize(raw)
        # Retain only the requested transition in timing comparisons; preparation must still pass physics.
        target_rows=[r for r in report['plateaus'] if abs(r['ramp_start']-start)<1e-6]
        valid=report['physical_passed'] and len(target_rows)==1 and target_rows[0]['tracking_passed']
        report.update(tag=tag,initial=initial,target=target,duration=duration,transition_passed=valid,
                      target_metrics=target_rows[0] if target_rows else None,
                      source_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),policy_sha256=sha,
                      failure=raw['result']['failure'],completed_steps=raw['result']['completed_steps'])
        (job/(tag+'_summary.json')).write_text(json.dumps(report,indent=2)+'\n')
        return {k:report[k] for k in ['tag','initial','target','duration','transition_passed','target_metrics','failure','completed_steps']}

    cases=[('start_pos',0,4),('start_neg',0,-4),('stop_pos',4,0),('stop_neg',-4,0),('reverse_pos',4,-4),('reverse_neg',-4,4)]
    if a.cases:cases=[c for c in cases if c[0] in a.cases]
    state['cases']=[c[0] for c in cases];save()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
            futures=[pool.submit(trial,name,i,t,d) for d in a.durations for name,i,t in cases]
            for future in concurrent.futures.as_completed(futures):
                state['results'].append(future.result());save()
        state['status']='stopped' if (job/'STOP').exists() else 'completed_pending_review';save()
    except BaseException as exc:
        state.update(status='error',error=str(exc));save();raise
