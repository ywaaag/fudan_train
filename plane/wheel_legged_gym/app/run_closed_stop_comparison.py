"""Matched full closed-chain reverse-stop ramps; diagnostic only, no automatic promotion."""
import sys
from pathlib import Path
import argparse,concurrent.futures,hashlib,json,os,subprocess,sys,time
from datetime import datetime
from wheel_legged_gym.evaluation.sequences import summarize
from wheel_legged_gym.app.completion import write_report,wake_session



def main(root):
    ROOT = Path(root)
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--durations',type=float,nargs='+',default=[.05,.25,1.])
    args=parser.parse_args()
    if any(d<=0 or d>10 for d in args.durations) or len(set(args.durations))!=len(args.durations):
        parser.error('Require unique ramp durations in (0,10]')
    job=ROOT/'plane/outputs'/('closed_stop_comparison_'+datetime.now().strftime('%Y%m%d_%H%M%S'));job.mkdir()
    sources={'baseline':ROOT/'plane/outputs/motion_goal_20260922_084744/accepted_basic_motion.onnx',
             'candidate':ROOT/'plane/outputs/motion_goal_20260922_111611/diagnostic_10300.onnx'}
    state={'status':'evaluating','supervisor_pid':os.getpid(),'results':[],'durations':args.durations}
    def save():
        tmp=job/'status.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(job/'status.json')
    print(job,flush=True);save()
    def trial(name,policy,duration):
        tag=f'{name}_ramp{duration}';seq=job/(tag+'_sequence.json');out=job/(tag+'.json')
        sequence={'interpolation':'linear','keyframes':[{'time':t,'command':[v,0,.4]} for t,v in
            [(0,0),(2,0),(3,-2),(7,-2),(7+duration,0)]]}
        seq.write_text(json.dumps(sequence,indent=2)+'\n')
        warmup=9+duration;steps=round((warmup+20)*1000)
        sha=hashlib.sha256(policy.read_bytes()).hexdigest()
        cmd=[sys.executable,str(ROOT/'tools/probe_closed_initialization.py'),'--policy',str(policy),
             '--out',str(out),'--initialization','tree_zero','--steps',str(steps),
             '--sequence-json',str(seq),'--metrics-warmup-seconds',str(warmup),'--trace']
        if (job/'STOP').exists():return {'tag':tag,'stopped':True}
        with (job/(tag+'.log')).open('w') as log:
            child=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            (job/(tag+'_process.json')).write_text(json.dumps({'pid':child.pid,'command':cmd})+'\n')
            try:
                while child.poll() is None:
                    if (job/'STOP').exists():child.terminate();child.wait();return {'tag':tag,'stopped':True}
                    time.sleep(2)
                if child.returncode:raise RuntimeError(tag+' failed process')
            finally:
                if child.poll() is None:child.terminate();child.wait()
        raw=json.loads(out.read_text());assert raw['policy_sha256']==sha
        report=summarize(raw);report.update(tag=tag,policy_sha256=sha,
            raw_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),failure=raw['result']['failure'],
            completed_steps=raw['result']['completed_steps'])
        (job/(tag+'_summary.json')).write_text(json.dumps(report,indent=2)+'\n')
        return report
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futures=[pool.submit(trial,n,p,d) for d in args.durations for n,p in sources.items()]
            for f in concurrent.futures.as_completed(futures):state['results'].append(f.result());save()
        state['status']='stopped' if (job/'STOP').exists() else 'finished_pending_review'
    except BaseException as exc:state.update(status='error',error=str(exc))
    finally:
        save();write_report(job,state)
        lines=['# 闭链后退停车对照','',f"状态：{state['status']}",'','| 案例 | 通过 | 完成步数 | 失败 |','|---|---|---:|---|']
        for r in state['results']:lines.append(f"| {r['tag']} | {r.get('passed')} | {r.get('completed_steps')} | {r.get('failure')} |")
        lines+=['','同一初始化/物理保护。候选尚未被提升；完整goal仍需高速稳态和动态验证。']
        (job/'completion_report.md').write_text('\n'.join(lines)+'\n')
        if os.environ.get('HAPI_SESSION_ID'):wake_session(job,os.environ['HAPI_SESSION_ID'])
