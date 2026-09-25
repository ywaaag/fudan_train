"""Batch explicit closed-chain sequences and notify once; no model promotion."""

import sys
from pathlib import Path
_cli_package_root = str(Path(__file__).resolve().parents[1] / "plane")
if _cli_package_root not in sys.path:
    sys.path.insert(0, _cli_package_root)
import argparse,concurrent.futures,hashlib,json,os,subprocess,sys,time
from pathlib import Path
from datetime import datetime
from wheel_legged_gym.domain.commands.sequence import CommandSequence
from wheel_legged_gym.evaluation.sequences import summarize
from wheel_legged_gym.app.completion import write_report,wake_session
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--policy',type=Path,required=True)
    p.add_argument('--sequences',type=Path,nargs='+',required=True)
    p.add_argument('--workers',type=int,choices=[1,2,3],default=3)
    a=p.parse_args();policy=a.policy.resolve();sha=hashlib.sha256(policy.read_bytes()).hexdigest()
    inputs=[]
    for path in a.sequences:
        data=json.loads(path.read_text());sequence=CommandSequence(data)
        inputs.append((path.resolve(),data,sequence.times[-1]))
    if len({p.stem for p,d,t in inputs})!=len(inputs):raise ValueError('Sequence names must be unique')
    job=ROOT/'plane/outputs'/('closed_sequences_'+datetime.now().strftime('%Y%m%d_%H%M%S'));job.mkdir()
    state={'status':'evaluating','supervisor_pid':os.getpid(),'policy':str(policy),'policy_sha256':sha,'results':[]}
    def save():
        tmp=job/'status.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(job/'status.json')
    print(job,flush=True);save()
    def trial(entry):
        source,data,end=entry;tag=source.stem;seq=job/(tag+'_input.json');out=job/(tag+'.json')
        seq.write_text(json.dumps(data,indent=2)+'\n');warmup=end+2
        cmd=[sys.executable,str(ROOT/'tools/probe_closed_initialization.py'),'--policy',str(policy),
             '--out',str(out),'--initialization','tree_zero','--sequence-json',str(seq),
             '--steps',str(round((warmup+20)*1000)),'--metrics-warmup-seconds',str(warmup),'--trace']
        if (job/'STOP').exists():return {'tag':tag,'stopped':True}
        with (job/(tag+'.log')).open('w') as log:
            child=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            (job/(tag+'_process.json')).write_text(json.dumps({'pid':child.pid,'command':cmd})+'\n')
            try:
                while child.poll() is None:
                    if (job/'STOP').exists():child.terminate();child.wait();return {'tag':tag,'stopped':True}
                    time.sleep(2)
                if child.returncode:raise RuntimeError(tag+' exited '+str(child.returncode))
            finally:
                if child.poll() is None:child.terminate();child.wait()
        raw=json.loads(out.read_text())
        if raw['policy_sha256']!=sha:raise RuntimeError('Policy changed during verification')
        report=summarize(raw);report.update(tag=tag,policy_sha256=sha,
            input_source=str(source),raw_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
            completed_steps=raw['result']['completed_steps'],failure=raw['result']['failure'])
        (job/(tag+'_summary.json')).write_text(json.dumps(report,indent=2)+'\n');return report
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
            for future in concurrent.futures.as_completed([pool.submit(trial,e) for e in inputs]):
                state['results'].append(future.result());save()
        state['status']='stopped' if (job/'STOP').exists() else 'finished_pending_review'
    except BaseException as exc:state.update(status='error',error=str(exc))
    finally:
        save();write_report(job,state)
        lines=['# 闭链连续序列验收','',f"状态：{state['status']}",f"策略：{policy}",'',
               '| 序列 | 通过 | 完成步数 | 失败 |','|---|---|---:|---|']
        for r in state['results']:lines.append(f"| {r['tag']} | {r.get('passed')} | {r.get('completed_steps')} | {r.get('failure')} |")
        lines+=['','只验证记录输入序列；不自动提升模型或完成goal。']
        if state.get('error'):lines+=['',state['error']]
        (job/'completion_report.md').write_text('\n'.join(lines)+'\n')
        if os.environ.get('HAPI_SESSION_ID'):wake_session(job,os.environ['HAPI_SESSION_ID'])


if __name__=='__main__':main()
