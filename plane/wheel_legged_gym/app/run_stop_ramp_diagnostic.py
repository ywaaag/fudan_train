"""Matched baseline/candidate reverse-stop diagnostic; no training or promotion."""
import sys
from pathlib import Path
import json,os,sys,subprocess,time,hashlib,fcntl
from datetime import datetime
from wheel_legged_gym.evaluation.transitions import summarize
from wheel_legged_gym.evaluation.gates import gate
from wheel_legged_gym.app.completion import write_report,wake_session



def main(root):
    ROOT = Path(root)
    lock=(ROOT/'plane/outputs/h3_low_speed_training.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    job=ROOT/'plane/outputs'/('reverse_stop_diagnostic_'+datetime.now().strftime('%Y%m%d_%H%M%S'));job.mkdir()
    state={'status':'evaluating','supervisor_pid':os.getpid(),'child_pid':None,'results':[],
           'scope':'Explicit step and .05/.1/.25-second ramps; same sources and seeds; no goal promotion'}
    sources={'baseline':ROOT/'plane/logs/wheel_legged/Sep22_08-49-16_motion_goal_20260922_084744_r01_basic_motion/model_10000.pt',
             'candidate':ROOT/'plane/logs/wheel_legged/Sep22_11-19-04_motion_goal_20260922_111611_r01_basic_motion/model_10300.pt'}
    def save():
        tmp=job/'status.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(job/'status.json')
    print(job,flush=True);save()
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',PYTHONPATH=str(ROOT/'plane'),LD_LIBRARY_PATH=str(Path(sys.executable).parent.parent/'lib'))
    try:
        for name,checkpoint in sources.items():
            sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            for ramp in [0.,.05,.1,.25]:
                for seed in [19,37,53]:
                    tag=f'{name}_ramp{ramp}_seed{seed}';out=job/(tag+'.json')
                    cmd=[sys.executable,str(ROOT/'plane/wheel_legged_gym/scripts/evaluate_policy_comparison.py'),
                        '--checkpoint',str(checkpoint),'--out',str(out),'--commands','0','--initial-commands','-2',
                        '--seed',str(seed),'--seconds','12','--warmup','7','--switch-at','4.37','--trace-stride','1',
                        '--transition-ramp-seconds',str(ramp)]
                    if (job/'STOP').exists():raise InterruptedError('User STOP')
                    with (job/(tag+'.log')).open('w') as log:
                        child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                        state.update(operation=tag,child_pid=child.pid);save()
                        try:
                            while child.poll() is None:
                                if (job/'STOP').exists():raise InterruptedError('User STOP')
                                time.sleep(2)
                            if child.returncode:raise RuntimeError(tag+' exited '+str(child.returncode))
                        finally:
                            if child.poll() is None:child.terminate();child.wait()
                            state['child_pid']=None;save()
                    data=json.loads(out.read_text());assert data['checkpoint_sha256']==sha
                    response=summarize(data)['rows'][0];row=data['results'][0]
                    geom=max(row['metrics']['knee_mirror_m'],row['metrics']['wheel_mirror_m'])
                    contact=response['transient']['wheel_contact_fraction']==1.
                    good=gate(row)['passed'] and geom<=.03 and contact and response['response']['vx']['settled_envs']==16
                    result={'name':name,'ramp':ramp,'seed':seed,'passed':good,'geometry_m':geom,
                        'steady_gate':gate(row),'response':response,'source_sha256':sha,
                        'minimum_post_switch_vertical_force_n':min(v for t in data['response_trace'] if t['time']>=data['switch_at'] for pair in t['wheel_vertical_force_n'] for v in pair)}
                    (job/(tag+'_summary.json')).write_text(json.dumps(result,indent=2)+'\n')
                    state['results'].append(result);save()
        state['status']='finished_pending_review'
    except BaseException as exc:
        state.update(status='stopped' if isinstance(exc,InterruptedError) else 'error',error=str(exc))
    finally:
        save();write_report(job,state)
        lines=['# 后退停车输入协议对照','', '| 策略 | 斜坡秒 | seed | 通过 | 接触比例 | 最差稳定秒 |','|---|---:|---:|---|---:|---:|']
        for r in state['results']:
            t=r['response'];lines.append(f"| {r['name']} | {r['ramp']} | {r['seed']} | {r['passed']} | {t['transient']['wheel_contact_fraction']:.7f} | {t['response']['vx']['worst_settling_s']} |")
        lines+=['','只证明记录输入协议。阶跃失败不被斜坡结果覆盖；未做闭链验收或模型提升。']
        (job/'completion_report.md').write_text('\n'.join(lines)+'\n')
        if os.environ.get('HAPI_SESSION_ID'):wake_session(job,os.environ['HAPI_SESSION_ID'])
