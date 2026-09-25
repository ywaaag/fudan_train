"""Supervise up to four 500-iteration segments with drift and posture gates."""

from wheel_legged_gym.evaluation.stand_continuation import assess_segment

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main(root):
    root = Path(root)
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--first-job',type=Path,required=True)
    p.add_argument('--baseline-job',type=Path,required=True)
    a=p.parse_args()
    job=a.first_job.resolve()
    baseline=a.baseline_job.resolve()
    out=job/'long_guard';out.mkdir(exist_ok=True)
    lock=(out/'lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',PYTHONPATH=str(root/'plane'),
     LD_LIBRARY_PATH=str(Path(sys.executable).parent.parent/'lib')+':'+os.environ.get('LD_LIBRARY_PATH',''))
    state={'status':'running','max_additional_iterations':2000,'segments':[],
     'baseline_job':str(baseline),'supervisor_pid':os.getpid()}
    def save():
     t=out/'status.tmp';t.write_text(json.dumps(state,indent=2));t.replace(out/'status.json')
    def stationary(folder,model):
     target=folder/'stationary_seed19.json'
     if not target.exists():
      with (folder/'stationary.log').open('w') as log:
       subprocess.run([sys.executable,str(root/'plane/wheel_legged_gym/scripts/evaluate_standing.py'),
        '--profile=STAND_SYMMETRIC','--checkpoint='+model,'--num-envs=32','--seed=19',
        '--randomization-level=1','--seconds=65','--warmup=5','--push-delta-v=0',
        '--out='+str(target)],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
     return json.loads(target.read_text())
    try:
     base=json.loads((baseline/'status.json').read_text())
     base_audits={json.loads(Path(f).read_text())['seed']:json.loads(Path(f).read_text()) for f in base['completed_audits']}
     # First segment is already running; do not restart it or contend with it.
     base_stationary=None
     for segment in range(1,5):
      state.update(current_job=str(job),stage='waiting_for_segment');save()
      while True:
       s=json.loads((job/'status.json').read_text())
       if s['status']=='finished_pending_review':break
       if s['status']=='error':raise RuntimeError(s.get('error','child error'))
       try:os.kill(s['supervisor_pid'],0)
       except ProcessLookupError:raise RuntimeError('Segment supervisor stopped; inspect child before recovery')
       time.sleep(30)
      state['stage']='checking_drift';save()
      if base_stationary is None:base_stationary=stationary(out,base['checkpoint'])
      current=stationary(job,s['checkpoint'])
      audits=[json.loads(Path(f).read_text()) for f in s['completed_audits']]
      reasons,drift=assess_segment(audits,base_audits,current,base_stationary)
      state['segments'].append({'job':str(job),'checkpoint':s['checkpoint'],
       'stationary_displacement_m':drift,'passed':not reasons,'reasons':reasons})
      save()
      if reasons:
       state['status']='paused_on_regression';break
      if segment==4:
       state['status']='finished_pending_review';break
      before=set((root/'plane/outputs').glob('stand_validated_*'))
      with (out/f'segment{segment+1}.log').open('w') as log:
       child=subprocess.Popen([sys.executable,str(root/'tools/continue_stand_validated.py'),
        '--source-job',str(job)],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
       while True:
        new=[f for f in set((root/'plane/outputs').glob('stand_validated_*'))-before
             if (f/'status.json').exists() and json.loads((f/'status.json').read_text()).get('supervisor_pid')==child.pid]
        if len(new)==1:job=new[0];break
        if child.poll() is not None:raise RuntimeError('Next segment did not start')
        time.sleep(1)
    except Exception as exc:
     state.update(status='error',error=str(exc));raise
    finally:save()
