"""One 500-iteration low-speed relay, followed by signed fixed-command audits."""
import json
import argparse
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime

root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(__doc__)
parser.add_argument('--profile',choices=['LOW_SPEED','LOW_SPEED_TRACKING'],default='LOW_SPEED')
args=parser.parse_args()
plane=root/'plane'
job=plane/'outputs'/('low_speed_05_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
job.mkdir()
env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',PYTHONPATH=str(plane),
 LD_LIBRARY_PATH=str(Path(sys.executable).parent.parent/'lib')+':'+os.environ.get('LD_LIBRARY_PATH',''))
state={'status':'training','additional_iterations':500,'completed_audits':[],
 'profile':args.profile,
 'source_checkpoint':'Sep18_21-21-37_stand_validated_20260918_212129/model_3100.pt','supervisor_pid':os.getpid()}
def save():
 t=job/'status.tmp';t.write_text(json.dumps(state,indent=2));t.replace(job/'status.json')
def run(cmd,tag):
 with (job/(tag+'.log')).open('w') as log:
  p=subprocess.Popen([sys.executable]+cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
  state.update(stage=tag,child_pid=p.pid);save()
  if p.wait():raise RuntimeError('Failed stage '+tag)
try:
 run([str(plane/'wheel_legged_gym/scripts/train.py'),'--task=wheel_legged','--headless','--num_envs=4096',
  '--resume','--resume_mode=full','--load_run=Sep18_21-21-37_stand_validated_20260918_212129',
  '--checkpoint=3100','--policy_experiment='+args.profile,'--max_iterations=500','--seed=23','--run_name='+job.name],'train')
 runs=list((plane/'logs/wheel_legged').glob('*_'+job.name))
 if len(runs)!=1:raise RuntimeError('Ambiguous run')
 model=runs[0]/'model_3600.pt'
 state.update(status='evaluating',checkpoint=str(model))
 run([str(root/'tools/summarize_training.py'),str(runs[0])],'summary')
 for seed in [19,37,53]:
  for vx in [-.5,0.,.5]:
   tag=f'audit_{seed}_{vx:+.1f}'
   out=job/(tag+'.json')
   run([str(plane/'wheel_legged_gym/scripts/evaluate_standing.py'),'--profile='+args.profile,
    '--checkpoint='+str(model),'--randomization-level=1','--num-envs=16','--seconds=25',
    '--warmup=5','--seed='+str(seed),'--vx='+str(vx),'--out='+str(out)],tag)
   state['completed_audits'].append(str(out));save()
 reports=[json.loads(Path(p).read_text()) for p in state['completed_audits']]
 state['gate_passed']=all(x['failure_count']==0 and x['timeout_count']==0
  and x['metrics']['nonwheel_contact_fraction']['mean']==0
  and min(x['metrics'][k]['mean'] for k in ['left_contact','right_contact'])>=.99
  and abs(x['metrics']['height_m']['mean']-.4)<.03
  and x['tracking_vx_mae']<=(.05 if x['command_vx']==0 else .10)
  and x['metrics']['yaw_rad_s']['mean_abs']<=.10 for x in reports)
 state['status']='finished_pending_review' if state['gate_passed'] else 'paused_on_regression'
except Exception as exc:
 state.update(status='error',error=str(exc));raise
finally:save()
