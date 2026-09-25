"""Matched source/candidate 60-second randomized horizontal impulse audits."""

import json
import os
from pathlib import Path
import subprocess
import sys


def main(root):
    root = Path(root)
    job = root/'plane/outputs/stand_push_comparison_v1'
    job.mkdir(exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', PYTHONPATH=str(root/'plane'),
               LD_LIBRARY_PATH=str(Path(sys.executable).parent.parent/'lib')+':'+os.environ.get('LD_LIBRARY_PATH',''))
    models = {
        'source': 'Sep17_17-41-11_stand_rand1_entropyfix_probe_sep17/model_200.pt',
        'candidate': 'Sep17_22-23-26_stand_ablation_20260917_221553_stand_symmetric_probe/model_400.pt',
    }
    status = {'status':'running', 'completed':[]}
    try:
        for model, checkpoint in models.items():
            for seed in (19,37,53):
                tag = f'{model}_{seed}'
                out = job/(tag+'.json')
                status['stage'] = tag
                (job/'status.json').write_text(json.dumps(status,indent=2))
                if not out.exists():
                    cmd = [sys.executable, str(root/'plane/wheel_legged_gym/scripts/evaluate_standing.py'),
                           '--profile=STAND_SYMMETRIC', '--checkpoint='+str(root/'plane/logs/wheel_legged'/checkpoint),
                           '--seconds=60','--warmup=5','--num-envs=32','--randomization-level=1',
                           '--push-delta-v=0.10','--seed='+str(seed),'--out='+str(out)]
                    with (job/(tag+'.log')).open('w') as log:
                        subprocess.run(cmd,env=env,cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
                data = json.loads(out.read_text())
                times = [t for p in data['push_results'] for t in p['recovery_seconds']]
                status['completed'].append({'model':model,'seed':seed,'failures':data['failure_count'],
                    'survival':data['survival_fraction'], 'recovered':sum(t is not None for t in times),
                    'push_trials':len(times), 'max_recovery_s':max((t for t in times if t is not None),default=None),
                    'nonwheel_contact':data['metrics']['nonwheel_contact_fraction']['mean'],
                    'geometry_mm':{k:v['mean_distance_m']*1000 for k,v in data['geometry_root_frame'].items()}})
        status['status']='finished_pending_review'
    except Exception as exc:
        status.update(status='error',error=str(exc))
        raise
    finally:
        (job/'status.json').write_text(json.dumps(status,indent=2))
