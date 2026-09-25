"""Height expansion gated by low-speed height tracking and full 40cm motion retention."""
import sys
from pathlib import Path
import os
import json
import hashlib
import subprocess
import fcntl
import argparse
import time
from datetime import datetime
from wheel_legged_gym.experiments.recipes.height_course import STAGES,height_bank
from wheel_legged_gym.evaluation.height_acceptance import assess_height_row

def main(root):
    ROOT = Path(root)
    PLANE = ROOT / 'plane'
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--variable-repeats',type=int,choices=[1,4],default=1)
    parser.add_argument('--stage-limit',type=int,choices=[1,2,3],default=3)
    parser.add_argument('--height-reward-gain',type=int,choices=[1,4],default=1)
    parser.add_argument('--stage',choices=['fixed35','fixed45','dual','micro']+list(STAGES))
    parser.add_argument('--source',type=Path)
    parser.add_argument('--job',type=Path)
    parser.add_argument('--seed',type=int,default=23)
    parser.add_argument('--height-switch-interval',type=float,choices=[0.,5.],default=0.)
    opts=parser.parse_args()
    lock=(PLANE/'outputs/h3_low_speed_training.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    job=opts.job.resolve() if opts.job else PLANE/'outputs'/('height_course_'+datetime.now().strftime('%Y%m%d_%H%M%S'));job.mkdir()
    source=opts.source.resolve() if opts.source else PLANE/'logs/wheel_legged/Sep20_03-42-29_motion_goal_20260920_002043_r24_yaw4/model_6900.pt'
    state={'status':'starting','supervisor_pid':os.getpid(),'child_pid':None,'source':str(source),'history':[]}
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',PYTHONPATH=str(PLANE),
             LD_LIBRARY_PATH=str(Path(sys.executable).parent.parent/'lib')+':'+os.environ.get('LD_LIBRARY_PATH',''))
    def save():
        tmp=job/'status.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(job/'status.json')
    def run(args,tag):
        if (job/'STOP').exists():raise InterruptedError('STOP requested')
        state.update(operation=tag,command=[sys.executable]+[str(a) for a in args]);save()
        with (job/(tag+'.log')).open('w') as log:
            child=subprocess.Popen(state['command'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            state['child_pid']=child.pid;save()
            try:
                while child.poll() is None:
                    if (job/'STOP').exists():raise InterruptedError('STOP requested')
                    time.sleep(2)
                result=child.returncode
                if result:raise RuntimeError(f'{tag} exited {result}')
            finally:
                if child.poll() is None:child.terminate();child.wait()
                state['child_pid']=None;save()
    print(job,flush=True);save()
    try:
        for stage in ([opts.stage] if opts.stage else STAGES[:opts.stage_limit]):
            iteration=int(source.stem.split('_')[-1])
            spec={'stage':stage,'variable_repeats':opts.variable_repeats,
                  'height_switch_interval':opts.height_switch_interval,
                  'height_reward_gain':opts.height_reward_gain,'source_checkpoint':str(source),
                  'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'source_iteration':iteration}
            specpath=job/(stage+'_spec.json');specpath.write_text(json.dumps(spec,indent=2)+'\n')
            env['FUDAN_HEIGHT_SPEC']=str(specpath)
            state.update(status='training',stage=stage,source=str(source));save()
            if opts.height_switch_interval:
                run([ROOT/'tools/audit_height_rollout.py','--checkpoint='+str(source),
                     '--out='+str(job/(stage+'_schedule_audit.json'))],stage+'_schedule_audit')
            name=(job.parent.name+'_' if opts.job else '')+job.name+'_'+stage
            args=[PLANE/'wheel_legged_gym/scripts/train.py','--task=wheel_legged','--headless',
                  '--resume','--resume_mode=full','--load_run='+source.parent.name,'--checkpoint='+str(iteration),
                  '--policy_experiment=HEIGHT_COURSE','--seed='+str(opts.seed)]
            run(args+['--num_envs=80','--max_iterations=1','--run_name='+name+'_smoke'],stage+'_smoke')
            run(args+['--num_envs=4096','--max_iterations=500','--run_name='+name],stage+'_train')
            folders=list((PLANE/'logs/wheel_legged').glob('*_'+name))
            if len(folders)!=1:raise RuntimeError('Ambiguous training run')
            folder=folders[0];state.update(status='evaluating',run_dir=str(folder));save()
            run([ROOT/'tools/summarize_training.py',folder,'--window=100'],stage+'_training_summary')
            candidate=folder/f'model_{iteration+500}.pt'
            # Every training command; height endpoints cannot silently disappear.
            bank=sorted(set(height_bank(stage,opts.variable_repeats)))
            records=[]
            for seed in [19,37,53]:
                tag=f'{stage}_seed{seed}';out=job/(tag+'.json')
                run([PLANE/'wheel_legged_gym/scripts/evaluate_policy_comparison.py',
                     '--checkpoint='+str(candidate),'--out='+str(out),'--seed='+str(seed),
                     '--commands']+[str(v) for v,w,h in bank]+['--yaw-commands']+[str(w) for v,w,h in bank]+
                     ['--height-commands']+[str(h) for v,w,h in bank],tag)
                data=json.loads(out.read_text())
                for row in data['results']:
                    assess_height_row(row,stage)
                    records.append(dict(row,seed=seed))
            passed=all(r['gate']['passed'] for r in records)
            result={'checkpoint':str(candidate),'stage':stage,'passed':passed,
                    'passed_count':sum(r['gate']['passed'] for r in records),'total':len(records),'records':records}
            (job/(stage+'_acceptance.json')).write_text(json.dumps(result,indent=2)+'\n')
            state['history'].append({k:v for k,v in result.items() if k!='records'})
            if not passed:
                state.update(status='paused_gate_failed',candidate=str(candidate));save();return
            onnx=job/(stage+'.onnx')
            run([PLANE/'export_onnx/export_onnx.py','--log_root='+str(PLANE/'logs/wheel_legged'),
                 '--load_run='+folder.name,'--checkpoint='+str(iteration+500),'--out='+str(onnx)],stage+'_export')
            run([PLANE/'export_onnx/verify_onnx.py','--checkpoint='+str(candidate),'--onnx='+str(onnx),'--batch=256'],stage+'_onnx_check')
            source=candidate;state['accepted']=str(source);save()
        state['status']='requested_stages_passed_pending_review';save()
    except BaseException as exc:
        state.update(status='error',error=str(exc));save();raise
