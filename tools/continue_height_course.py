"""Bounded continuous height curriculum; reviewable per-round sources and gates."""
import json
import os
import sys
import fcntl
import subprocess
import time
import argparse
from pathlib import Path
from datetime import datetime

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'plane/outputs'


def assess(records,stage):
    safe=all(r['failure_count']==0 and r['timeout_count']==0 and
             r['nonwheel_contact_full_fraction']==0 and
             min(r['metrics']['left_contact'],r['metrics']['right_contact'])>=.99 and
             r['metrics']['height_mae']<=.03 and r['metrics']['torque_saturation']<=.01 for r in records)
    ratios=[]
    for r in records:
        m=r['metrics'];v,w,h=r['command']
        ratios.extend([m['vx_mae']/(.05 if v==0 else .1),m['yaw_mae']/.1,
                       m['height_mae']/(.005 if stage=='micro' else .015)])
    return safe,max(ratios)+sum(max(0,x-1) for x in ratios)/len(ratios)


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--screen-job',type=Path)
    parser.add_argument('--height-switch-interval',type=float,choices=[0.,5.],default=0.)
    opts=parser.parse_args()
    lock=(OUTPUT/'height_continuation.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    job=OUTPUT/('height_continuation_'+datetime.now().strftime('%Y%m%d_%H%M%S'));job.mkdir()
    source=ROOT/'plane/logs/wheel_legged/Sep20_03-42-29_motion_goal_20260920_002043_r24_yaw4/model_6900.pt'
    stages=['micro','near','middle','full'];stage_index=0;stagnant=0;best=float('inf')
    state={'status':'starting','supervisor_pid':os.getpid(),'child_pid':None,'history':[],
           'source':str(source),'accepted_motion':str(source),'goal_complete':False,
           'max_rounds':12,'max_stagnant':3}
    def save():
        tmp=job/'status.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(job/'status.json')
        (job/'progress.md').write_text('# 连续高度课程\n\n'+json.dumps(state,indent=2,ensure_ascii=False)+'\n')
    print(job,flush=True);save()
    try:
        if opts.screen_job:
            state['status']='waiting_for_checkpoint_screen';save()
            while True:
                if (job/'STOP').exists():raise InterruptedError('User STOP')
                screening=json.loads((opts.screen_job/'status.json').read_text())
                if screening['status']=='error':raise RuntimeError('Checkpoint screening failed')
                if screening['status']=='complete':break
                # A dead screener must not leave an infinite silent wait.
                os.kill(screening['pid'],0)
                time.sleep(2)
            if any(r['passed'] for r in screening['results']):
                state['status']='screen_found_passing_checkpoint_review_before_new_training';save();return
        if opts.height_switch_interval:
            stages=['micro'];state['max_rounds']=3;save()
        for n in range(1,state['max_rounds']+1):
            stage=stages[stage_index];childjob=job/f'round{n:02d}_{stage}'
            state.update(status='running',round=n,stage=stage,source=str(source),child_job=str(childjob));save()
            # First try repeats the reviewed recipe; retries vary only training RNG.
            seed=[23,37,53][stagnant%3]
            cmd=[sys.executable,str(ROOT/'tools/run_height_course.py'),'--stage='+stage,
                 '--source='+str(source),'--job='+str(childjob),'--variable-repeats=4',
                 '--height-reward-gain=1','--seed='+str(seed)]
            cmd+=['--height-switch-interval='+str(opts.height_switch_interval)]
            with (job/f'round{n:02d}.log').open('w') as log:
                child=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                state['child_pid']=child.pid;save()
                try:
                    while child.poll() is None:
                        if (job/'STOP').exists():
                            childjob.mkdir(exist_ok=True);(childjob/'STOP').touch()
                        time.sleep(2)
                    if (job/'STOP').exists():raise InterruptedError('User STOP')
                    if child.returncode:raise RuntimeError(f'Round {n} failed: inspect log')
                finally:
                    if child.poll() is None:
                        (childjob/'STOP').touch();child.wait()
                    state['child_pid']=None;save()
            result=json.loads((childjob/(stage+'_acceptance.json')).read_text())
            safe,score=assess(result['records'],stage)
            entry={'round':n,'stage':stage,'seed':seed,'checkpoint':result['checkpoint'],
                   'passed':result['passed'],'passed_count':result['passed_count'],
                   'total':result['total'],'safe':safe,'score':score}
            if result['passed'] and safe:
                source=Path(result['checkpoint']);state['accepted_height']=str(source)
                entry['decision']='stage accepted';stage_index+=1;stagnant=0;best=float('inf')
            elif safe and score<best-.01:
                # Only keep a partial candidate if all old 40cm motion checks also pass.
                retained=all(r['gate']['passed'] for r in result['records'] if r['command'][2]==.4)
                if retained:
                    source=Path(result['checkpoint']);best=score;stagnant=0
                    entry['decision']='same-stage candidate; retained all 40cm motion'
                else:
                    stagnant+=1;entry['decision']='rollback: old motion failed'
            else:
                stagnant+=1;entry['decision']='rollback: unsafe or no improvement'
            state['history'].append(entry);state.update(source=str(source),stagnant=stagnant);save()
            if stage_index==len(stages):
                state['status']='fixed_height_passed_pending_dynamic_height_and_motion_envelope';save();return
            if stagnant>=3:
                state['status']='paused_three_failed_rounds_needs_diagnosis';save();return
        state['status']='paused_round_limit';save()
    except BaseException as e:
        state.update(status='stopped' if isinstance(e,InterruptedError) else 'error',error=str(e));save();raise


if __name__=='__main__':main()
