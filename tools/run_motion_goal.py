"""Unattended, bounded train/evaluate/select loop. Never promotes on reward alone."""
import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PLANE = ROOT/'plane'
sys.path.insert(0, str(PLANE))
import isaacgym
from wheel_legged_gym.envs.wheel_legged.motion_goal import STAGES, stage_bank
from compare_policy_versions import gate
from summarize_transitions import summarize

SOURCE = PLANE/'logs/wheel_legged/Sep20_00-08-57_legacy_yaw_20260920_000851/model_1900.pt'
ACCEPTED = PLANE/'logs/wheel_legged/Sep19_23-08-55_legacy_speed2_stop_20260919_230848/model_1400.pt'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assess(records):
    """Safety is a hard eligibility condition; tracking chooses same-stage candidates."""
    rows = [r for d in records for r in d['results']]
    safe = all(r['failure_count']==0 and r['timeout_count']==0 and
        r['nonwheel_contact_full_fraction']==0 and
        min(r['metrics']['left_contact'],r['metrics']['right_contact'])>=.99 and
        r['metrics']['height_mae']<=.03 and r['metrics']['torque_saturation']<=.01 for r in rows)
    ratios=[]
    for r in rows:
        m=r['metrics']
        ratios += [m['vx_mae']/(.05 if r['command'][0]==0 else .10), m['yaw_mae']/.10]
    score=max(ratios)+sum(max(0.,v-1) for v in ratios)/len(ratios)
    return {'safe':safe,'passed':safe and all(gate(r)['passed'] for r in rows),
            'passed_count':sum(gate(r)['passed'] for r in rows),'total':len(rows),'score':score,
            'failed_commands':sorted({tuple(r['command'][:2]) for r in rows if not gate(r)['passed']})}


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--max-rounds',type=int,default=40)
    parser.add_argument('--max-stagnant',type=int,default=3)
    parser.add_argument('--turn-curriculum',action='store_true',help='Resume accepted6900, expand combined turns gradually at fixed .40m')
    opts=parser.parse_args()
    stages=('turn1','turn2','turn3','turn4') if opts.turn_curriculum else STAGES
    initial_source=(PLANE/'logs/wheel_legged/Sep20_03-42-29_motion_goal_20260920_002043_r24_yaw4/model_6900.pt'
                    if opts.turn_curriculum else SOURCE)
    initial_accepted=initial_source if opts.turn_curriculum else ACCEPTED
    if opts.max_rounds<1 or opts.max_stagnant<1:parser.error('positive limits required')
    lock=(PLANE/'outputs/h3_low_speed_training.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    job=PLANE/'outputs'/('motion_goal_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    job.mkdir()
    state={'status':'starting','supervisor_pid':os.getpid(),'job':str(job),'stage':stages[0],
           'source':str(initial_source),'accepted':str(initial_accepted),'round':0,'history':[],
           'goal_complete':False,'limits':vars(opts),'child_pid':None}
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',PYTHONPATH=str(PLANE),
             LD_LIBRARY_PATH=str(Path(sys.executable).parent.parent/'lib')+':'+os.environ.get('LD_LIBRARY_PATH',''))
    def save():
        tmp=job/'status.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(job/'status.json')
        lines=['# Autonomous motion goal progress','',f"Status: {state['status']}",
               f"Stage: {state['stage']}; round: {state['round']}",
               f"Accepted checkpoint: `{state['accepted']}`",'',
               'Goal remains incomplete until final motion and independent sim2sim review.','',
               'Each round: 500 iterations, full-state resume, fixed recipe, 3-seed gates.','',
               '## Decisions','']
        lines += [json.dumps(h,ensure_ascii=False) for h in state['history']]
        report=job/'progress.md';report.write_text('\n'.join(lines)+'\n')
    def run(args,tag,extra=None):
        if (job/'STOP').exists():raise InterruptedError('User STOP file')
        cmd=[sys.executable]+[str(a) for a in args]
        state.update(operation=tag,command=cmd)
        with (job/(tag+'.log')).open('w') as log:
            child=subprocess.Popen(cmd,cwd=ROOT,env=dict(env,**(extra or {})),stdout=log,stderr=subprocess.STDOUT)
            state['child_pid']=child.pid;save()
            try:
                while child.poll() is None:
                    if (job/'STOP').exists():raise InterruptedError('User STOP file')
                    time.sleep(2)
                if child.returncode:raise RuntimeError(f'{tag} exited {child.returncode}')
            finally:
                if child.poll() is None:child.terminate();child.wait()
                state['child_pid']=None;save()
    def evaluate(checkpoint,stage,seeds,tag,transition=False):
        commands=sorted(set(stage_bank(stage)))
        initials=None
        if transition:
            pairs=[((0,0),(.5,0)),((.5,0),(0,0)),((.5,0),(-.5,0)),((-.5,0),(.5,0)),
                   ((0,-.5),(0,.5)),((0,.5),(0,-.5)),((2,0),(-2,0)),((-2,0),(2,0)),
                   ((0,0),(4,0)),((4,0),(0,0)),((4,0),(-4,0)),((-4,0),(4,0)),
                   ((0,-4),(0,4)),((0,4),(0,-4)),((1,-1),(1,1)),((-1,-1),(-1,1))]
            initials,commands=zip(*pairs)
        records=[]
        for seed in seeds:
            output=job/f'{tag}_seed{seed}.json'
            args=[PLANE/'wheel_legged_gym/scripts/evaluate_policy_comparison.py',
                  '--checkpoint='+str(checkpoint),'--out='+str(output),'--seed='+str(seed),
                  '--commands']+[str(v) for v,w in commands]+['--yaw-commands']+[str(w) for v,w in commands]
            if transition:
                args+=['--initial-commands']+[str(v) for v,w in initials]+['--initial-yaw-commands']+[str(w) for v,w in initials]+['--warmup=10','--switch-at=5']
            if not output.exists():run(args,f'{tag}_seed{seed}')
            data=json.loads(output.read_text())
            if data['checkpoint_sha256']!=digest(checkpoint):raise RuntimeError('Evaluation source changed')
            records.append(data)
        result=assess(records)
        if transition:
            summaries=[summarize(d) for d in records]
            (job/(tag+'_response.json')).write_text(json.dumps(summaries,indent=2)+'\n')
            # Conservative operational response gate; this is not a jerk/smoothness certification.
            response_ok=all(axis['settled_envs']==axis['total_envs'] and
                            axis['worst_settling_s'] is not None and axis['worst_settling_s']<=3.
                            for d in summaries for row in d['rows'] for axis in row['response'].values())
            result.update(response_passed=response_ok,passed=result['passed'] and response_ok)
        (job/(tag+'_summary.json')).write_text(json.dumps(result,indent=2)+'\n')
        return result
    def export(checkpoint,tag):
        onnx=job/(tag+'.onnx')
        run([PLANE/'export_onnx/export_onnx.py','--log_root='+str(PLANE/'logs/wheel_legged'),
             '--load_run='+checkpoint.parent.name,'--checkpoint='+checkpoint.stem.split('_')[-1],
             '--out='+str(onnx)],tag+'_export')
        run([PLANE/'export_onnx/verify_onnx.py','--checkpoint='+str(checkpoint),'--onnx='+str(onnx),'--batch=256'],tag+'_onnx_check')
    print(job,flush=True);save()
    source=initial_source;stage_index=0;stagnant=0;focus=[]
    try:
        baseline=evaluate(source,stages[0],[19,37,53],'initial')
        if not baseline['safe']:raise RuntimeError('Initial source fails safety gate')
        for round_id in range(1,opts.max_rounds+1):
            stage=stages[stage_index]
            state.update(status='training',stage=stage,round=round_id,source=str(source));save()
            iteration=int(source.stem.split('_')[-1])
            spec={'stage':stage,'focus':focus,'source_checkpoint':str(source),
                  'source_sha256':digest(source),'source_iteration':iteration}
            specpath=job/f'round{round_id:02d}_spec.json';specpath.write_text(json.dumps(spec,indent=2)+'\n')
            extra={'FUDAN_MOTION_GOAL_SPEC':str(specpath)}
            name=f'{job.name}_r{round_id:02d}_{stage}'
            args=[PLANE/'wheel_legged_gym/scripts/train.py','--task=wheel_legged','--headless',
                  '--resume','--resume_mode=full','--load_run='+source.parent.name,
                  '--checkpoint='+str(iteration),'--policy_experiment=MOTION_GOAL','--seed=23']
            run(args+['--num_envs=80','--max_iterations=1','--run_name='+name+'_smoke'],f'r{round_id:02d}_smoke',extra)
            run(args+['--num_envs=4096','--max_iterations=500','--run_name='+name],f'r{round_id:02d}_train',extra)
            folders=list((PLANE/'logs/wheel_legged').glob('*_'+name))
            if len(folders)!=1:raise RuntimeError('Ambiguous run directory')
            folder=folders[0];state.update(status='evaluating',run_dir=str(folder));save()
            run([ROOT/'tools/summarize_training.py',folder,'--window=100'],f'r{round_id:02d}_training_summary')
            screened=[]
            for it in range(iteration+100,iteration+501,100):
                checkpoint=folder/f'model_{it}.pt'
                score=evaluate(checkpoint,stage,[19],f'r{round_id:02d}_m{it}')
                if score['safe']:screened.append((score['score'],checkpoint))
            candidates=[]
            for _,checkpoint in sorted(screened)[:2]:
                score=evaluate(checkpoint,stage,[19,37,53],f'r{round_id:02d}_m{checkpoint.stem.split("_")[-1]}')
                if score['safe']:candidates.append((score['score'],checkpoint,score))
            if not candidates:
                state['history'].append({'round':round_id,'stage':stage,'decision':'no safe candidate; retained source'})
                stagnant+=1
            else:
                _,candidate,result=min(candidates)
                if result['passed'] and stage=='switches':
                    transition=evaluate(candidate,stage,[19,37,53],f'r{round_id:02d}_transition',transition=True)
                    result['passed']=transition['passed']
                    result['transition']=transition
                improved=result['score']<baseline['score']-.01
                if result['passed']:
                    export(candidate,f'accepted_{stage}')
                    state['accepted']=str(candidate)
                    state['history'].append({'round':round_id,'stage':stage,'decision':'stage accepted',
                                             'checkpoint':str(candidate),'result':result})
                    source=candidate;stage_index+=1;focus=[];stagnant=0
                    if stage_index==len(stages):
                        state['status']=('turn_grid_passed_pending_transitions_and_sim2sim' if opts.turn_curriculum
                            else 'simulation_curriculum_passed_pending_sim2sim_and_smoothness_review');save();return
                    baseline=evaluate(source,stages[stage_index],[19,37,53],f'enter_{stages[stage_index]}')
                else:
                    state['history'].append({'round':round_id,'stage':stage,'decision':'improved candidate' if improved else 'rollback retained source',
                                             'checkpoint':str(candidate),'result':result})
                    if improved:source=candidate;baseline=result;stagnant=0
                    else:stagnant+=1
                    # Only sampling weights change. Every stage anchor remains present.
                    focus=[list(p) for p in baseline['failed_commands']] * min(4,stagnant+2)
            state.update(source=str(source),stagnant_rounds=stagnant);save()
            if stagnant>=opts.max_stagnant:
                state['status']='paused_no_improvement_needs_diagnosis';save();return
        state['status']='paused_round_budget_review';save()
    except BaseException as exc:
        state.update(status='stopped' if isinstance(exc,InterruptedError) else 'error',error=str(exc));save();raise


if __name__=='__main__':main()
