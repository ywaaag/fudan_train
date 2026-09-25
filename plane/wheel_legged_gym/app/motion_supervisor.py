"""Unattended, bounded train/evaluate/select loop. Never promotes on reward alone."""

from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys

from wheel_legged_gym.experiments.recipes.motion_goal import STAGES, stage_bank
from wheel_legged_gym.adapters.processes.python_job import PythonJobRunner
from wheel_legged_gym.adapters.artifacts.job_files import JobFiles
from wheel_legged_gym.app.motion_options import build_parser
from wheel_legged_gym.app.motion_resume import restore_evaluation_options
from wheel_legged_gym.app.motion_status import save_motion_status
from wheel_legged_gym.app.completion import write_report, wake_session
from wheel_legged_gym.workflows.motion_evaluation import EvaluationOptions, evaluate as evaluate_motion
from wheel_legged_gym.workflows.policy_export import export_verified_policy
from wheel_legged_gym.workflows.candidate_screening import screen_candidates
from wheel_legged_gym.workflows.motion_round import RoundMode, RoundProgress, advance_round
from wheel_legged_gym.workflows.motion_training_plan import (
    TrainingOptions, experiment_spec, experiment_signature, training_arguments,
)



def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(root):
    ROOT = Path(root)
    PLANE = ROOT / "plane"
    SOURCE = PLANE/'logs/wheel_legged/Sep20_00-08-57_legacy_yaw_20260920_000851/model_1900.pt'
    ACCEPTED = PLANE/'logs/wheel_legged/Sep19_23-08-55_legacy_speed2_stop_20260919_230848/model_1400.pt'

    parser = build_parser(__doc__)
    opts=parser.parse_args()
    resumed=None
    if opts.resume_evaluation_job:
        resume_job,resumed=restore_evaluation_options(opts,parser)
    if opts.recover_from and not opts.recover_motion:parser.error('--recover-from requires recovery mode')
    stages=('turn1','turn2','turn3','turn4') if opts.turn_curriculum else STAGES
    initial_source=(PLANE/'logs/wheel_legged/Sep20_03-42-29_motion_goal_20260920_002043_r24_yaw4/model_6900.pt'
                    if opts.turn_curriculum else SOURCE)
    initial_accepted=initial_source if opts.turn_curriculum else ACCEPTED
    if opts.geometry_symmetry:
        stages=('turn1',)
        initial_source=PLANE/'logs/wheel_legged/Sep20_14-09-30_motion_goal_20260920_135435_r02_turn1/model_7400.pt'
        initial_accepted=initial_source
        opts.max_rounds=1
    if opts.recover_motion:
        if opts.geometry_symmetry or opts.turn_curriculum:parser.error('Recovery is a separate fixed-stage mode')
        stages=('turn1',)
        initial_source=PLANE/'logs/wheel_legged/Sep20_15-24-28_motion_goal_20260920_152146_r01_turn1/model_7900.pt'
        if opts.recover_from:
            initial_source=Path(opts.recover_from).resolve()
            parent=json.loads((initial_source.parent/'policy_experiment.json').read_text())
            if not parent.get('motion_goal_spec',{}).get('geometry_symmetry'):
                parser.error('Recovery source must already use geometry symmetry')
        initial_accepted=None
    if opts.speed_envelope:
        if not opts.recover_motion:parser.error('Envelope requires --recover-motion to retain geometry')
        stages=('envelope1','envelope2','envelope3','envelope4')
    if opts.basic_motion:
        if not opts.recover_motion or opts.speed_envelope:parser.error('Basic motion requires recovery, excludes envelope')
        stages=('basic_motion',)
    if opts.max_rounds<1 or opts.max_stagnant<1:parser.error('positive limits required')
    if opts.command_switch_seconds and not opts.basic_motion:parser.error('Switching requires basic motion mode')
    if opts.freeze_motion_encoder and not opts.basic_motion:parser.error('Encoder ablation requires basic motion mode')
    if opts.geometry_weight != -.1 and not (opts.basic_motion and opts.freeze_motion_encoder):
        parser.error('Geometry weight ablation requires frozen-encoder basic motion')
    if opts.start_stop_ramp_seconds and not (opts.basic_motion and opts.freeze_motion_encoder and not opts.command_switch_seconds):
        parser.error('Start-stop requires frozen basic motion and excludes global switches')
    if opts.dynamic_fixed_lr and not opts.start_stop_ramp_seconds:parser.error('Fixed LR ablation requires dynamic curriculum')
    if opts.dynamic_equivariance and not opts.start_stop_ramp_seconds:parser.error('Equivariance ablation requires dynamic curriculum')
    if opts.dynamic_reference_coef and (not opts.start_stop_ramp_seconds or not opts.start_stop_fraction):
        parser.error('Policy reference requires a nonempty dynamic cohort')
    lock=(PLANE/'outputs/h3_low_speed_training.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if resumed and (resume_job/'STOP').exists():
        stamp=datetime.now().strftime('%Y%m%d_%H%M%S')
        (resume_job/'STOP').rename(resume_job/('STOP_acknowledged_'+stamp))
        (resume_job/('status_before_resume_'+stamp+'.json')).write_text(json.dumps(resumed,indent=2)+'\n')
    job=resume_job if resumed else PLANE/'outputs'/('motion_goal_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    if not resumed:job.mkdir()
    state={'status':'starting','supervisor_pid':os.getpid(),'job':str(job),'stage':stages[0],
           'source':str(initial_source),'accepted':str(initial_accepted) if initial_accepted else None,'round':0,'history':[],
           'goal_complete':False,'limits':vars(opts),'child_pid':None}
    state['limits']['resume_evaluation_job']=str(opts.resume_evaluation_job) if opts.resume_evaluation_job else None
    if resumed:
        state=resumed
        state.update(status='evaluating',supervisor_pid=os.getpid(),child_pid=None,resumed_from_operation=resumed.get('operation'))
        state.pop('error',None)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',PYTHONPATH=str(PLANE),
             LD_LIBRARY_PATH=str(Path(sys.executable).parent.parent/'lib')+':'+os.environ.get('LD_LIBRARY_PATH',''))
    recovery_geometry=dict(resumed.get('initial_geometry_reference',{})) if resumed else {}
    files = JobFiles(job)

    def save():
        save_motion_status(
            files, state, environment=os.environ,
            report=write_report, wake=wake_session,
        )

    def child_preparing(tag, command):
        state.update(operation=tag, command=command)

    def child_started(pid):
        state['child_pid'] = pid
        save()

    def child_finished():
        state['child_pid'] = None
        save()

    process_runner = PythonJobRunner(
        executable=sys.executable, working_directory=ROOT,
        job_directory=job, environment=env,
        on_preparing=child_preparing, on_started=child_started, on_finished=child_finished,
    )
    run = process_runner.run

    evaluation_options = EvaluationOptions(
        start_stop_ramp_seconds=opts.start_stop_ramp_seconds,
        start_stop_speed=opts.start_stop_speed,
        geometry_symmetry=opts.geometry_symmetry,
        recover_motion=opts.recover_motion,
        high_speed_geometry_floor=opts.high_speed_geometry_floor,
    )
    round_mode = RoundMode(
        start_stop_enabled=bool(opts.start_stop_ramp_seconds),
        recover_motion=opts.recover_motion,
        turn_curriculum=opts.turn_curriculum,
        speed_envelope=opts.speed_envelope,
        basic_motion=opts.basic_motion,
    )
    training_options = TrainingOptions(
        training_seed=opts.training_seed,
        geometry_symmetry=opts.geometry_symmetry,
        recover_motion=opts.recover_motion,
        command_switch_seconds=opts.command_switch_seconds,
        freeze_motion_encoder=opts.freeze_motion_encoder,
        geometry_weight=opts.geometry_weight,
        start_stop_ramp_seconds=opts.start_stop_ramp_seconds,
        start_stop_speed=opts.start_stop_speed,
        start_stop_fraction=opts.start_stop_fraction,
        dynamic_fixed_lr=opts.dynamic_fixed_lr,
        dynamic_equivariance=opts.dynamic_equivariance,
        dynamic_reference_coef=opts.dynamic_reference_coef,
    )

    def evaluate(checkpoint, stage, seeds, tag, transition=False):
        return evaluate_motion(
            checkpoint, stage_bank(stage), seeds, tag, transition,
            options=evaluation_options, recovery_geometry=recovery_geometry,
            evaluation_script=PLANE/'wheel_legged_gym/scripts/evaluate_policy_comparison.py',
            files=files, run=run, digest=digest,
        )

    def export(checkpoint,tag):
        export_verified_policy(checkpoint, tag, plane=PLANE, job=job, run=run)
    print(job,flush=True);save()
    source=initial_source;stage_index=0;stagnant=0;focus=[];attempted=set()
    try:
        start_round=1
        if resumed:
            source=Path(resumed['source']);stage_index=stages.index(resumed['stage'])
            start_round=resumed['round'];stagnant=resumed.get('stagnant_rounds',0)
            saved_spec=json.loads((job/f'round{start_round:02d}_spec.json').read_text());focus=saved_spec.get('focus',[])
            if digest(source)!=saved_spec['source_sha256']:raise RuntimeError('Resume source changed')
            baseline=next((h['result'] for h in reversed(state['history']) if h.get('checkpoint')==str(source)
                           and h['decision'] in ['improved candidate','stage accepted']),None)
            if baseline is None:baseline=evaluate(source,stages[stage_index],[19,37,53],'resume_source')
        else:
            baseline=evaluate(source,stages[0],[19,37,53],'initial')
        if not baseline['safe']:raise RuntimeError('Initial source fails safety gate')
        if opts.start_stop_ramp_seconds:
            initial_dynamic=evaluate(source,stages[0],[19,37,53],'initial_start_stop',transition=True)
            state['initial_start_stop']=initial_dynamic;save()
            if baseline['passed'] and initial_dynamic['passed']:
                export(source,'accepted_basic_motion')
                state.update(status='dynamic_source_passed_pending_closed_validation',accepted=str(source))
                save();return
        if opts.recover_motion and not resumed:
            recovery_geometry.update(baseline['geometry_by_command'])
            state['initial_geometry_reference']=recovery_geometry;save()
        for round_id in range(start_round,opts.max_rounds+1):
            stage=stages[stage_index]
            state.update(status='training',stage=stage,round=round_id,source=str(source));save()
            iteration=int(source.stem.split('_')[-1])
            spec=experiment_spec(source,digest(source),stage,focus,training_options)
            signature=experiment_signature(spec,training_options.training_seed)
            if signature in attempted:
                state.update(status='paused_duplicate_experiment_prevented',
                    reason='Same source SHA, curriculum, reward mode and seed already attempted in this job')
                save();return
            attempted.add(signature)
            state['training_seed']=opts.training_seed
            specpath=job/f'round{round_id:02d}_spec.json';specpath.write_text(json.dumps(spec,indent=2)+'\n')
            extra={'FUDAN_MOTION_GOAL_SPEC':str(specpath)}
            name=f'{job.name}_r{round_id:02d}_{stage}'
            args=training_arguments(PLANE,source,training_options.training_seed)
            if not (resumed and round_id==start_round):
                run(args+['--num_envs=80','--max_iterations=1','--run_name='+name+'_smoke'],f'r{round_id:02d}_smoke',extra)
                if opts.dynamic_reference_coef:
                    from wheel_legged_gym.adapters.artifacts.reference_smoke import verify
                    smoke_dirs=list((PLANE/'logs/wheel_legged').glob('*_'+name+'_smoke'))
                    if len(smoke_dirs)!=1:raise RuntimeError('Ambiguous reference smoke directory')
                    verification=verify(source,smoke_dirs[0],opts.start_stop_fraction)
                    (job/f'r{round_id:02d}_startup_verification.json').write_text(json.dumps(verification,indent=2)+'\n')
                run(args+['--num_envs=4096','--max_iterations=500','--run_name='+name],f'r{round_id:02d}_train',extra)
            folders=list((PLANE/'logs/wheel_legged').glob('*_'+name))
            if len(folders)!=1:raise RuntimeError('Ambiguous run directory')
            folder=folders[0];state.update(status='evaluating',run_dir=str(folder));save()
            if not (folder/f'model_{iteration+500}.pt').exists():raise RuntimeError('Completed training checkpoint missing')
            run([ROOT/'tools/summarize_training.py',folder,'--window=100'],f'r{round_id:02d}_training_summary')
            candidates=screen_candidates(folder, iteration, stage, round_id, evaluate=evaluate)
            progress=advance_round(
                candidates, RoundProgress(source,stage_index,baseline,stagnant,focus),
                stages=stages, round_id=round_id, mode=round_mode,
                state=state, evaluate=evaluate, export=export,
            )
            source=progress.source;stage_index=progress.stage_index
            baseline=progress.baseline;stagnant=progress.stagnant;focus=progress.focus
            if progress.curriculum_finished:
                save();return
            state.update(source=str(source),stagnant_rounds=stagnant);save()
            if stagnant>=opts.max_stagnant:
                state['status']='paused_no_improvement_needs_diagnosis';save();return
        state['status']='paused_round_budget_review';save()
    except BaseException as exc:
        state.update(status='stopped' if isinstance(exc,InterruptedError) else 'error',error=str(exc));save();raise
