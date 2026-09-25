"""Motion evaluation workflow: explicit commands, process and artifact dependencies."""
from dataclasses import dataclass
from wheel_legged_gym.ports.processes import PythonJob
from wheel_legged_gym.ports.artifacts import EvaluationArtifacts
from wheel_legged_gym.evaluation.motion_candidates import assess, geometry_retained
from wheel_legged_gym.evaluation.transitions import summarize


@dataclass(frozen=True)
class EvaluationOptions:
    start_stop_ramp_seconds: float
    start_stop_speed: float
    geometry_symmetry: bool
    recover_motion: bool
    high_speed_geometry_floor: float


def evaluate(checkpoint, stage_commands, seeds, tag, transition=False, *,
             options, recovery_geometry, evaluation_script,
             files: EvaluationArtifacts, run: PythonJob, digest):
    """Run/reuse evaluations and summarize them through explicit I/O ports."""
    opts = options
    job = files.directory
    commands=sorted(set(stage_commands))
    initials=None
    if transition:
        pairs=[((0,0),(.5,0)),((.5,0),(0,0)),((.5,0),(-.5,0)),((-.5,0),(.5,0)),
               ((0,-.5),(0,.5)),((0,.5),(0,-.5)),((2,0),(-2,0)),((-2,0),(2,0)),
               ((0,0),(4,0)),((4,0),(0,0)),((4,0),(-4,0)),((-4,0),(4,0)),
               ((0,-4),(0,4)),((0,4),(0,-4)),((1,-1),(1,1)),((-1,-1),(-1,1))]
        initials,commands=zip(*pairs)
        if opts.start_stop_ramp_seconds:
            v=opts.start_stop_speed
            initials=[(0.,0.),(v,0.),(0.,0.),(-v,0.),(.8*v,0.),(-.3*v,0.),(.6*v,0.)]
            commands=[(v,0.),(0.,0.),(-v,0.),(0.,0.),(-.3*v,0.),(.6*v,0.),(0.,0.)]
    records=[]
    for seed in seeds:
        output=job/f'{tag}_seed{seed}.json'
        args=[evaluation_script,
              '--checkpoint='+str(checkpoint),'--out='+str(output),'--seed='+str(seed),
              '--commands']+[str(v) for v,w in commands]+['--yaw-commands']+[str(w) for v,w in commands]
        if transition:
            args+=['--initial-commands']+[str(v) for v,w in initials]+['--initial-yaw-commands']+[str(w) for v,w in initials]+['--warmup=10','--switch-at=5']
            if opts.start_stop_ramp_seconds:
                args+=['--trace-stride=1','--switch-at='+str({19:4.37,37:6.13,53:3.71}.get(seed,5.29))]
        if not files.exists(output):run(args,f'{tag}_seed{seed}')
        data=files.read_json(output)
        if data['checkpoint_sha256']!=digest(checkpoint):raise RuntimeError('Evaluation source changed')
        records.append(data)
    result=assess(records)
    if opts.geometry_symmetry or opts.recover_motion:
        motion_score=result['score']
        rows=[r for d in records for r in d['results'] if abs(r['command'][1])<.01]
        geometry=max(max(r['metrics']['knee_mirror_m'],r['metrics']['wheel_mirror_m']) for r in rows)
        by_command={}
        for r in rows:
            key=str(r['command'][0])
            by_command[key]=max(by_command.get(key,0),r['metrics']['knee_mirror_m'],r['metrics']['wheel_mirror_m'])
        result.update(motion_passed=result['passed'],geometry_passed=geometry<=.03,
                      max_mean_mirror_distance_m=geometry,geometry_threshold_m=.03,
                      geometry_by_command=by_command,motion_score=motion_score)
        result['passed']=result['passed'] and result['geometry_passed']
        result['score']=max(result['score'],geometry/.03)
        if opts.recover_motion:
            result['posture_retained']=not recovery_geometry or geometry_retained(
                by_command,recovery_geometry,opts.high_speed_geometry_floor)
            result['score']=motion_score
    if transition:
        summaries=[summarize(d) for d in records]
        files.write_json(tag+'_response.json', summaries)
        # Conservative operational response gate; this is not a jerk/smoothness certification.
        response_ok=all(axis['settled_envs']==axis['total_envs'] and
                        axis['worst_settling_s'] is not None and
                        (bool(opts.start_stop_ramp_seconds) or axis['worst_settling_s']<=3.)
                        for d in summaries for row in d['rows'] for axis in row['response'].values())
        result.update(response_passed=response_ok,passed=result['passed'] and response_ok)
        if opts.start_stop_ramp_seconds:
            dynamic_ok=all(row['transient']['wheel_contact_fraction']==1. and
                row['response']['vx']['settled_envs']==row['response']['vx']['total_envs'] and
                row['response']['vx']['worst_settling_s'] is not None and
                row['response']['vx']['worst_settling_s']>=0.
                for d in summaries for row in d['rows'])
            result.update(start_stop_passed=dynamic_ok,passed=result['passed'] and dynamic_ok)
    files.write_json(tag+'_summary.json', result)
    return result
