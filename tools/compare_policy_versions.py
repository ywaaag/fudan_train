"""Resume-safe matched historical/current policy evaluation, without training."""

import sys
from pathlib import Path
_cli_package_root = str(Path(__file__).resolve().parents[1] / "plane")
if _cli_package_root not in sys.path:
    sys.path.insert(0, _cli_package_root)
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
PLANE = ROOT / 'plane'
EVALUATOR = PLANE / 'wheel_legged_gym/scripts/evaluate_policy_comparison.py'
CANDIDATES = {
    'legacy_h3': ('Sep05_17-41-43_H3_from_H2_best_v1', 15800),
    'legacy_h7_fixed': ('Sep06_16-54-55_H7_fixed_3ms_stable_v1', 16200),
    'legacy_h7_bounded': ('Sep06_18-52-26_H7_bounded_reward_long_v1', 17700),
    'method_translate2': ('Sep07_16-10-41_method_v1_translate_20_long_v1', 5000),
    'method_translate3_symmetry': ('Sep15_22-48-01_method_v1_translate_30_symmetry_v1', 1500),
    'standing': ('Sep18_21-21-37_stand_validated_20260918_212129', 3100),
    'low_speed': ('Sep18_23-25-44_low_speed_05_20260918_232537', 3600),
    'low_speed_tracking': ('Sep18_23-49-14_low_speed_05_20260918_234907', 3600),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2)+'\n')
    tmp.replace(path)


from wheel_legged_gym.evaluation.gates import gate


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--job', type=Path, help='Resume this exact job without repeating completed grids')
    p.add_argument('--candidates', nargs='+', choices=list(CANDIDATES), default=list(CANDIDATES))
    p.add_argument('--seeds', nargs='+', type=int, default=[19, 37, 53])
    p.add_argument('--randomization-levels', nargs='+', type=int, choices=[0, 1], default=[0, 1])
    a = p.parse_args()
    job = a.job.resolve() if a.job else PLANE/'outputs'/('policy_comparison_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    job.mkdir(parents=True, exist_ok=bool(a.job))
    lock = (job/'lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest_path = job/'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest['evaluator_sha256'] != sha(EVALUATOR) or manifest['runner_sha256'] != sha(__file__):
            raise RuntimeError('Evaluation code changed; use a new job')
    else:
        candidates = {}
        for name in a.candidates:
            run, iteration = CANDIDATES[name]
            folder = PLANE/'logs/wheel_legged'/run
            checkpoint = folder/f'model_{iteration}.pt'
            candidates[name] = {'checkpoint': str(checkpoint), 'sha256': sha(checkpoint),
                'source_manifest': json.loads((folder/'policy_experiment.json').read_text()),
                'archived_source_sha256': {f.name:sha(f) for f in folder.glob('*.py')}}
        manifest = {'schema': 'policy_comparison_v1', 'candidates': candidates,
            'seeds': a.seeds, 'randomization_levels': a.randomization_levels,
            'seconds':25, 'warmup':5, 'envs_per_command':16,
            'initial_commands':[0, -.5, .5, -1, 1], 'higher_stages':[2, 3],
            'evaluator_sha256':sha(EVALUATOR), 'runner_sha256':sha(__file__),
            'asset_sha256':sha(ROOT/'assets/wheel_leg_train.urdf'),
            'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'git_status':subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),
            'selection': 'H3 15800 retained historical candidate; H7 fixed16200 and bounded17700 minimize 100-step pre-checkpoint score (1-survival)+zero_abs_vx+forward_relative_error+reverse_relative_error+yaw_relative_error within their respective runs. This is a screening heuristic, not proof of global best. Other branches use documented completed endpoints.',
            'gates': {'zero_vx_mae':.05,'moving_vx_mae':.10,'yaw_mae':.10,'height_mae':.03,
                      'wheel_contact_min':.99,'torque_saturation_max':.01,'failures':0,'timeouts':0,'nonwheel_contact':0},
            'scope':'Common current physics/termination, actor mean, no pushes/no observation noise. No training, compensation or critic inference.'}
        write(manifest_path, manifest)
    for c in manifest['candidates'].values():
        if sha(c['checkpoint']) != c['sha256']:
            raise RuntimeError('Checkpoint changed')
    if sha(ROOT/'assets/wheel_leg_train.urdf') != manifest['asset_sha256']:
        raise RuntimeError('Asset changed')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', PYTHONPATH=str(PLANE),
               LD_LIBRARY_PATH=str(Path(sys.executable).parent.parent/'lib')+':'+os.environ.get('LD_LIBRARY_PATH',''))
    state = {'status':'evaluating','supervisor_pid':os.getpid(),'completed':[],'job':str(job)}
    print(str(job), flush=True)
    summary = {}

    def evaluate(name, rand, seed, commands, stage):
        tag = f'{name}_rand{rand}_seed{seed}_{stage}'
        out = job/(tag+'.json')
        c = manifest['candidates'][name]
        if not out.exists():
            cmd = [sys.executable, str(EVALUATOR), '--checkpoint='+c['checkpoint'], '--out='+str(out),
                   '--randomization-level='+str(rand), '--seed='+str(seed), '--commands']+[str(v) for v in commands]
            start = time.monotonic()
            with (job/(tag+'.log')).open('a') as log:
                child = subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                state.update(stage=tag,child_pid=child.pid)
                write(job/'status.json',state)
                try:
                    code = child.wait()
                except BaseException:
                    child.terminate()
                    child.wait()
                    raise
                if code:
                    raise RuntimeError(f'{tag} exited {code}; inspect its log')
            state['last_grid_seconds'] = round(time.monotonic()-start,2)
        data = json.loads(out.read_text())
        if (data['checkpoint_sha256'] != c['sha256'] or data['evaluator_sha256'] != manifest['evaluator_sha256']
                or [r['command'][0] for r in data['results']] != commands
                or data['seed'] != seed or data['randomization_level'] != rand):
            raise RuntimeError('Cached grid does not match job: '+tag)
        state['completed'].append(tag)
        write(job/'status.json',state)
        return data

    try:
        # Nominal and randomized tests are separate regimes. Promotion never
        # borrows a passing seed or regime from a different one.
        for rand in manifest['randomization_levels']:
            for name in manifest['candidates']:
                records = [evaluate(name,rand,s,manifest['initial_commands'],'initial') for s in manifest['seeds']]
                key = f'{name}_rand{rand}'
                summary[key] = {'stages':[]}
                stage_sets = [('initial', records)]
                for stage, grids in stage_sets:
                    rows = [dict(seed=g['seed'], **r, gate=gate(r)) for g in grids for r in g['results']]
                    passed = all(r['gate']['passed'] for r in rows)
                    summary[key]['stages'].append({'stage':stage,'passed':passed,'rows':rows})
                    write(job/'summary.json',summary)
                    print(json.dumps({'candidate':name,'randomization':rand,'stage':stage,'passed':passed,
                                      'completed_grids':len(state['completed'])}),flush=True)
                    if passed:
                        next_speed = 2 if stage == 'initial' else int(stage)+1
                        if next_speed <= 3:
                            commands = [0., -float(next_speed), float(next_speed)]
                            nxt = [evaluate(name,rand,s,commands,str(next_speed)) for s in manifest['seeds']]
                            stage_sets.append((str(next_speed),nxt))
                    else:
                        summary[key]['higher_speed_skipped'] = 'Prerequisite command grid failed; no speed promotion.'
                write(job/'summary.json',summary)
        state.update(status='completed',child_pid=None)
    except BaseException as exc:
        state.update(status='error',error=str(exc),child_pid=None)
        raise
    finally:
        write(job/'status.json',state)


if __name__ == '__main__':
    main()
