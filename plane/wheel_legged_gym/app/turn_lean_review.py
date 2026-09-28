"""Run one bounded candidate through the existing Isaac evaluation CLI."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from wheel_legged_gym.experiments.recipes.motion_goal import stage_bank
from wheel_legged_gym.experiments.recipes.turn_lean_long import validate_spec


def target_heights(pairs, spec):
    if spec is None:
        return [.4] * len(pairs)
    stage = spec['stage']
    schedule = stage.get('height_schedule', ())
    heights = []
    for vx, yaw in pairs:
        height = stage['turn_height']
        for threshold, target in schedule:
            if abs(vx * yaw) >= threshold:
                height = target
        heights.append(height)
    return heights


def entry_exit_pairs(spec, train_pairs):
    if 'height_schedule' not in spec['stage']:
        return [(sv*2., sw*.5) for sv in (-1.,1.) for sw in (-1.,1.)]
    selected = []
    for target in dict.fromkeys(height for _, height in spec['stage']['height_schedule']):
        pair = next((pair for pair in train_pairs
                     if target_heights([pair], spec)[0] == target), None)
        if pair is None:
            raise ValueError('No training turn pair reaches scheduled height {}'.format(target))
        selected.extend((sv*pair[0], sw*pair[1])
                        for sv in (-1.,1.) for sw in (-1.,1.))
    return selected


def command(root, checkpoint, out, speeds, yaws, heights, seed, envs, *,
            spec=None, exit_at=None, reverse=False):
    options = [sys.executable, 'wheel_legged_gym/scripts/evaluate_policy_comparison.py',
               '--checkpoint', str(checkpoint), '--out', str(out),
               '--commands', *map(str,speeds), '--yaw-commands', *map(str,yaws),
               '--height-commands', *map(str,heights),
               '--profile', 'method_v1', '--randomization-level', '1',
               '--envs-per-command', str(envs), '--seed', str(seed),
               '--trace-stride', '10' if exit_at is not None or
               (spec is not None and spec.get('experiment_id') in (
                   'inward_cornering_r10200_v1', 'roll_constraint_ablation_v1'))
               else '50']
    if spec is None:
        return options + ['--seconds','25','--warmup','5']
    stage = spec['stage']
    if spec.get('experiment_id') in ('inward_cornering_r10200_v1',
                                     'roll_constraint_ablation_v1'):
        options += ['--geometry-trace']
    options += ['--initial-commands', *(['0']*len(speeds)),
                '--initial-height-commands', *(['0.4']*len(speeds)),
                '--switch-at','1', '--transition-ramp-seconds',
                str(stage['speed_ramp_seconds']), '--yaw-delay-seconds',
                str(stage['speed_ramp_seconds']), '--height-delay-seconds',
                str(stage['speed_ramp_seconds']), '--lean-reference-max-deg',
                str(stage['lean_max_deg']), '--warmup','8']
    if exit_at is None:
        return options + ['--seconds','18']
    options += ['--yaw-exit-at',str(exit_at), '--yaw-exit-ramp-seconds',
                '4' if reverse else '2']
    if reverse:
        options += ['--yaw-exit-factor','-1','--seconds','19']
    else:
        options += ['--height-return-at-exit','--seconds','18']
    return options


def reusable_result(path, argv, checkpoint_sha, evaluator_sha):
    """A failed physical case is reusable when its evaluation is complete."""
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise ValueError('Existing review output is unreadable: {}'.format(path)) from exc
    def values(name, until=None):
        start = argv.index(name) + 1
        end = argv.index(until, start) if until else next((i for i in range(start,len(argv))
            if argv[i].startswith('--')),len(argv))
        return [float(value) for value in argv[start:end]]
    commands = list(zip(values('--commands','--yaw-commands'),
                        values('--yaw-commands','--height-commands'),
                        values('--height-commands')))
    def number(name, default=None):
        return float(argv[argv.index(name)+1]) if name in argv else default
    expected = {'checkpoint':argv[argv.index('--checkpoint')+1],
        'checkpoint_sha256':checkpoint_sha,'evaluator_sha256':evaluator_sha,
        'seed':int(number('--seed')),'envs_per_command':int(number('--envs-per-command')),
        'randomization_level':int(number('--randomization-level')),
        'seconds':number('--seconds'),'warmup':number('--warmup'),
        'geometry_trace':'--geometry-trace' in argv,
        'switch_at':number('--switch-at'),
        'transition_ramp_seconds':number('--transition-ramp-seconds'),
        'yaw_delay_seconds':number('--yaw-delay-seconds'),
        'height_delay_seconds':number('--height-delay-seconds'),
        'lean_reference_max_deg':number('--lean-reference-max-deg'),
        'yaw_exit_at':number('--yaw-exit-at'),
        'yaw_exit_ramp_seconds':number('--yaw-exit-ramp-seconds'),
        'yaw_exit_factor':number('--yaw-exit-factor',
            0. if '--yaw-exit-at' in argv else None),
        'height_return_at_exit':'--height-return-at-exit' in argv
            if '--initial-commands' in argv else None}
    for key,value in expected.items():
        observed=data.get(key)
        if (abs(observed-value)>1e-5 if isinstance(value,float) and observed is not None
                else observed!=value):
            raise ValueError('Existing review identity differs at {}: {}'.format(key,path))
    schema = ('matched_policy_transition_v1' if '--switch-at' in argv
              else 'matched_policy_grid_v1')
    if (data.get('schema')!=schema or
            len(data.get('results',[]))!=len(commands) or
            [row.get('command') for row in data['results']]!=[list(cmd) for cmd in commands] or
            any(not row.get('metrics') or not row.get('per_env_metrics')
                for row in data['results']) or
            (expected['geometry_trace'] and not data.get('response_trace')) or
            ('--yaw-exit-at' in argv and not data.get('response_trace'))):
        raise ValueError('Existing review is incomplete: {}'.format(path))
    if '--initial-commands' in argv:
        initial = [[value, 0., .4] for value in values('--initial-commands',
            '--initial-height-commands')]
        if data.get('initial_commands') != initial:
            raise ValueError('Existing review initial commands differ: {}'.format(path))
    return data


def main(root):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--job',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--spec',type=Path,required=True)
    parser.add_argument('--review-name',required=True)
    parser.add_argument('--model-label',required=True)
    parser.add_argument('--seeds',type=int,nargs='+',default=[19])
    parser.add_argument('--envs-per-command',type=int,default=8)
    parser.add_argument('--groups',nargs='+',choices=(
        'retention','turn_train','turn_holdout','entry_exit','slow_reverse',
        'height_skill','height_entry_exit'),
        help='Evaluate only these groups; omitted groups remain not_evaluated')
    parser.add_argument('--resume',action='store_true',
                        help='Reuse only complete outputs with matching checkpoint and protocol identity')
    args = parser.parse_args()
    if (not args.review_name.replace('_','').isalnum() or
            not args.model_label.replace('_','').isalnum() or
            not set(args.seeds).issubset({19,37,53}) or
            not 1 <= args.envs_per_command <= 16):
        parser.error('Invalid bounded review options')
    job = args.job.resolve(strict=True)
    checkpoint = args.checkpoint.resolve(strict=True)
    spec = validate_spec(json.loads(args.spec.read_text()))
    source_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    evaluator_sha = hashlib.sha256((Path(__file__).resolve().parents[1]/'scripts/'
        'evaluate_policy_comparison.py').read_bytes()).hexdigest()
    review = job / 'phase_reviews' / args.review_name
    review.mkdir(parents=True, exist_ok=True)
    holdout = json.loads((job/'holdout_commands.json').read_text())['pairs']
    train_pairs = [tuple(pair) for pair in spec['stage']['pairs']]
    if set(map(tuple,holdout)) & set(train_pairs):
        raise ValueError('Holdout command appears in the training bank')
    retention = list(dict.fromkeys(stage_bank('basic_motion')))
    assert len(retention) == 25
    configurations = {
        'retention':(retention,None,None),
        'turn_train':([(sv*v,sw*w) for v,w in train_pairs
                       for sv in (-1.,1.) for sw in (-1.,1.)],spec,None),
        'turn_holdout':([(sv*v,sw*w) for v,w in holdout
                         for sv in (-1.,1.) for sw in (-1.,1.)],spec,None),
        'entry_exit':(entry_exit_pairs(spec,train_pairs),spec,12),
        'slow_reverse':([(sv*2.,sw*.5) for sv in (-1.,1.) for sw in (-1.,1.)],spec,11),
    }
    if spec['stage'].get('cohort_plan') is not None:
        height_bank = spec['stage']['cohort_plan']['height_bank']
        height_pairs = [(vx,0.) for vx, _ in height_bank]
        configurations['height_skill'] = (height_pairs,spec,None)
        configurations['height_entry_exit'] = (height_pairs,spec,12)
    if args.groups is not None:
        if len(set(args.groups)) != len(args.groups) or not set(args.groups).issubset(configurations):
            parser.error('Requested review groups are duplicated or unavailable')
        configurations = {name:configurations[name] for name in args.groups}
    commands = []
    for seed in args.seeds:
        folder = review / 'acceptance' / args.model_label / ('seed'+str(seed))
        folder.mkdir(parents=True, exist_ok=True)
        for group,(pairs,turn_spec,exit_at) in configurations.items():
            speeds,yaws=zip(*pairs)
            heights=([height for _, height in height_bank]
                     if group in ('height_skill','height_entry_exit')
                     else target_heights(pairs,turn_spec))
            out = folder / (group+'.json')
            if out.exists() and not args.resume:
                raise FileExistsError(out)
            argv = command(root,checkpoint,out,speeds,yaws,heights,seed,
                args.envs_per_command,spec=turn_spec,exit_at=exit_at,
                reverse=group=='slow_reverse')
            commands.append(argv)
            if out.exists():
                data = reusable_result(out,argv,source_sha,evaluator_sha)
            else:
                with (folder/(group+'.log')).open('x') as log:
                    subprocess.run(argv,cwd=Path(root)/'plane',check=True,
                                   stdout=log,stderr=subprocess.STDOUT)
                data=reusable_result(out,argv,source_sha,evaluator_sha)
            if data['checkpoint_sha256']!=source_sha:
                raise ValueError('Review checkpoint hash changed')
    manifest=review/(args.model_label+'_commands.json')
    content={
        'checkpoint':str(checkpoint),'checkpoint_sha256':source_sha,
        'spec':str(args.spec.resolve()),'commands':commands,
        'requested_groups':list(configurations),
        'missing_groups':sorted(set(('retention','turn_train','turn_holdout',
            'entry_exit','slow_reverse'))-set(configurations)),
        'expected_files':[argv[argv.index('--out')+1] for argv in commands]}
    if manifest.exists() and json.loads(manifest.read_text()) != content:
        legacy=json.loads(manifest.read_text())
        if not args.resume or any(legacy.get(key)!=content[key] for key in
                                  ('checkpoint','checkpoint_sha256','spec','commands')):
            raise ValueError('Existing review manifest differs')
    else:
        manifest.write_text(json.dumps(content,indent=2)+'\n')
    print(json.dumps({'review':str(review),'model':args.model_label,
                      'seeds':args.seeds,'groups':list(configurations)}))
