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
               '--trace-stride', '10' if exit_at is not None else '50']
    if spec is None:
        return options + ['--seconds','25','--warmup','5']
    stage = spec['stage']
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


def main(root):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--job',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--spec',type=Path,required=True)
    parser.add_argument('--review-name',required=True)
    parser.add_argument('--model-label',required=True)
    parser.add_argument('--seeds',type=int,nargs='+',default=[19])
    parser.add_argument('--envs-per-command',type=int,default=8)
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
        height_pairs = [(vx,0.) for height in (.38,.36) for vx in (0.,-.5,.5)]
        configurations['height_skill'] = (height_pairs,spec,None)
        configurations['height_entry_exit'] = (height_pairs,spec,12)
    commands = []
    for seed in args.seeds:
        folder = review / 'acceptance' / args.model_label / ('seed'+str(seed))
        folder.mkdir(parents=True, exist_ok=True)
        for group,(pairs,turn_spec,exit_at) in configurations.items():
            speeds,yaws=zip(*pairs)
            heights=([height for height in (.38,.36) for _ in (0.,-.5,.5)]
                     if group in ('height_skill','height_entry_exit')
                     else target_heights(pairs,turn_spec))
            out = folder / (group+'.json')
            if out.exists():
                raise FileExistsError(out)
            argv = command(root,checkpoint,out,speeds,yaws,heights,seed,
                args.envs_per_command,spec=turn_spec,exit_at=exit_at,
                reverse=group=='slow_reverse')
            commands.append(argv)
            with (folder/(group+'.log')).open('x') as log:
                subprocess.run(argv,cwd=Path(root)/'plane',check=True,
                               stdout=log,stderr=subprocess.STDOUT)
            data=json.loads(out.read_text())
            if data['checkpoint_sha256']!=source_sha:
                raise ValueError('Review checkpoint hash changed')
    (review/(args.model_label+'_commands.json')).write_text(json.dumps({
        'checkpoint':str(checkpoint),'checkpoint_sha256':source_sha,
        'spec':str(args.spec.resolve()),'commands':commands},indent=2)+'\n')
    print(json.dumps({'review':str(review),'model':args.model_label,
                      'seeds':args.seeds,'groups':list(configurations)}))
