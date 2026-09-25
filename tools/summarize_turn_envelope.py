"""Summarize measured turn points without interpolating untested commands."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from wheel_legged_gym.evaluation.gates import gate


def turn_checks(row):
    metrics = row['metrics']
    checks = {
        'original_gate': gate(row)['passed'],
        'yaw_mae_0p05': metrics['yaw_mae'] <= .05,
        'lateral_velocity_0p10': metrics['abs_lateral_velocity'] <= .10,
        'slip_0p10': metrics['slip_rms'] <= .10,
        'pitch_0p10': metrics['abs_pitch'] <= .10,
        'roll_0p10': metrics['abs_roll'] <= .10,
    }
    return {'passed': all(checks.values()), 'checks': checks}


def collect(job):
    rows = []
    for path in sorted((job / 'eval').glob('*/seed*/*.json')):
        if path.stem not in ('retention', 'turn_low', 'turn_high', 'turn_unseen'):
            continue
        data = json.loads(path.read_text())
        model, seed = path.parent.parent.name, int(path.parent.name[4:])
        for row in data['results']:
            command = row['command']
            rows.append({'model': model, 'seed': seed, 'group': path.stem,
                         'command': command, 'original_gate': gate(row),
                         'turn_gate': turn_checks(row) if path.stem != 'retention' else None,
                         'failure_count': row['failure_count'],
                         'timeout_count': row['timeout_count'],
                         'survival_fraction': row['survival_fraction'],
                         'metrics': row['metrics'], 'raw_json': str(path)})
    return rows


def grouped(rows):
    bins = defaultdict(list)
    for row in rows:
        bins[(row['model'], row['group'], *row['command'][:2])].append(row)
    result = []
    for (model, group, vx, yaw), members in sorted(bins.items()):
        seeds = sorted({r['seed'] for r in members})
        complete = seeds == [19, 37, 53]
        key = 'original_gate' if group == 'retention' else 'turn_gate'
        passed = complete and all(r[key]['passed'] for r in members)
        result.append({'model': model, 'group': group, 'command': [vx, yaw, .4],
                       'status': 'unmeasured' if not complete else ('passed' if passed else 'failed'),
                       'seeds': seeds, 'environments': sum(16 for _ in members),
                       'max_vx_mae': max(r['metrics']['vx_mae'] for r in members),
                       'max_yaw_mae': max(r['metrics']['yaw_mae'] for r in members),
                       'max_slip_rms': max(r['metrics']['slip_rms'] for r in members),
                       'max_abs_lateral_velocity': max(r['metrics'].get('abs_lateral_velocity', 0) for r in members),
                       'max_abs_roll': max(r['metrics']['abs_roll'] for r in members),
                       'mean_signed_roll': sum(r['metrics'].get('roll', 0) for r in members) / len(members),
                       'mean_roll_target': sum(r['metrics'].get('roll_target', 0) for r in members) / len(members),
                       'failures': sum(r['failure_count'] for r in members),
                       'timeouts': sum(r['timeout_count'] for r in members),
                       'worst_seed': max(members, key=lambda r: (r['failure_count'], r['metrics']['yaw_mae']))['seed'],
                       'raw_json': [r['raw_json'] for r in members]})
    return result


def transitions(job):
    summaries = []
    for path in sorted((job / 'eval').glob('*/seed*/*.json')):
        if path.stem not in ('entry_exit', 'slow_reverse'):
            continue
        data = json.loads(path.read_text())
        n = data['envs_per_command']
        windows = ({'entry': (1., 8.), 'turn': (8., 11.9),
                    'exit_ramp': (12., 14.), 'recovery': (16., 18.)}
                   if path.stem == 'entry_exit' else
                   {'entry': (1., 8.), 'turn': (8., 10.9),
                    'reverse_ramp': (11., 15.), 'reverse_hold': (16., 19.)})
        for index, row in enumerate(data['results']):
            lo, hi = index*n, (index+1)*n
            phases = {}
            for name, (start, end) in windows.items():
                frames = [frame for frame in data['response_trace']
                          if start <= frame['time'] < end]
                if not frames:
                    raise ValueError('Missing transition frames: ' + str(path))
                count = len(frames)*n
                def average(expression):
                    return sum(expression(frame, env) for frame in frames
                               for env in range(lo, hi)) / count
                phases[name] = {
                    'vx_mae': average(lambda f,e: abs(f['vx'][e]-f['applied_command'][e][0])),
                    'yaw_mae': average(lambda f,e: abs(f['yaw'][e]-f['applied_command'][e][1])),
                    'height_mae': average(lambda f,e: abs(f['height'][e]-.4)),
                    'abs_roll': average(lambda f,e: abs(f['roll'][e])),
                    'mean_yaw_command': average(lambda f,e: f['applied_command'][e][1]),
                    'wheel_contact': average(lambda f,e: float(all(f['wheel_contacts'][e]))),
                }
            final = phases['recovery' if path.stem == 'entry_exit' else 'reverse_hold']
            passed = (row['failure_count'] == 0 and row['timeout_count'] == 0 and
                      row['metrics']['nonwheel_contact'] == 0 and
                      row['metrics']['torque_saturation'] <= .01 and
                      final['vx_mae'] <= .10 and final['yaw_mae'] <= .10 and
                      final['height_mae'] <= .03 and final['wheel_contact'] >= .99)
            summaries.append({'model': path.parent.parent.name,
                              'seed': int(path.parent.name[4:]), 'protocol': path.stem,
                              'command': row['command'], 'failure_count': row['failure_count'],
                              'timeout_count': row['timeout_count'],
                              'final_phase_passed': passed, 'phases': phases,
                              'raw_json': str(path)})
    return summaries


def heatmap(path, points):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    import numpy as np

    speeds, yaws = [.5, 1., 2., 3., 4.], [.25, .5, .75, 1.]
    lookup = {(p['model'], p['command'][0], p['command'][1]): p for p in points
              if p['group'] in ('turn_low', 'turn_high')}
    fig, axes = plt.subplots(3, 4, figsize=(14, 9), sharex=True, sharey=True)
    colors = ListedColormap(['#e5e7eb', '#cf5146', '#28966e'])
    for i, model in enumerate(('R', 'A', 'B')):
        for j, (sv, sw) in enumerate(((-1,-1),(-1,1),(1,-1),(1,1))):
            ax = axes[i, j]
            cells = np.zeros((len(speeds), len(yaws)), dtype=int)
            for r, v in enumerate(speeds):
                for c, w in enumerate(yaws):
                    point = lookup.get((model, sv*v, sw*w))
                    status = point['status'] if point else 'unmeasured'
                    cells[r,c] = {'unmeasured':0,'failed':1,'passed':2}[status]
                    ax.text(c, r, {'unmeasured':'-', 'failed':'F', 'passed':'P'}[status],
                            ha='center', va='center', color='#111827', fontsize=10)
            ax.imshow(cells, cmap=colors, vmin=0, vmax=2, aspect='auto')
            ax.set_title(f'{model}: vx {"+" if sv>0 else "-"}, yaw {"+" if sw>0 else "-"}')
            ax.set_xticks(range(len(yaws)), yaws)
            ax.set_yticks(range(len(speeds)), speeds)
            ax.set_xlabel('|yaw| rad/s')
            ax.set_ylabel('|vx| m/s')
    fig.suptitle('Strict turn gate, all seeds (16 env/seed); P pass, F fail, - unmeasured')
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def extract_failures(job):
    summary = json.loads((job / 'evaluation_summary.json').read_text())
    failures = []
    for point in summary['points']:
        if point['status'] != 'failed' or point['group'] == 'retention':
            continue
        source = next(Path(path) for path in point['raw_json']
                      if '/seed{}/'.format(point['worst_seed']) in path)
        data = json.loads(source.read_text())
        index = next(i for i, row in enumerate(data['results'])
                     if row['command'] == point['command'])
        row = data['results'][index]
        per_env = row['per_env_metrics']['yaw_mae']
        local_env = max(range(len(per_env)), key=lambda i: per_env[i])
        env = index * data['envs_per_command'] + local_env
        trajectory = [{**{key: frame[key][env] for key in
                          ('applied_command', 'policy_command_scaled',
                           'history_latest_command_scaled', 'vx', 'yaw', 'vy',
                           'roll', 'roll_target', 'height', 'wheel_contacts', 'base_xy')},
                       'time': frame['time']} for frame in data['response_trace']]
        failures.append({'model': point['model'], 'group': point['group'],
                         'command': point['command'], 'seed': point['worst_seed'],
                         'environment_index': env,
                         'per_environment_yaw_mae': per_env[local_env],
                         'raw_json': str(source), 'trajectory': trajectory})
    out = job / 'failure_trajectories.json'
    with out.open('x') as output:
        json.dump({'schema':'turn_failure_trajectories_v1',
                   'failed_points':failures}, output, indent=2)
        output.write('\n')
    return out


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--extract-failures-only', action='store_true')
    args = parser.parse_args()
    job = args.job.resolve(strict=True)
    if args.extract_failures_only:
        print(extract_failures(job))
        return
    rows = collect(job)
    points = grouped(rows)
    dynamic = transitions(job)
    summary = job / 'evaluation_summary.json'
    figure = job / 'turn_heatmap.png'
    if summary.exists() or figure.exists():
        raise FileExistsError('Summary artifacts already exist')
    with summary.open('x') as output:
        json.dump({'schema':'turn_envelope_review_v1','individual_rows':rows,
                   'points':points, 'transitions':dynamic}, output, indent=2)
        output.write('\n')
    heatmap(figure, points)
    print(json.dumps({'rows':len(rows),'points':len(points),
                      'summary':str(summary),'heatmap':str(figure)}))


if __name__ == '__main__':
    main()
