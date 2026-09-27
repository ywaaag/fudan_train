"""Summarize measured turn points without interpolating untested commands."""
import argparse
import hashlib
import json
import math
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


def lean_aware_checks(row):
    metrics = row['metrics']
    per_env = row['per_env_metrics']
    target = abs(metrics['roll_target'])
    peak_roll = row['max_abs_metrics']['abs_roll']
    checks = {
        'original_gate': gate(row)['passed'],
        'all_envs_survive': row['survival_fraction'] == 1 and row['timeout_count'] == 0,
        'all_envs_track_vx': max(per_env['vx_mae']) <= .10,
        'all_envs_track_yaw': max(per_env['yaw_mae']) <= .05,
        'all_envs_height_mae': max(per_env['height_mae']) <= .03,
        'all_envs_wheel_contact': (min(per_env['left_contact']) >= .99 and
                                   min(per_env['right_contact']) >= .99),
        'height_reached': max(per_env['height']) <= metrics['height_target'] + .015,
        'roll_target_mae': max(per_env['roll_target_mae']) <= .035,
        'roll_overshoot': max(per_env['roll_overshoot']) <= .035,
        'roll_safety_cap': peak_roll <= min(.25, target+.05),
        'pitch': max(per_env['abs_pitch']) <= .10,
        'lateral_velocity': max(per_env['abs_lateral_velocity']) <= .10,
        'slip': max(per_env['slip_rms']) <= .10,
        'soft_joint_margin': min(per_env['soft_joint_limit_margin']) >= 0.,
    }
    return {'passed':all(checks.values()), 'checks':checks,
            'historical_strict_turn':turn_checks(row)}


def height_skill_checks(row):
    per_env = row['per_env_metrics']
    checks = {
        'original_gate': gate(row)['passed'],
        'all_envs_survive': row['survival_fraction'] == 1 and row['timeout_count'] == 0,
        'all_envs_height_mae': max(per_env['height_mae']) <= .015,
        'all_envs_vx_mae': max(per_env['vx_mae']) <=
            (.05 if row['command'][0] == 0 else .10),
        'all_envs_yaw_mae': max(per_env['yaw_mae']) <= .05,
        'all_envs_contact': (min(per_env['left_contact']) >= .99 and
                             min(per_env['right_contact']) >= .99),
        'all_envs_slip': max(per_env['slip_rms']) <= .10,
        'all_envs_pitch': max(per_env['abs_pitch']) <= .10,
        'all_envs_joint_margin': min(per_env['soft_joint_limit_margin']) >= 0.,
    }
    return {'passed': all(checks.values()), 'checks': checks}


def require_geometry_direction(verdict, case, row=None, protocol=None):
    checks = {'independent_geometry_inward':case is not None and case['status']=='inward'}
    if protocol == 'inward_stage1_v1':
        environments = case['environments'] if case is not None else []
        checks['geometry_coverage_0p80'] = bool(environments) and all(
            env['valid_windows'] >= .8 * (env['valid_windows'] +
                sum(env['rejected_windows'].values())) for env in environments)
        checks['inward_projection_0p5deg'] = bool(environments) and all(
            env['mean_up_dot_inward'] is not None and
            env['mean_up_dot_inward'] >= math.sin(math.radians(.5))
            for env in environments)
        checks['roll_target_mae_0p02'] = (row is not None and
            max(row['per_env_metrics']['roll_target_mae']) <= .02)
    verdict['checks'].update(checks)
    verdict['passed'] = verdict['passed'] and all(checks.values())
    return verdict


def collect_long(job, inward_protocol=None):
    rows = []
    for path in sorted((job / 'acceptance').glob('*/seed*/*.json')):
        if path.stem not in ('retention', 'turn_train', 'turn_holdout', 'height_skill'):
            continue
        data = json.loads(path.read_text())
        model, seed = path.parent.parent.name, int(path.parent.name[4:])
        geometry_cases = (geometry_sign_audit(data)['cases']
                          if data.get('geometry_trace') and data.get('response_trace') else None)
        for row_index, row in enumerate(data['results']):
            verdict = (height_skill_checks(row) if path.stem == 'height_skill'
                       else lean_aware_checks(row) if path.stem != 'retention'
                       else {'passed':gate(row)['passed'] and
                             max(row['metrics']['knee_mirror_m'],
                                 row['metrics']['wheel_mirror_m']) <= .03,
                             'original_gate':gate(row)})
            active_lean = (geometry_cases is not None and
                           path.stem in ('turn_train','turn_holdout') and
                           abs(row['metrics']['roll_target']) >= math.radians(1.))
            if active_lean or (inward_protocol and path.stem in ('turn_train','turn_holdout')
                               and abs(row['metrics']['roll_target']) >= math.radians(1.)):
                verdict = require_geometry_direction(verdict,
                    geometry_cases[row_index] if geometry_cases is not None else None,
                    row, inward_protocol)
            rows.append({'model':model, 'seed':seed, 'group':path.stem,
                         'command':row['command'], 'environments':data['envs_per_command'],
                         'verdict':verdict,
                         'worst_environment_yaw_mae':max(row['per_env_metrics']['yaw_mae']),
                         'worst_environment_index':max(
                             range(len(row['per_env_metrics']['yaw_mae'])),
                             key=lambda index:row['per_env_metrics']['yaw_mae'][index]),
                         'worst_environment_slip_rms':max(row['per_env_metrics']['slip_rms']),
                         'worst_environment_contact':min(
                             row['per_env_metrics']['left_contact']+
                             row['per_env_metrics']['right_contact']),
                         'geometry_direction_status':(
                             geometry_cases[row_index]['status'] if active_lean else
                             'neutral_not_required' if geometry_cases is not None else
                             'legacy_unmeasured'),
                         'failure_count':row['failure_count'],
                         'reset_reason_counts':row.get('reset_reason_counts'),
                         'metrics':row['metrics'], 'max_abs_metrics':row['max_abs_metrics'],
                         'raw_json':str(path)})
    return rows


def long_evaluation_identity(job, inward_protocol, metric_sha, gate_sha):
    manifests = sorted(job.glob('*_commands.json'))
    manifest = json.loads(manifests[0].read_text()) if len(manifests) == 1 else {}
    checkpoint = manifest.get('checkpoint')
    spec_path = Path(manifest.get('spec', '')) if manifest.get('spec') else None
    spec = json.loads(spec_path.read_text()) if spec_path and spec_path.is_file() else {}
    stage = spec.get('stage', {})
    protocol = ('aggressive_cornering_v2' if stage.get('aggressive_plan', {}).get(
                    'sampling_version') == 2 else
                'cornering_height_skill_v1' if stage.get('cohort_plan') else
                inward_protocol or 'legacy_unknown')
    requested = []
    for argv in manifest.get('commands', []):
        if '--out' in argv:
            requested.append(str(Path(argv[argv.index('--out')+1])))
    records = []
    for path in sorted((job/'acceptance').glob('*/seed*/*.json')):
        data = json.loads(path.read_text())
        rows = data.get('results', [])
        strict_safety = [
            min(row['per_env_metrics']['left_contact']) >= .99 and
            min(row['per_env_metrics']['right_contact']) >= .99 and
            max(row['per_env_metrics']['slip_rms']) <= .10
            for row in rows]
        records.append({'path':str(path), 'checkpoint':data.get('checkpoint'),
            'checkpoint_sha256':data.get('checkpoint_sha256'),
            'evaluator_sha256':data.get('evaluator_sha256'),
            'seed':data.get('seed'), 'envs_per_command':data.get('envs_per_command'),
            'commands':[row.get('command') for row in rows],
            'geometry_trace':data.get('geometry_trace',False),
            'completed':bool(rows),
            'old_gate_passed':sum(gate(row)['passed'] for row in rows),
            'strict_contact_slip_passed':sum(strict_safety),
            'strict_contact_slip_total':len(rows),
            'failure_count':sum(row.get('failure_count',0) for row in rows),
            'timeout_count':sum(row.get('timeout_count',0) for row in rows),
            'reset_count':sum(row.get('failure_count',0)+row.get('timeout_count',0)
                              for row in rows),
            'skipped':False})
    completed = {record['path'] for record in records if record['completed']}
    if records and (len({record['checkpoint_sha256'] for record in records}) != 1 or
                    len({record['evaluator_sha256'] for record in records}) != 1 or
                    (manifest.get('checkpoint_sha256') and
                     records[0]['checkpoint_sha256'] != manifest['checkpoint_sha256'])):
        raise ValueError('Review mixes checkpoint or evaluator identities')
    return {'protocol_id':protocol, 'gate_id':gate_sha,
        'metric_definition_sha256':metric_sha,
        'checkpoint':checkpoint,
        'checkpoint_sha256':manifest.get('checkpoint_sha256'),
        'actual_iteration':int(Path(checkpoint).stem.split('_')[-1])
            if checkpoint and Path(checkpoint).stem.split('_')[-1].isdigit() else None,
        'requested_files':requested if requested else None,
        'completed_files':sorted(completed),
        'missing_files':sorted(set(requested)-completed) if requested else None,
        'skipped_files':[],
        'evaluation_records':records}


def grouped_long(rows):
    bins = defaultdict(list)
    for row in rows:
        bins[(row['model'],row['group'],*row['command'])].append(row)
    points = []
    for (model,group,vx,yaw,height), members in sorted(bins.items()):
        seeds = sorted({row['seed'] for row in members})
        complete = seeds == [19,37,53]
        passed = complete and all(row['verdict']['passed'] for row in members)
        points.append({'model':model,'group':group,'command':[vx,yaw,height],
            'status':'unmeasured' if not complete else ('passed' if passed else 'failed'),
            'seeds':seeds,'environments':sum(row['environments'] for row in members),
            'max_vx_mae':max(row['metrics']['vx_mae'] for row in members),
            'max_yaw_mae':max(row['metrics']['yaw_mae'] for row in members),
            'max_height_mae':max(row['metrics']['height_mae'] for row in members),
            'mean_com_height':sum(row['metrics']['com_height'] for row in members)/len(members),
            'max_roll_target_mae':max(row['metrics']['roll_target_mae'] for row in members),
            'max_abs_lateral_velocity':max(row['metrics']['abs_lateral_velocity'] for row in members),
            'max_slip_rms':max(row['metrics']['slip_rms'] for row in members),
            'failures':sum(row['failure_count'] for row in members),
            'worst_seed':max(members,key=lambda row:(row['failure_count'],
                row['metrics']['yaw_mae']))['seed'],
            'worst_environment_seed':max(members,
                key=lambda row:row['worst_environment_yaw_mae'])['seed'],
            'worst_environment_index':max(members,
                key=lambda row:row['worst_environment_yaw_mae'])['worst_environment_index'],
            'raw_json':[row['raw_json'] for row in members]})
    return points


def transitions_long(job):
    summaries=[]
    for path in sorted((job/'acceptance').glob('*/seed*/*.json')):
        if path.stem not in ('entry_exit','slow_reverse','height_entry_exit'):
            continue
        data=json.loads(path.read_text())
        n=data['envs_per_command']
        windows=({'entry':(1.,8.),'turn':(8.,11.9),
                  'exit_ramp':(12.,14.),'recovery':(16.,18.)}
                 if path.stem in ('entry_exit','height_entry_exit') else
                 {'entry':(1.,8.),'turn':(8.,10.9),
                  'reverse_ramp':(11.,15.),'reverse_hold':(16.,19.)})
        for index,row in enumerate(data['results']):
            lo,hi=index*n,(index+1)*n
            phases={}
            for name,(start,end) in windows.items():
                frames=[frame for frame in data['response_trace']
                        if start<=frame['time']<end]
                if not frames:raise ValueError('Missing dynamic frames: '+str(path))
                count=len(frames)*n
                def average(fn):
                    return sum(fn(frame,env) for frame in frames
                               for env in range(lo,hi))/count
                def worst_env(fn):
                    return max(sum(fn(frame,env) for frame in frames)/len(frames)
                               for env in range(lo,hi))
                phases[name]={
                    'vx_mae':average(lambda f,e:abs(f['vx'][e]-f['applied_command'][e][0])),
                    'yaw_mae':average(lambda f,e:abs(f['yaw'][e]-f['applied_command'][e][1])),
                    'height_mae':average(lambda f,e:abs(f['height'][e]-f['applied_command'][e][2])),
                    'worst_env_height_mae':worst_env(lambda f,e:abs(f['height'][e]-f['applied_command'][e][2])),
                    'root_height':average(lambda f,e:f['height'][e]),
                    'com_height':average(lambda f,e:f['com_height'][e]),
                    'height_target':average(lambda f,e:f['applied_command'][e][2]),
                    'roll_target_mae':average(lambda f,e:abs(f['roll'][e]-f['roll_target'][e])),
                    'wheel_contact':average(lambda f,e:float(all(f['wheel_contacts'][e]))),
                }
            final=phases['recovery' if path.stem in ('entry_exit','height_entry_exit') else 'reverse_hold']
            passed=(row['failure_count']==0 and row['timeout_count']==0 and
                    row['metrics']['nonwheel_contact']==0 and
                    row['metrics']['torque_saturation']<=.01 and
                    final['vx_mae']<=.10 and final['yaw_mae']<=.10 and
                    final['height_mae']<=.03 and final['wheel_contact']>=.99 and
                    final['roll_target_mae']<=.035)
            if path.stem == 'height_entry_exit':
                passed = (passed and phases['turn']['worst_env_height_mae']<=.015
                          and final['worst_env_height_mae']<=.03
                          and phases['turn']['wheel_contact']>=.99)
            summaries.append({'model':path.parent.parent.name,
                'seed':int(path.parent.name[4:]),'protocol':path.stem,
                'command':row['command'],'environments':n,
                'failure_count':row['failure_count'],
                'reset_reason_counts':row.get('reset_reason_counts'),
                'final_phase_passed':passed,'phases':phases,'raw_json':str(path)})
    return summaries


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


def heatmap_long(path, points):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    import numpy as np

    models = sorted({p['model'] for p in points}, key=lambda label:(label!='R',label))
    turns = [p for p in points if p['group'] in ('turn_train','turn_holdout')]
    speeds = sorted({abs(p['command'][0]) for p in turns})
    yaws = sorted({abs(p['command'][1]) for p in turns})
    if not models or not speeds or not yaws:
        raise ValueError('No measured turn points for heatmap')
    lookup = {(p['model'],p['command'][0],p['command'][1]):p for p in turns}
    fig, axes = plt.subplots(len(models),4,figsize=(14,max(3,len(models)*3)),
                             squeeze=False,sharex=True,sharey=True)
    colors = ListedColormap(['#e5e7eb','#cf5146','#28966e'])
    for i, model in enumerate(models):
        for j,(sv,sw) in enumerate(((-1,-1),(-1,1),(1,-1),(1,1))):
            ax=axes[i,j]
            cells=np.zeros((len(speeds),len(yaws)),dtype=int)
            for r,v in enumerate(speeds):
                for c,w in enumerate(yaws):
                    p=lookup.get((model,sv*v,sw*w))
                    status=p['status'] if p else 'unmeasured'
                    cells[r,c]={'unmeasured':0,'failed':1,'passed':2}[status]
                    ax.text(c,r,{'unmeasured':'-','failed':'F','passed':'P'}[status],
                            ha='center',va='center',fontsize=8)
            ax.imshow(cells,cmap=colors,vmin=0,vmax=2,aspect='auto')
            ax.set_title('{}: vx {}, yaw {}'.format(model,'+' if sv>0 else '-',
                '+' if sw>0 else '-'))
            ax.set_xticks(range(len(yaws)),yaws)
            ax.set_yticks(range(len(speeds)),speeds)
            ax.set_xlabel('|yaw| rad/s')
            ax.set_ylabel('|vx| m/s')
    fig.suptitle('Lean-aware discrete gate: P pass, F fail, - unmeasured')
    fig.tight_layout()
    fig.savefig(path,dpi=160)
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


def geometry_sign_audit(data):
    """Infer inward direction from adjacent world velocities, independently of roll targets."""
    frames = data['response_trace']
    width = data['envs_per_command']
    cases = []
    for index, row in enumerate(data['results']):
      environments = []
      for env in range(index * width, (index + 1) * width):
        valid = []
        rejected = {'speed':0,'curvature':0,'contact':0,'slip':0,'reset':0}
        for previous, current in zip(frames, frames[1:]):
            if current['time'] < data['warmup']:
                continue
            velocity = previous['velocity_world'][env][:2]
            next_velocity = current['velocity_world'][env][:2]
            speed = math.hypot(*velocity)
            if speed < .3:
                rejected['speed'] += 1
                continue
            delta = [next_velocity[k]-velocity[k] for k in (0,1)]
            cross = velocity[0]*delta[1]-velocity[1]*delta[0]
            curvature = cross / max(speed**3*(current['time']-previous['time']),1e-9)
            if abs(curvature) < .03:
                rejected['curvature'] += 1
                continue
            if not all(current['wheel_contacts'][env]):
                rejected['contact'] += 1
                continue
            if row['per_env_metrics']['slip_rms'][env-index*width] > .10:
                rejected['slip'] += 1
                continue
            if current['resets'][env]:
                rejected['reset'] += 1
                continue
            sign = 1 if cross > 0 else -1
            inward = [-sign*velocity[1]/speed, sign*velocity[0]/speed]
            up = current['body_axes_world'][env][2][:2]
            support = [sum(w[k] for w in current['wheel_positions_world'][env])/2
                       for k in (0,1)]
            com = current['com_position_world'][env][:2]
            valid.append({'time':current['time'],'curvature':curvature,
                'inward_world_xy':inward,'body_up_horizontal_world_xy':up,
                'up_dot_inward':sum(up[k]*inward[k] for k in (0,1)),
                'com_support_inward_m':sum((com[k]-support[k])*inward[k] for k in (0,1)),
                'velocity_world_xy':next_velocity,'body_axes_world':current['body_axes_world'][env],
                'root_world_xy':current['root_position_world'][env][:2],
                'wheel_center_world_xy':[wheel[:2] for wheel in current['wheel_positions_world'][env]],
                'com_world_xy':com,'projected_gravity':current['projected_gravity'][env],
                'root_quaternion_xyzw':current['root_quaternion_xyzw'][env]})
        dot = sum(item['up_dot_inward'] for item in valid)/len(valid) if valid else None
        com_dot = sum(item['com_support_inward_m'] for item in valid)/len(valid) if valid else None
        inward_fraction = (sum(item['up_dot_inward'] > 0 for item in valid)/len(valid)
                           if valid else None)
        midpoint = len(valid)//2
        environments.append({'environment_index':env,
            'status':('unreliable' if not valid else 'inward' if inward_fraction >= .9
                      else 'outward' if inward_fraction <= .1 else 'mixed'),
            'valid_windows':len(valid),'rejected_windows':rejected,
            'inward_window_fraction':inward_fraction,
            'mean_up_dot_inward':dot,'mean_com_support_inward_m':com_dot,
            'mean_roll_rad':row['per_env_metrics']['roll'][env-index*width],
            'mean_slip_rms_m_s':row['per_env_metrics']['slip_rms'][env-index*width],
            'example':valid[midpoint] if valid else None,
            'short_trajectory_world_xy':[item['root_world_xy'] for item in
                valid[max(0,midpoint-5):midpoint+6]]})
      reliable = [item for item in environments if item['valid_windows']]
      cases.append({'command':row['command'],
          'status':('unreliable' if len(reliable)!=width else
              'inward' if all(item['status']=='inward' for item in reliable) else
              'outward' if all(item['status']=='outward' for item in reliable) else 'mixed'),
          'valid_windows':sum(item['valid_windows'] for item in environments),
          'environment_count':width,'reliable_environment_count':len(reliable),
          'mean_up_dot_inward':(sum(item['mean_up_dot_inward'] for item in reliable)/len(reliable)
                                  if reliable else None),
          'worst_environment_up_dot_inward':(min(item['mean_up_dot_inward'] for item in reliable)
                                              if reliable else None),
          'mean_com_support_inward_m':(sum(item['mean_com_support_inward_m'] for item in reliable)/len(reliable)
                                        if reliable else None),
          'mean_roll_rad':row['metrics']['roll'],
          'mean_slip_rms_m_s':row['metrics']['slip_rms'],
          'failure_count':row['failure_count'],'environments':environments,
          'example':reliable[0]['example'] if reliable else None,
          'short_trajectory_world_xy':reliable[0]['short_trajectory_world_xy'] if reliable else []})
    return {'schema':'turn_geometry_sign_v1','checkpoint':data['checkpoint'],
        'checkpoint_sha256':data['checkpoint_sha256'],'evaluator_sha256':data['evaluator_sha256'],
        'seed':data['seed'],'method':'world velocity finite-difference curvature; all environments per command',
        'wheel_marker':'rigid-body wheel center projected to world XY; not measured contact patch',
        'acceptance_scope':'independent direction diagnostic; historical roll-target gate unchanged',
        'cases':cases}


def geometry_plot(path, audit):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(audit['cases']), figsize=(5*len(audit['cases']),5))
    if len(audit['cases']) == 1:
        axes = [axes]
    for ax, case in zip(axes, audit['cases']):
        sample = case['example']
        if sample is None:
            ax.set_title('{}: unreliable'.format(case['command'][:2]))
            continue
        root = sample['root_world_xy']
        trajectory = case['short_trajectory_world_xy']
        if trajectory:
            ax.plot([point[0] for point in trajectory],
                    [point[1] for point in trajectory],color='#555555',
                    linewidth=1.5,label='short-window trajectory')
        for label, vector, color in (
                ('velocity',sample['velocity_world_xy'],'#1767a6'),
                ('inward',sample['inward_world_xy'],'#198754'),
                ('body +X',sample['body_axes_world'][0][:2],'#a56410'),
                ('body +Y',sample['body_axes_world'][1][:2],'#7941a1'),
                ('body +Z XY',sample['body_up_horizontal_world_xy'],'#c5233e')):
            length = math.hypot(*vector)
            if length:
                ax.arrow(root[0],root[1],.35*vector[0]/length,.35*vector[1]/length,
                         color=color,head_width=.035,length_includes_head=True,label=label)
        for name, point in zip(('left wheel center','right wheel center'),sample['wheel_center_world_xy']):
            ax.scatter(*point,s=40,label=name)
        ax.scatter(*sample['com_world_xy'],marker='x',s=65,color='black',label='COM')
        ax.set_title('{}: {} (t={:.1f}s)'.format(case['command'][:2],case['status'],sample['time']))
        ax.set_xlabel('world X (m)'); ax.set_ylabel('world Y (m)')
        ax.set_aspect('equal'); ax.grid(True); ax.legend(fontsize=7,loc='best')
    fig.suptitle('World XY, +Z toward viewer; arrows normalized for direction, not magnitude')
    fig.tight_layout(); fig.savefig(path,dpi=150); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--job', type=Path)
    parser.add_argument('--extract-failures-only', action='store_true')
    parser.add_argument('--long', action='store_true', help='Summarize the new variable-height exploration')
    parser.add_argument('--inward-protocol', choices=['inward_stage1_v1'],
                        help='Apply the frozen independent direction/coverage/amplitude gate')
    parser.add_argument('--summary-out', type=Path,
                        help='With --long, write a new versioned summary without altering historical artifacts')
    parser.add_argument('--geometry-json', type=Path,
                        help='Analyze one evaluator JSON containing --geometry-trace')
    parser.add_argument('--geometry-out', type=Path,
                        help='New JSON summary path; also writes a PNG at the same stem')
    parser.add_argument('--geometry-no-plot', action='store_true',
                        help='For repeated milestone probes, write only the geometry JSON')
    args = parser.parse_args()
    if args.geometry_json is not None:
        if args.geometry_out is None:
            parser.error('--geometry-json requires --geometry-out')
        data = json.loads(args.geometry_json.read_text())
        if not data.get('geometry_trace') or not data.get('response_trace'):
            raise ValueError('Input lacks geometry response trace')
        audit = geometry_sign_audit(data)
        if args.geometry_out.exists() or (not args.geometry_no_plot and
                                          args.geometry_out.with_suffix('.png').exists()):
            raise FileExistsError(args.geometry_out)
        with args.geometry_out.open('x') as output:
            json.dump(audit,output,indent=2)
            output.write('\n')
        if not args.geometry_no_plot:
            geometry_plot(args.geometry_out.with_suffix('.png'),audit)
        print(json.dumps({'summary':str(args.geometry_out),
                          'figure':str(args.geometry_out.with_suffix('.png'))}))
        return
    if args.job is None:
        parser.error('--job is required unless --geometry-json is used')
    if args.summary_out is not None and not args.long:
        parser.error('--summary-out requires --long')
    if args.inward_protocol is not None and not args.long:
        parser.error('--inward-protocol requires --long')
    job = args.job.resolve(strict=True)
    if args.extract_failures_only:
        print(extract_failures(job))
        return
    if args.long:
        rows = collect_long(job,args.inward_protocol)
        points = grouped_long(rows)
        dynamic = transitions_long(job)
        summary = args.summary_out or job / 'long_evaluation_summary.json'
        figure = None if args.summary_out is not None else job / 'long_turn_heatmap.png'
        if summary.exists() or (figure is not None and figure.exists()):
            raise FileExistsError('Long exploration summary already exists')
        metric_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        gate_sha = hashlib.sha256(Path(gate.__code__.co_filename).read_bytes()).hexdigest()
        identity = long_evaluation_identity(job,args.inward_protocol,metric_sha,gate_sha)
        summary.parent.mkdir(parents=True,exist_ok=True)
        with summary.open('x') as output:
            json.dump({'schema':'turn_lean_long_review_v3',
                       'metric_definition_sha256':metric_sha,
                       'gate_sha256':gate_sha,
                       'inward_protocol':args.inward_protocol,
                       **identity,
                       'individual_rows':rows,'points':points,
                       'transitions':dynamic},output,indent=2)
            output.write('\n')
        if figure is not None:
            heatmap_long(figure,points)
        print(json.dumps({'rows':len(rows),'points':len(points),
                          'summary':str(summary),'heatmap':str(figure) if figure else None}))
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
