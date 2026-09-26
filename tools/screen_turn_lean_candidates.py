"""Compare completed turn-lean review summaries; never launch simulation."""
import argparse
import hashlib
import json
import re
from pathlib import Path

SCHEMA = 'turn_lean_candidate_screen_v2'
SEEDS = (19, 37, 53)
GROUPS = ('retention', 'turn_train', 'turn_holdout', 'entry_exit', 'slow_reverse')
TRANSITION_FIELDS = ('switch_at', 'transition_ramp_seconds', 'yaw_delay_seconds',
                     'height_delay_seconds', 'yaw_exit_at', 'yaw_exit_ramp_seconds',
                     'yaw_exit_factor', 'height_return_at_exit', 'lean_reference_max_deg')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(',', ':')).encode()).hexdigest()


def review_location(path):
    path = Path(path).resolve()
    if path.is_file() and path.name == 'turn_train.json':
        model = path.parent.parent.name
        return path.parent.parent.parent.parent, model
    if path.is_file() and path.name in ('long_evaluation_summary.json',
                                       'long_evaluation_summary_v2.json'):
        return path.parent, None
    if path.is_dir() and path.parent.name == 'acceptance':
        return path.parent.parent, path.name
    if path.is_dir():
        return path, None
    return path.parent, None


def protocol_identity(review):
    for parent in (review, *review.parents):
        for name in ('protocol.json', 'protocol.md'):
            path = parent / name
            if path.is_file():
                return {'path': str(path), 'sha256': digest(path)}
    return None


def raw_identity(data):
    identity = {key: data.get(key) for key in ('schema', 'evaluator_sha256',
        'randomization_level', 'envs_per_command', 'seconds', 'warmup',
        'deterministic', 'noise', *TRANSITION_FIELDS)}
    identity.update(commands=[row.get('command') for row in data.get('results', [])],
                    initial_commands=data.get('initial_commands'))
    return identity


def load_candidate(path):
    review, label = review_location(path)
    result = {'path': str(Path(path).resolve()), 'status': 'incomplete',
              'eligible': False, 'reviewable': False, 'reasons': []}
    summary_path = (review / 'long_evaluation_summary_v2.json'
                    if (review / 'long_evaluation_summary_v2.json').is_file()
                    else review / 'long_evaluation_summary.json')
    if not summary_path.is_file():
        result['reasons'].append('long_evaluation_summary.json missing; run the existing --long summarizer first')
        return result
    summary = json.loads(summary_path.read_text())
    if summary.get('schema') not in ('turn_lean_long_review_v1',
                                    'turn_lean_long_review_v2'):
        result['reasons'].append('unknown metric summary schema')
        return result
    labels = sorted({point['model'] for point in summary.get('points', [])})
    if label is None:
        if len(labels) != 1:
            result['reasons'].append('review contains multiple models; pass acceptance/MODEL')
            return result
        label = labels[0]
    result['model'] = label
    manifest_path = review / (label + '_commands.json')
    if not manifest_path.is_file():
        result['status'] = 'legacy_protocol_unknown'
        result['reasons'].append('review command manifest missing')
        return result
    manifest = json.loads(manifest_path.read_text())
    protocol = protocol_identity(review)
    result['protocol'] = protocol
    if protocol is None:
        result['status'] = 'legacy_protocol_unknown'
        result['reasons'].append('protocol identity missing')
        return result
    checkpoint = Path(manifest.get('checkpoint', ''))
    result['checkpoint'] = str(checkpoint)
    result['checkpoint_sha256'] = manifest.get('checkpoint_sha256')
    match = re.fullmatch(r'model_(\d+)\.pt', checkpoint.name)
    result['actual_iteration'] = int(match.group(1)) if match else None
    if not match or not checkpoint.is_file() or digest(checkpoint) != result['checkpoint_sha256']:
        result['reasons'].append('checkpoint path, iteration or SHA mismatch')
        return result
    spec_path = Path(manifest.get('spec', ''))
    if not spec_path.is_file():
        result['reasons'].append('review spec missing')
        return result
    spec = json.loads(spec_path.read_text())
    stage = spec.get('stage', {})
    height_skill = bool(stage.get('cohort_plan'))
    required_groups = GROUPS + (('height_skill', 'height_entry_exit') if height_skill else ())
    result['required_groups'] = list(required_groups)
    model_dir = review / 'acceptance' / label
    identities = {}
    missing = []
    completed_files = 0
    for group in required_groups:
        for seed in SEEDS:
            raw_path = model_dir / ('seed' + str(seed)) / (group + '.json')
            if not raw_path.is_file():
                missing.append(str(raw_path))
                continue
            completed_files += 1
            data = json.loads(raw_path.read_text())
            if (data.get('seed') != seed or data.get('checkpoint') != str(checkpoint)
                    or data.get('checkpoint_sha256') != result['checkpoint_sha256']
                    or not data.get('results') or any(not row.get('metrics')
                                                      for row in data['results'])):
                result['reasons'].append('missing or conflicting metadata/results: ' + str(raw_path))
                return result
            identity = raw_identity(data)
            if group in identities and identities[group] != identity:
                result['reasons'].append('evaluation protocol differs across seeds: ' + group)
                return result
            identities[group] = identity
    result['seeds'] = list(SEEDS)
    result['requested_files'] = len(required_groups) * len(SEEDS)
    result['completed_files'] = completed_files
    result['missing'] = missing
    if missing:
        result['reasons'].append('{} required group/seed files missing'.format(len(missing)))
        return result
    if stage.get('pairs'):
        expected = [(sv*v, sw*w) for v, w in stage['pairs']
                    for sv in (-1., 1.) for sw in (-1., 1.)]
        actual = [command[:2] for command in identities['turn_train']['commands']]
        if sorted(map(tuple, expected)) != sorted(map(tuple, actual)):
            result['reasons'].append('training turn grid differs from review spec')
            return result
    if height_skill:
        expected_height = [(vx, 0., height) for height in (.38, .36)
                           for vx in (0., -.5, .5)]
        for group in ('height_skill', 'height_entry_exit'):
            if sorted(map(tuple, identities[group]['commands'])) != sorted(expected_height):
                result['reasons'].append('independent height grid incomplete: ' + group)
                return result
    points = [point for point in summary['points'] if point['model'] == label]
    transitions = [row for row in summary.get('transitions', []) if row['model'] == label]
    point_names = tuple(group for group in required_groups
                        if group in ('retention','turn_train','turn_holdout','height_skill'))
    transition_names = tuple(group for group in required_groups
                             if group in ('entry_exit','slow_reverse','height_entry_exit'))
    point_groups = {group: [point for point in points if point['group'] == group]
                    for group in point_names}
    if (len(point_groups['retention']) != 25 or
            any(not point_groups[group] for group in point_names) or
            any(sorted(tuple(point['command']) for point in point_groups[group]) !=
                sorted(tuple(command) for command in identities[group]['commands'])
                for group in point_names) or
            any(point['status'] == 'unmeasured' for point in points) or
            any(point['status'] == 'passed' and point.get('failures', 0) > 0
                for point in points) or
            any(row['final_phase_passed'] and row.get('failure_count', 0) > 0
                for row in transitions) or
            len(transitions) != sum(len(identities[group]['commands']) * len(SEEDS)
                                    for group in transition_names)):
        result['reasons'].append('summary has missing commands, seeds or transitions')
        return result
    metric_sha = summary.get('metric_definition_sha256')
    gate_sha = summary.get('gate_sha256')
    if not metric_sha or not gate_sha:
        result['status'] = 'legacy_metric_unknown'
        result['reasons'].append('summary lacks metric/gate source identity; generate a new sidecar summary')
        return result
    identity = {'protocol_sha256': protocol['sha256'],
                'gate_sha256': gate_sha,
                'metric_definition_sha256': metric_sha,
                'summary_schema': summary['schema'], 'groups': identities}
    result['evaluation_signature'] = fingerprint(identity)
    result['metric_definition_sha256'] = identity['metric_definition_sha256']
    result['gate_sha256'] = identity['gate_sha256']
    result['evaluator_sha256'] = identities['turn_train']['evaluator_sha256']
    result['summary'] = str(summary_path)
    result['raw_review'] = str(model_dir)
    result['retention_passed'] = sum(point['status'] == 'passed'
                                      for point in point_groups['retention'])
    result['turn_train_passed'] = sum(point['status'] == 'passed'
                                       for point in point_groups['turn_train'])
    result['turn_train_total'] = len(point_groups['turn_train'])
    result['turn_holdout_passed'] = sum(point['status'] == 'passed'
                                         for point in point_groups['turn_holdout'])
    result['turn_holdout_total'] = len(point_groups['turn_holdout'])
    if height_skill:
        result['height_skill_passed'] = sum(point['status'] == 'passed'
                                            for point in point_groups['height_skill'])
        result['height_skill_total'] = len(point_groups['height_skill'])
    result['transition_passed'] = sum(row['final_phase_passed'] for row in transitions)
    result['transition_total'] = len(transitions)
    turns = point_groups['turn_train'] + point_groups['turn_holdout']
    result['turn_failures'] = sum(point['failures'] for point in turns)
    result['worst_yaw_mae'] = max(point['max_yaw_mae'] for point in turns)
    result['worst_slip_rms'] = max(point['max_slip_rms'] for point in turns)
    result['worst_height_mae'] = max(point['max_height_mae'] for point in turns)
    result['height_targets'] = sorted({point['command'][2] for point in turns})
    result['reviewable'] = True
    if not any(height < .4 for height in result['height_targets']):
        result['status'] = 'incompatible_fixed_height'
        result['reasons'].append('no lowered-height turn target')
        result['reviewable'] = False
    elif result['retention_passed'] != 25:
        result['status'] = 'retention_regression'
        result['reasons'].append('original 25-command regression failed')
    elif (result['turn_train_passed'] != result['turn_train_total'] or
          result['turn_holdout_passed'] != result['turn_holdout_total'] or
          (height_skill and result['height_skill_passed'] != result['height_skill_total']) or
          result['transition_passed'] != result['transition_total']):
        result['status'] = 'task_gate_failed'
        result['reasons'].append('turn, height skill or transition gate failed; no accepted policy')
    else:
        result['status'] = 'accepted_review'
        result['eligible'] = True
    return result


def dominates(left, right):
    higher = ('turn_train_passed', 'turn_holdout_passed')
    lower = ('turn_failures', 'worst_yaw_mae', 'worst_slip_rms', 'worst_height_mae')
    no_worse = all(left[key] >= right[key] for key in higher) and all(
        left[key] <= right[key] for key in lower)
    strictly_better = any(left[key] > right[key] for key in higher) or any(
        left[key] < right[key] for key in lower)
    return no_worse and strictly_better


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('candidates', nargs='+', type=Path,
                        help='completed review directory or acceptance/MODEL')
    parser.add_argument('--out', type=Path, required=True,
                        help='new JSON path; existing files are not overwritten')
    args = parser.parse_args(argv)
    rows = [load_candidate(path) for path in args.candidates]
    groups = {}
    for row in rows:
        if row['reviewable'] and row['retention_passed'] == 25:
            groups.setdefault(row['evaluation_signature'], []).append(row)
    comparable = [group for group in groups.values() if len(group) >= 2]
    pareto = [row for group in comparable for row in group
              if not any(other is not row and dominates(other, row) for other in group)]
    accepted = [row for row in pareto if row['eligible']]
    recommendation = accepted[0] if len(accepted) == 1 else None
    output = {'schema_version': SCHEMA, 'candidates': rows,
              'comparable_count': sum(map(len, comparable)),
              'pareto_candidates': [row['checkpoint'] for row in pareto],
              'recommendation': recommendation,
              'selection_rule': 'complete three-seed protocol and retention first; then Pareto on identical command sets; only an accepted unique candidate is recommended',
              'limitations': ['single-seed probes and missing groups cannot be accepted',
                              'different protocol/evaluator/gate/metric signatures are not compared',
                              'independent height skill requires separate review groups',
                              'holdout should remain reserved for final validation']}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as target:
        json.dump(output, target, indent=2)
        target.write('\n')
    print(json.dumps({'out': str(args.out), 'comparable_count': output['comparable_count'],
                      'recommendation': recommendation['checkpoint'] if recommendation else None}))


if __name__ == '__main__':
    main()
