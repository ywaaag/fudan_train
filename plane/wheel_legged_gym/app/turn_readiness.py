"""Generate a turn-job sidecar report from existing structured summaries."""
import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text()) if path.is_file() else None


def latest_probe(job):
    paths = sorted(job.glob('probes/iter_*_summary.json'),
                   key=lambda path: int(path.stem.split('_')[1]))
    return (paths[-1], read(paths[-1])) if paths else (None, None)


def review_rows(job):
    found = []
    for review_dir in sorted((job.parent / 'phase_reviews').glob('*')):
        path = next((review_dir / name for name in (
            'long_evaluation_summary_v3.json',
            'long_evaluation_summary_v2.json',
            'long_evaluation_summary.json') if (review_dir / name).is_file()), None)
        if path is None:
            continue
        data = read(path)
        if data is None:
            continue
        commands = list(path.parent.glob('*_commands.json'))
        manifest = read(commands[0]) if len(commands) == 1 else {}
        requested_counts = {}
        for argv in manifest.get('commands', []):
            if '--out' not in argv or '--commands' not in argv or '--yaw-commands' not in argv:
                continue
            group = Path(argv[argv.index('--out')+1]).stem
            count = argv.index('--yaw-commands')-argv.index('--commands')-1
            requested_counts[group] = requested_counts.get(group,0)+count
        groups = {}
        for row in data.get('individual_rows', []):
            group = groups.setdefault(row['group'], {'passed': 0, 'total': 0,
                'seeds': set(), 'envs': set(), 'failures': [],
                'geometry_missing': False, 'strict_safety_missing': False,
                'strict_safety_failed': False})
            group['total'] += 1
            group['passed'] += bool(row.get('verdict', {}).get('passed'))
            group['seeds'].add(row.get('seed'))
            group['envs'].add(row.get('environments'))
            if row['group'] != 'retention' and row.get('geometry_direction_status') == 'legacy_unmeasured':
                group['geometry_missing'] = True
            safety_keys = ({'all_envs_contact', 'all_envs_slip'}
                if row['group'] == 'height_skill' else
                {'all_envs_wheel_contact', 'slip'} if row['group'] in
                ('turn_train', 'turn_holdout') else set())
            checks = row.get('verdict', {}).get('checks', {})
            if safety_keys and not safety_keys <= checks.keys():
                group['strict_safety_missing'] = True
            if safety_keys and any(checks.get(key) is False for key in safety_keys):
                group['strict_safety_failed'] = True
            if not row.get('verdict', {}).get('passed'):
                group['failures'].append({'command': row.get('command'),
                    'seed': row.get('seed'), 'checks': [key for key, value in
                    row.get('verdict', {}).get('checks', {}).items() if value is False]})
        for row in data.get('transitions', []):
            name = row['protocol']
            group = groups.setdefault(name, {'passed': 0, 'total': 0,
                'seeds': set(), 'envs': set(), 'failures': [],
                'geometry_missing': False, 'strict_safety_missing': False,
                'strict_safety_failed': False})
            group['total'] += 1
            group['passed'] += bool(row.get('final_phase_passed'))
            group['seeds'].add(row.get('seed'))
            group['envs'].add(row.get('environments'))
            if not row.get('final_phase_passed'):
                group['failures'].append({'command': row.get('command'),
                                          'seed': row.get('seed'), 'checks': ['recovery']})
        for group in groups.values():
            group['seeds'] = sorted(group['seeds'])
            group['envs'] = sorted(group['envs'])
        records = data.get('evaluation_records', [])
        identity_complete = (data.get('schema') == 'turn_lean_long_review_v3' and
            bool(records) and not data.get('missing_files') and
            data.get('checkpoint_sha256') == manifest.get('checkpoint_sha256') and
            data.get('protocol_id') not in (None, 'legacy_unknown') and
            len({record.get('evaluator_sha256') for record in records}) == 1 and
            all(record.get('evaluator_sha256') and
                record.get('checkpoint_sha256') == manifest.get('checkpoint_sha256')
                and record.get('completed') for record in records))
        found.append({'path': str(path), 'checkpoint': manifest.get('checkpoint'),
            'sha': manifest.get('checkpoint_sha256'),
            'protocol': data.get('protocol_id') or data.get('inward_protocol'),
            'gate': data.get('gate_id') or data.get('gate_sha256'),
            'metric': data.get('metric_definition_sha256'),
            'evaluator':records[0].get('evaluator_sha256') if identity_complete else None,
            'requested_counts':requested_counts, 'groups': groups,
            'identity_complete':identity_complete})
    return found


BASE_GROUPS = ('retention', 'turn_train', 'turn_holdout', 'entry_exit', 'slow_reverse')
HEIGHT_GROUPS = ('height_skill', 'height_entry_exit')


def required_groups(protocol):
    if protocol in ('inward_stage1_v1', 'aggressive_cornering_v2'):
        return BASE_GROUPS
    if protocol in ('cornering_height_skill_v1', 'variable_height_v1'):
        return BASE_GROUPS + HEIGHT_GROUPS
    return ()


def complete(review):
    required = required_groups(review.get('protocol'))
    groups = review.get('groups', {})
    return (bool(required) and review.get('identity_complete') is True and
            set(required) <= groups.keys() and
            bool(review.get('gate')) and bool(review.get('metric')) and
            bool(review.get('sha')) and all(
                groups[name]['seeds'] == [19, 37, 53] and
                groups[name]['envs'] == [16] and
                review.get('requested_counts', {}).get(name,0) > 0 and
                groups[name]['total'] == review['requested_counts'][name] and
                not groups[name].get('geometry_missing', False) and
                not groups[name].get('strict_safety_missing', False) and
                not groups[name].get('strict_safety_failed', False)
                for name in required))


def same_review_identity(reviews):
    return bool(reviews) and all(row.get('identity_complete') for row in reviews) and len({
        (row.get('protocol'),row.get('gate'),row.get('metric'),row.get('evaluator'))
        for row in reviews}) == 1


def build(job):
    status = read(job / 'status.json')
    if status is None:
        raise FileNotFoundError(job / 'status.json')
    probe_path, probe = latest_probe(job)
    reviews = review_rows(job)
    stage_spec = ((read(Path(status['run']) / 'policy_experiment.json') or {})
                  .get('turn_long_spec', {}).get('stage', {}) if status.get('run') else {})
    protocol = ('aggressive_cornering_v2' if stage_spec.get('aggressive_plan', {})
                .get('sampling_version') == 2 else
                'cornering_height_skill_v1' if stage_spec.get('cohort_plan') else
                'inward_stage1_v1' if status.get('stage') == 1 else
                next((row['protocol'] for row in reviews if row['protocol']), 'legacy_unknown'))
    required = required_groups(protocol)
    ledger = read(job.parent / 'budget_ledger.json')
    source = (read(job.parent / 'spec_stage2_height.json') or {}).get('source_iteration')
    prior = sum(item['actual_new_iterations'] for item in ledger['entries']) if ledger else None
    used = (prior + max(0, status['latest_iteration'] - source)
            if prior is not None and source is not None and status.get('latest_iteration') else None)
    budget = {'limit': ledger['total_limit'] if ledger else None, 'used': used,
              'remaining': ledger['total_limit'] - used if used is not None else None,
              'path': str(job.parent / 'budget_ledger.json') if ledger else None}
    base_ok = bool(probe and probe.get('retention_passed') == probe.get('retention_total'))
    inward_ok = bool(probe and probe.get('inward_geometry', {}).get('inward_quadrants') == 4)
    inward_measured = bool(probe and 'inward_geometry' in probe)
    turn_rows = ((probe or {}).get('mid_rows', []) + (probe or {}).get('high_rows', [])
                 or (probe or {}).get('turn_rows', []))
    contact_ok = bool(turn_rows and all(
        row.get('min_wheel_contact', 0) >= .99 and
        row.get('slip_rms', float('inf')) <= .10 and
        row.get('failure_count', 1) == 0 for row in turn_rows))
    height_rows = [row for row in (probe or {}).get('height_rows', [])
                   if row.get('command', [None, None, None])[2] == .38]
    height_ok = bool(height_rows and all(row.get('height_mae', 1) <= .015 and
        row.get('min_wheel_contact', 0) >= .99 and row.get('failure_count', 1) == 0
        for row in height_rows))
    protocol_reviews = [row for row in reviews if row['protocol'] == protocol]
    comparable = same_review_identity(protocol_reviews)
    accepted = [row for row in protocol_reviews if comparable and
        complete(row) and all(
        row['groups'][name]['passed'] == row['groups'][name]['total']
        for name in required_groups(row['protocol']))]
    matching = [row for row in reviews if row['protocol'] == protocol and
                probe and row['sha'] == probe.get('checkpoint_sha256')]
    complete_groups = sorted(name for name in required if any(
        name in row['groups'] and row['groups'][name]['seeds'] == [19,37,53] and
        row['groups'][name]['envs'] == [16] and
        row['groups'][name]['total'] == row.get('requested_counts',{}).get(name) and
        not row['groups'][name].get('geometry_missing', False) and
        not row['groups'][name].get('strict_safety_missing', False) and
        not row['groups'][name].get('strict_safety_failed', False)
        for row in matching))
    safety_reviews = [row for row in matching if complete(row) and
        row['groups']['retention']['passed'] == row['groups']['retention']['total'] and
        all(not row['groups'][name].get('strict_safety_failed', True)
            for name in ('turn_train', 'turn_holdout'))]
    height_reviews = [row for row in safety_reviews if
        'height_skill' in required and
        row['groups']['height_skill']['passed'] == row['groups']['height_skill']['total'] and
        not row['groups']['height_skill'].get('strict_safety_failed', True)]
    checkpoint_path = (probe or {}).get('checkpoint')
    actual_iteration = (int(Path(checkpoint_path).stem.split('_')[-1])
                        if checkpoint_path and Path(checkpoint_path).stem.split('_')[-1].isdigit()
                        else None)
    readiness = {'schema_version': 2, 'protocol_id': protocol,
        'gate_id': matching[0]['gate'] if matching else 'unknown',
        'metric_definition': matching[0]['metric'] if matching else 'unknown',
        'current_stage': status.get('stage'),
        'checkpoint': (probe or {}).get('checkpoint'),
        'checkpoint_iteration': actual_iteration,
        'checkpoint_sha256': (probe or {}).get('checkpoint_sha256'),
        'required_groups': list(required), 'completed_groups': complete_groups,
        'missing_groups': sorted(set(required)-set(complete_groups)),
        'decisions': {
            'low_speed_height_skill': 'ready' if base_ok and contact_ok and safety_reviews
                else 'insufficient_evidence',
            'height_to_turn': 'ready' if base_ok and inward_ok and height_ok and
                contact_ok and height_reviews
                else 'not_ready' if height_rows else 'insufficient_evidence',
            'higher_lateral_load': 'ready' if any(row in matching for row in accepted) and
                (HEIGHT_GROUPS[0] not in required or height_ok) else 'not_ready',
            'final_acceptance': 'ready' if any(row in matching for row in accepted)
                else 'insufficient_evidence'},
        'satisfied': [name for name, ok in [('basic_motion_probe', base_ok),
            ('inward_geometry_probe', inward_ok), ('turn_contact_probe', contact_ok),
            ('height_0p38_probe', height_ok)] if ok],
        'failed': [name for name, ok in [('basic_motion_probe', base_ok),
            ('turn_contact_probe', contact_ok), ('height_0p38_probe', height_ok)]
            if probe and not ok and (name != 'height_0p38_probe' or height_rows)]
            + (['inward_geometry_probe'] if inward_measured and not inward_ok else []),
        'missing': ([] if any(complete(row) for row in matching) else
            ['complete_three_seed_16_env_review']) +
            ([] if not protocol_reviews or comparable else
             ['same_protocol_gate_metric_evaluator']) +
            ([] if HEIGHT_GROUPS[0] not in required or height_rows else ['height_0p38_probe']) +
            ([] if inward_measured else ['inward_geometry_probe']),
        'evidence': {'probe': str(probe_path) if probe_path else None,
                     'reviews': [row['path'] for row in reviews]},
        'authorized_to_start_training': False,
        'recommendation': accepted[0]['checkpoint'] if len(accepted) == 1 else None}
    return status, probe, reviews, accepted, budget, readiness


def report(job, status, probe, reviews, accepted, budget, readiness):
    event = read(job / 'hapi_notification.json') or {}
    budget_text = (f"`{budget['used']}` / `{budget['limit']}` / `{budget['remaining']}` "
                   f"([ledger]({budget['path']}))" if budget['used'] is not None else
                   'unknown (budget ledger missing)')
    lines = ['# Turn training completion report', '',
        f"- event_id: `{event.get('event_id', 'unknown')}`",
        f"- status: `{status.get('status', 'unknown')}`",
        f"- stop_reason: `{status.get('stop_reason', 'unknown')}`",
        f"- error: `{status.get('error') or 'none'}`",
        f"- budget used/limit/remaining: {budget_text}",
        f"- last checkpoint: `{(probe or {}).get('checkpoint', 'unmeasured')}`",
        f"- actual iteration: `{status.get('latest_iteration', 'unmeasured')}`",
        f"- SHA256: `{(probe or {}).get('checkpoint_sha256', 'unmeasured')}`",
        f"- best fully accepted candidate: `{accepted[0]['checkpoint'] if len(accepted) == 1 else 'none'}`",
        '- best screening candidate: no same-protocol complete comparison unless listed below.', '',
        '## Reviews', '']
    if not reviews:
        lines.append('- unmeasured for the latest checkpoint')
    for item in reviews:
        lines.append(f"- [{Path(item['path']).parent.name}]({item['path']}): protocol "
            f"`{item['protocol'] or 'unknown'}`, complete three-seed `{complete(item)}`")
        for name, group in sorted(item['groups'].items()):
            lines.append(f"  - {name}: `{group['passed']}/{group['total']}` rows; seeds "
                f"`{group['seeds']}`, envs/command `{group['envs']}`; "
                f"failures `{group['failures'][:8]}`")
    if probe:
        lines.append(f"- latest probe: retention `{probe.get('retention_passed', 'unmeasured')}/"
            f"{probe.get('retention_total', 'unmeasured')}`, height "
            f"`{probe.get('height_skill_passed', 'unmeasured')}/"
            f"{probe.get('height_skill_total', 'unmeasured')}`, inward quadrants "
            f"`{probe.get('inward_geometry', {}).get('inward_quadrants', 'unmeasured')}`")
    lines += ['', '## Stage readiness', '',
        '- [stage_readiness.json](stage_readiness.json); evidence readiness does not authorize training.',
        '- Missing and partial reviews are not passes. Isaac tree results do not establish MuJoCo or real-robot acceptance.']
    return '\n'.join(lines) + '\n'


def main(root=None, argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True,
                        help='New sidecar directory; never overwrite a historical report')
    args = parser.parse_args(argv)
    job = args.job.resolve(strict=True)
    out = args.out_dir.resolve()
    if out.exists():
        raise FileExistsError(out)
    status, probe, reviews, accepted, budget, readiness = build(job)
    out.mkdir(parents=True)
    try:
        (out / 'stage_readiness.json').write_text(json.dumps(readiness, indent=2) + '\n')
        (out / 'completion_report.md').write_text(
            report(job, status, probe, reviews, accepted, budget, readiness))
    except Exception as exc:
        (out / 'report_error.json').write_text(json.dumps({'error': str(exc),
            'source_status': status.get('status')}, indent=2) + '\n')
        raise
    print(json.dumps({'report': str(out / 'completion_report.md'),
                      'readiness': str(out / 'stage_readiness.json')}))


if __name__ == '__main__':
    main()
