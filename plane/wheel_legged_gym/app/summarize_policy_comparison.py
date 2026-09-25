"""Render a compact matched-policy comparison report."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from wheel_legged_gym.evaluation.gates import gate


def main(argv=None):
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('job', type=Path)
    a = p.parse_args(argv)
    manifest = json.loads((a.job/'manifest.json').read_text())
    status = json.loads((a.job/'status.json').read_text())
    baselines, rows, violations = {}, [], []
    for tag in status['completed']:
        data = json.loads((a.job/(tag+'.json')).read_text())
        name = tag.split('_rand')[0]
        key = (data['randomization_level'], data['seed'], tuple(r['command'][0] for r in data['results']))
        # Identical actor-independent initial conditions are required, not
        # merely the same seed printed in two reports.
        matched = {k:data[k] for k in ['initial_root_states','initial_observations','configuration',
                                       'dof_names','torque_limits','policy_dt','physics_dt']}
        if key in baselines and baselines[key] != matched:
            violations.append(tag)
        baselines.setdefault(key,matched)
        for result in data['results']:
            outcome = gate(result)
            rows.append(dict(candidate=name, randomization=data['randomization_level'],seed=data['seed'],
                command_vx=result['command'][0], **result['metrics'],
                failure_count=result['failure_count'],timeout_count=result['timeout_count'],
                passed=outcome['passed'],failed_checks=','.join(outcome['failed_checks'])))
    if not rows:
        raise RuntimeError('No completed grids')
    with (a.job/'comparison.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r['randomization'],r['candidate'],r['command_vx'])].append(r)
    lines = ['# Matched policy comparison', '',
             f"Status: {status['status']}; completed grids: {len(status['completed'])}.",
             f"Matched initial-state/configuration mismatches: {len(violations)}.", '',
             '25 s per grid, 5 s warmup; 16 environments per command; seeds 19/37/53 by default.',
             'Deterministic actor, common current tree physics and contact termination, no pushes.',
             'Randomization 0 and 1 are separate regimes. Random initial velocities remain in both.',
             'Averages below are descriptive; every seed must pass independently.',
             'Acceptance: vx MAE <=0.05 at zero, <=0.10 otherwise; yaw MAE <=0.10;',
             'height MAE <=0.03; each wheel contact >=0.99; no failure/timeout/nonwheel contact;',
             'preclip torque saturation <=0.01. Geometry is diagnostic, not a motion gate.', '',
             '| Rand | Candidate | Cmd vx | Actual vx | vx MAE | abs yaw | Height MAE | Wheel contact min | Failures | Seeds passed |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for (rand,name,vx), group in sorted(grouped.items()):
        avg = lambda k:sum(r[k] for r in group)/len(group)
        contact = min(min(r['left_contact'],r['right_contact']) for r in group)
        passed = sum(r['passed'] for r in group)
        lines.append(f'| {rand} | {name} | {vx:+.1f} | {avg("vx"):+.4f} | {avg("vx_mae"):.4f} | '
                     f'{avg("abs_yaw"):.4f} | {avg("height_mae"):.4f} | {contact:.4f} | '
                     f'{sum(r["failure_count"] for r in group):g} | {passed}/{len(group)} |')
    lines.extend(['','## Candidate provenance',''])
    for name,c in manifest['candidates'].items():
        lines.append(f'- {name}: `{c["checkpoint"]}`; SHA256 `{c["sha256"]}`')
    lines.extend(['', '## Limits', '',
        '- Bounded candidate selection, not an exhaustive checkpoint search or causal refactor ablation.',
        '- Historical profiles are not replayed: all policies face the same current evaluation environment.',
        '- Higher speeds only tested after every prerequisite command/seed passed in that regime.',
        '- No yaw-command, command-transition, ONNX or closed-chain sim2sim acceptance in this experiment.',
        '- Post-warmup metrics include reset trajectories if any; full-episode failures are counted separately.',
        '- Mean performance does not guarantee every individual environment tracks within the same bound.',
        '- Source configuration comparison covers archived literal control/reset/normalization/simulation/asset fields; it is not a complete historical runtime reconstruction.', ''])
    (a.job/'comparison.md').write_text('\n'.join(lines))
    (a.job/'matching_check.json').write_text(json.dumps({'checked_grids':len(status['completed']),
        'matched_groups':len(baselines),'mismatches':violations},indent=2)+'\n')
    print(json.dumps({'completed_grids':len(status['completed']),'rows':len(rows),'mismatches':violations}))
    if violations:
        raise RuntimeError('Initial conditions/configurations do not match; comparison invalid')


if __name__ == '__main__':
    main()
