"""Summarize measured 10 Hz step responses without treating steady gates as smoothness gates."""
import argparse
import json
from pathlib import Path
import numpy as np


def response_metrics(times, values, target, initial, switch_at, tolerance, dwell=.5):
    times, values = np.asarray(times), np.asarray(values)
    mask = times >= switch_at
    t, x = times[mask] - switch_at, values[mask]
    in_band = np.abs(x - target) <= tolerance
    first_dwell = None
    for i in range(len(t)):
        end = np.searchsorted(t, t[i] + dwell - 1e-6)
        if end < len(t) and in_band[i:end+1].all():
            first_dwell = float(t[i])
            break
    # Settling requires staying in band to the end, with at least dwell remaining.
    last_bad = np.flatnonzero(~in_band)
    index = int(last_bad[-1]+1) if len(last_bad) else 0
    settling = float(t[index]) if index < len(t) and t[-1]-t[index] >= dwell else None
    direction = np.sign(target-initial)
    return {'first_sustained_band_s': first_dwell, 'settling_time_s': settling,
            'directional_overshoot': float(max(0., np.max(direction*(x-target)))) if direction else None,
            'peak_abs_error': float(np.max(np.abs(x-target))), 'tolerance': tolerance}


def summarize(data):
    traces = data['response_trace']
    times = [r['time'] for r in traces]
    count = data['envs_per_command']
    rows = []
    for i, (initial, result) in enumerate(zip(data['initial_commands'], data['results'])):
        axes = {}
        for col, name in enumerate(('vx','yaw')):
            target = result['command'][col]
            tolerance = (.05 if target == 0 else .10) if name == 'vx' else .10
            outcomes = [response_metrics(times,[t[name][e] for t in traces],target,initial[col],
                        data['switch_at'],tolerance) for e in range(i*count,(i+1)*count)]
            settled = [r['settling_time_s'] for r in outcomes if r['settling_time_s'] is not None]
            axes[name] = {'settled_envs':len(settled), 'total_envs':count,
                          'worst_settling_s':max(settled) if settled else None,
                          'per_env':outcomes}
        rows.append({'initial':initial,'target':result['command'],'response':axes,
                     'failure_count':result['failure_count'],'timeout_count':result['timeout_count']})
    return {'checkpoint':data['checkpoint'],'seed':data['seed'],'rows':rows,
            'limitations':'10 Hz trace; timing resolution 0.1 s. No smoothness acceptance threshold or acceleration/jerk acceptance. Settling relative to requested command, not biased final speed.'}


if __name__ == '__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('input',type=Path)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    args.out.write_text(json.dumps(summarize(json.loads(args.input.read_text())),indent=2)+'\n')
