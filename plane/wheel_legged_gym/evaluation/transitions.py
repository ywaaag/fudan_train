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
        names = ('vx','yaw','height') if 'height' in traces[0] else ('vx','yaw')
        for col, name in enumerate(names):
            target = result['command'][col]
            tolerance = (.05 if target == 0 else .10) if name == 'vx' else .10
            if name == 'height': tolerance = .015
            outcomes = [response_metrics(times,[t[name][e] for t in traces],target,initial[col],
                        data['switch_at'],tolerance) for e in range(i*count,(i+1)*count)]
            settled = [r['settling_time_s'] for r in outcomes if r['settling_time_s'] is not None]
            axes[name] = {'settled_envs':len(settled), 'total_envs':count,
                          'worst_settling_s':max(settled) if settled else None,
                          'per_env':outcomes}
        rows.append({'initial':initial,'target':result['command'],'response':axes,
                     'failure_count':result['failure_count'],'timeout_count':result['timeout_count']})
        if 'path_since_switch_m' in traces[0]:
            post=[t for t in traces if t['time']>=data['switch_at']]
            rows[-1]['transient']={
                'max_abs_roll':max(abs(t['roll'][e]) for t in post for e in range(i*count,(i+1)*count)),
                'wheel_contact_fraction':sum(all(t['wheel_contacts'][e]) for t in post for e in range(i*count,(i+1)*count))/(len(post)*count)}
            if initial[0]!=0 and result['command'][:2]==[0,0]:
                distances=[]
                for e,outcome in zip(range(i*count,(i+1)*count),axes['vx']['per_env']):
                    settle=outcome['settling_time_s']
                    valid=result['failure_count']==0 and result['timeout_count']==0
                    sample=next((t for t in post if settle is not None and t['time']-data['switch_at']>=settle-1e-6),None)
                    distances.append(sample['path_since_switch_m'][e] if valid and sample else None)
                rows[-1]['stop_distance_m']=distances
    return {'checkpoint':data['checkpoint'],'seed':data['seed'],'rows':rows,
            'trace_dt':data.get('trace_dt',.1),
            'limitations':'Settling relative to requested command. Stop distance is planar path to settling, invalid on any reset. No jerk certification; sampled contacts do not cover every physics substep.'}

