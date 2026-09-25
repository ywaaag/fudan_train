import argparse,json,math,hashlib

from pathlib import Path

def summarize(data):
    points=data['command_sequence']['keyframes'];trace=data['trace'];result=data['result']
    physical=bool(result['passed'] and not result['failure'] and not result['stopped_by_user'] and
        result['completed_steps']==result['requested_steps'] and min(result['min_wheel_contacts'])>0 and
        result['max_nonwheel_ground_contacts']==0 and result['max_nonwheel_obstacle_contacts']==0)
    extended=points+[{'time':result['requested_steps']*result['physics_dt_s'],'command':points[-1]['command']}]
    rows=[]
    for i in range(1,len(extended)-1):
        p,after=extended[i:i+2]
        if p['command']!=after['command'] or p['command']==extended[i-1]['command']:continue
        start=extended[i-1]['time'];arrival=p['time'];end=after['time'];v,w,h=p['command']
        plateau=[t for t in trace if arrival<=t['time']<end]
        all_transition=[t for t in trace if start<=t['time']<end]
        tolerance=.05 if v==0 else .1
        settled=None
        # Report settlement only with >=.5s subsequent uninterrupted in-band samples.
        bad=[j for j,t in enumerate(plateau) if abs(t['vx']-v)>tolerance or abs(t['yaw']-w)>.1]
        first=bad[-1]+1 if bad else 0
        if physical and first<len(plateau) and plateau[-1]['time']-plateau[first]['time']>=.5:
            settled=plateau[first]['time']
        # Fixed .5s exclusion is reported, not silently counted as fast response.
        steady=[t for t in plateau if t['time']>=arrival+.5]
        mae=lambda key,target:sum(abs(t[key]-target) for t in steady)/len(steady) if steady else None
        distance=None
        if v==0 and w==0 and settled is not None:
            path=[t['base_position_world'] for t in all_transition if t['time']<=settled and 'base_position_world' in t]
            if len(path)>1:distance=sum(math.hypot(b[0]-a[0],b[1]-a[1]) for a,b in zip(path,path[1:]))
        direction=1 if v>extended[i-1]['command'][0] else -1
        row={'ramp_start':start,'ramp_end':arrival,'hold_end':end,'target':p['command'],
             'trace_samples':len(plateau),'steady_samples':len(steady),'vx_mae':mae('vx',v),'yaw_mae':mae('yaw',w),
             'settling_from_ramp_start_s':None if settled is None else settled-start,
             'settling_after_ramp_s':None if settled is None else settled-arrival,
             'directional_overshoot_m_s':max([0.]+[direction*(t['vx']-v) for t in all_transition]),
             'stop_path_to_settling_m':distance,
             'peak_abs_roll_rad':max([0.]+[abs(t['roll']) for t in all_transition]),
             'max_leg_difference_m':max([0.]+[abs(t['leg_lengths_m'][0]-t['leg_lengths_m'][1]) for t in all_transition])}
        row['height_mae']=sum(abs(t['base_position_world'][2]-h) for t in steady)/len(steady) if steady and all('base_position_world' in t for t in steady) else None
        row['tracking_passed']=bool(steady and row['vx_mae']<=tolerance and row['yaw_mae']<=.1 and settled is not None
            and row['height_mae'] is not None and row['height_mae']<=.03)
        rows.append(row)
    return {'physical_passed':physical,'plateaus':rows,'passed':physical and bool(rows) and all(r['tracking_passed'] for r in rows),
            'limits':'Only this input sequence; dynamics sampled at trace rate, protections checked every physics step. Settling measured from ramp start and end; no arbitrary latency gate. Steady excludes first .5s of each hold. Stop distance ends at observed settlement, not exactly zero velocity.'}

