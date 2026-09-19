"""Pool numerators by actual samples, not averages of episode/window means."""
import argparse
import json
from pathlib import Path


def summarize(path, window=100):
    records=[]
    with path.open() as stream:
        for line in stream:
            try:records.append(json.loads(line))
            except json.JSONDecodeError:break  # active writer may not have finished its last line
    records=records[-window:] if window else records
    if not records:raise ValueError('No completed diagnostic records')
    commands=sorted({r['command_vx'] for d in records for r in d['rows']})
    total=sum(r['samples'] for d in records for r in d['rows'])
    result={'iterations':[records[0]['iteration'],records[-1]['iteration']],
            'samples':total,'commands':[]}
    for command in commands:
        rows=[r for d in records for r in d['rows'] if r['command_vx']==command]
        count=sum(r['samples'] for r in rows)
        row={'command_vx':command,'samples':count,'env_seconds':sum(r['env_seconds'] for r in rows),
             'fraction':count/total if total else None,
             'failures':sum(r['failures'] for r in rows),'timeouts':sum(r['timeouts'] for r in rows)}
        for key in ['vx_mean','vx_mae','encoder_vx_bias','encoder_vx_mae','yaw_mae','height_mae']:
            row[key]=sum(r[key]*r['samples'] for r in rows if r['samples'])/count if count else None
        seconds=row['env_seconds']
        row['reward_per_env_second']={key:sum(r['reward_per_env_second'][key]*r['env_seconds']
            for r in rows if r['samples'])/seconds if seconds else None for key in rows[0]['reward_per_env_second']}
        result['commands'].append(row)
    shifts=[d['encoder_update_action_shift'] for d in records if d.get('encoder_update_action_shift')]
    if shifts:
        kl=sorted(d['implied_mean_policy_kl'] for d in shifts)
        result['encoder_update_action_shift']={'mean_abs_by_channel':[
            sum(s['mean_abs_by_channel'][i] for s in shifts)/len(shifts) for i in range(6)],
            'mean_implied_kl':sum(kl)/len(kl),'max_implied_kl':max(kl),
            'p95_implied_kl':kl[min(len(kl)-1,int(len(kl)*.95))]}
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('run',type=Path)
    p.add_argument('--window',type=int,default=100);p.add_argument('--out',type=Path)
    a=p.parse_args();result=summarize(a.run/'command_diagnostics.jsonl',a.window)
    if a.out:a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({**result,'commands':[{k:(round(v,6) if isinstance(v,float) else v)
        for k,v in r.items() if k!='reward_per_env_second'} for r in result['commands']]},indent=2))
