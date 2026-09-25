"""Audit the five requested capabilities without replacing missing tests with success."""
import hashlib,json,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--candidate',choices=['9500','10000'],default='9500')
    args=parser.parse_args()
    policy=ROOT/'plane/outputs/motion_goal_20260921_143444/accepted_basic_motion.onnx'
    sha=hashlib.sha256(policy.read_bytes()).hexdigest()
    cases=[('stop','closed_ramp_v2_20260921_202701/zero.json'),
           ('forward4','closed_ramp_v2_20260921_202701/vx_1_4.json'),
           ('reverse4','closed_ramp_v2_20260921_202701/vx_-1_4.json'),
           ('yaw_positive4','closed_yaw_boundary_20260922/slow30_yaw_4.json'),
           ('yaw_negative4','closed_ramp_v2_20260921_202701/yaw_-1_4.json')]
    if args.candidate=='10000':
        policy=ROOT/'plane/outputs/motion_goal_20260922_084744/accepted_basic_motion.onnx'
        sha=hashlib.sha256(policy.read_bytes()).hexdigest()
        cases=[('stop','closed_ramp_v2_20260922_085731/zero.json')]+[
            (name,'closed_target_trace_20260922_10000/'+file+'.json') for name,file in [
                ('forward4','vx_pos4'),('reverse4','vx_neg4'),('yaw_positive4','yaw_pos4'),
                ('yaw_negative4','yaw_neg4'),('stop_from_forward4','stop_from_forward4'),
                ('stop_from_reverse4','stop_from_reverse4')]]
    expected={'stop':[0,0,.4],'forward4':[4,0,.4],'reverse4':[-4,0,.4],
              'yaw_positive4':[0,4,.4],'yaw_negative4':[0,-4,.4],
              'stop_from_forward4':[0,0,.4],'stop_from_reverse4':[0,0,.4]}
    rows=[]
    for name,relative in cases:
        path=ROOT/'plane/outputs'/relative;data=json.loads(path.read_text())
        result=data['result'];m=data['measurements'];schedule=data['schedule']
        target=schedule.get('measurement_target',schedule['target']);v,w,h=target
        physical=bool(result['passed'] and result['completed_steps']==result['requested_steps'] and
                      not result['failure'] and not result['stopped_by_user'] and min(result['min_wheel_contacts'])>0
                      and result['max_base_contacts']==0 and result['max_nonwheel_ground_contacts']==0)
        measured=(m['samples']==20000 and result['metrics_samples']==20000 and schedule['steady_window'])
        checks={'same_policy':data['policy_sha256']==sha,'initialization':data['initialization']=='tree_zero',
                'correct_target':target==expected[name],
                'reset_height':abs(data['details']['initial_root_height']-.4)<1e-9,
                'fixed_step':result['physics_dt_s']==.001,
                'full_policy_only':result['mode']=='full' and not result['external_gas_spring_assist'] and not result['extra_guard_gate_trim'],
                'contract':result['observation_dim']==25 and result['history_dim']==125 and result['policy_rate_hz']==100,
                'physical':physical,'steady_20s':measured,
                'vx_tracking':m['vx_mae'] is not None and m['vx_mae']<=(.1 if v else .05),
                'yaw_tracking':m['yaw_mae'] is not None and m['yaw_mae']<=.1,
                'height_tracking':m['height_mae'] is not None and m['height_mae']<=.03}
        if name.startswith('stop_from_'):
            desired=4 if name=='stop_from_forward4' else -4
            cruise=[r for r in data['trace'] if 14<r['time']<34]
            checks['reached_speed_before_stop']=bool(cruise) and abs(sum(r['vx'] for r in cruise)/len(cruise)-desired)<=.1
            checks['explicit_return_protocol']=schedule['return_to_zero_after']==34 and schedule['return_ramp_seconds']==10
        rows.append({'capability':name,'source':str(path.relative_to(ROOT)),
                     'evidence_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                     'schedule':schedule,'checks':checks,'passed':all(checks.values()),
                     'vx_mae':m['vx_mae'],'yaw_mae':m['yaw_mae'],'failure':result['failure']})
    output={'policy':str(policy.relative_to(ROOT)),'sha256':sha,'capabilities':rows,
            'goal_complete':all(r['passed'] for r in rows),
            'limits':'Passes apply only to each recorded ramp-and-hold protocol, not arbitrary commands or real hardware. Mapping and all safety gates unchanged.'}
    if args.candidate=='10000':
        source=ROOT/'plane/outputs/motion_goal_20260922_084744/r01_m10000_summary.json'
        source_data=json.loads(source.read_text())
        output['source_acceptance']={'path':str(source.relative_to(ROOT)),'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'passed':source_data['passed'] and source_data['passed_count']==75 and source_data['total']==75 and
                source_data['geometry_passed'] and source_data['max_mean_mirror_distance_m']<=.03}
        output['goal_complete'] &= output['source_acceptance']['passed']
        output['step_response']='Four +/-4 direct steps failed contact protection; not an arbitrary-command or fast-response acceptance.'
    out=ROOT/f'docs/data/sim2sim_goal_{args.candidate}_20260922.json';out.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({'passed':[r['capability'] for r in rows if r['passed']],
                      'failed':[r['capability'] for r in rows if not r['passed']],
                      'goal_complete':output['goal_complete']},indent=2))


if __name__=='__main__':main()
