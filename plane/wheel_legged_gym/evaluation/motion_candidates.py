"""Motion candidate eligibility, score and posture retention thresholds."""
from wheel_legged_gym.evaluation.gates import gate


def assess(records):
    """Safety is a hard eligibility condition; tracking chooses same-stage candidates."""
    rows = [r for d in records for r in d['results']]
    safe = all(r['failure_count']==0 and r['timeout_count']==0 and
        r['nonwheel_contact_full_fraction']==0 and
        min(r['metrics']['left_contact'],r['metrics']['right_contact'])>=.99 and
        r['metrics']['height_mae']<=.03 and r['metrics']['torque_saturation']<=.01 for r in rows)
    ratios=[]
    for r in rows:
        m=r['metrics']
        ratios += [m['vx_mae']/(.05 if r['command'][0]==0 else .10), m['yaw_mae']/.10]
    score=max(ratios)+sum(max(0.,v-1) for v in ratios)/len(ratios)
    return {'safe':safe,'passed':safe and all(gate(r)['passed'] for r in rows),
            'passed_count':sum(gate(r)['passed'] for r in rows),'total':len(rows),'score':score,
            'failed_commands':sorted({tuple(r['command'][:2]) for r in rows if not gate(r)['passed']})}


def geometry_retained(current, reference, high_speed_floor=.03):
    """Preserve low-speed 3cm quality and prevent >5mm regression elsewhere."""
    def limit(key):
        floor=high_speed_floor if abs(float(key))>=2. else .03
        return floor if reference[key]<=floor else reference[key]+.005
    return all(key in reference and value<=limit(key)
               for key,value in current.items()) and set(current)==set(reference)


