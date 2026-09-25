"""Historical standing continuation gates, independent of files and processes."""
def healthy(x):
    m = x['metrics']
    return (x['failure_count'] == 0 and x['timeout_count'] == 0
            and m['nonwheel_contact_fraction']['mean'] == 0
            and min(m[k]['mean'] for k in ('left_contact', 'right_contact')) >= .99
            and abs(m['height_m']['mean'] - .4) < .025
            and max(m[k]['mean_abs'] for k in ('roll_rad', 'pitch_rad')) < .10)


def assess_segment(audits, base_audits, current, base_stationary):
    """Return ordered rejection reasons and drift; thresholds retain original semantics."""
    reasons = []
    if len(audits) != 3 or {x['seed'] for x in audits} != {19, 37, 53}:
        reasons.append('missing push audits')
    for x in audits:
        if not healthy(x):
            reasons.append('posture/contact/failure gate seed ' + str(x['seed']))
        times = [t for push in x['push_results'] for t in push['recovery_seconds']]
        if len(times) != 160 or any(t is None or t > 2 for t in times):
            reasons.append('push recovery gate')
        for landmark in ('knee', 'wheel'):
            old = base_audits[x['seed']]['geometry_root_frame'][landmark]['mean_distance_m']
            if x['geometry_root_frame'][landmark]['mean_distance_m'] > old * 1.2 + .002:
                reasons.append('geometry regression ' + landmark)
    if not healthy(current):
        reasons.append('stationary stability gate')
    drift = current['position_drift']['final_displacement_mean_m']
    if drift > base_stationary['position_drift']['final_displacement_mean_m'] * 1.2 + .05:
        reasons.append('stationary drift regression')
    return reasons, drift
