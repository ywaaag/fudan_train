"""Shared absolute acceptance thresholds; no processes or simulator imports."""
def gate(row):
    """Same absolute tracking thresholds at every speed, for every candidate."""
    m = row['metrics']
    tests = {
        'failures': row['failure_count'] == 0,
        'timeouts': row['timeout_count'] == 0,
        'nonwheel_contact': m['nonwheel_contact'] == 0,
        'wheel_contact': min(m['left_contact'], m['right_contact']) >= .99,
        'height_mae': m['height_mae'] <= .03,
        'vx_mae': m['vx_mae'] <= (.05 if row['command'][0] == 0 else .10),
        'yaw_mae': (m['yaw_mae'] if row['command'][1] != 0 else m['abs_yaw']) <= .10,
        # Explicit operational threshold; not a historical training gate.
        'torque_saturation': m['torque_saturation'] <= .01,
    }
    return {'passed': all(tests.values()), 'failed_checks': [k for k,v in tests.items() if not v]}

