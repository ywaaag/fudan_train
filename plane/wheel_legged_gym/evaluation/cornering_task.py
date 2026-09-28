"""Versioned movement gate for autonomous inward lean in turns."""
from wheel_legged_gym.evaluation.gates import gate


PROTOCOL_ID = 'cornering_task_v2'
ROLL_BOUND_RAD = .25


def turn_verdict(row, geometry_case=None):
    metrics = row['metrics']
    per_env = row['per_env_metrics']
    active_lean = abs(metrics['roll_target']) >= .017453292519943295
    checks = {
        'original_gate': gate(row)['passed'],
        'all_envs_survive': row['survival_fraction'] == 1 and row['timeout_count'] == 0,
        'all_envs_track_vx': max(per_env['vx_mae']) <= .10,
        'all_envs_track_yaw': max(per_env['yaw_mae']) <= .05,
        'all_envs_height_mae': max(per_env['height_mae']) <= .03,
        'all_envs_wheel_contact': min(per_env['left_contact']) >= .99 and
                                  min(per_env['right_contact']) >= .99,
        'height_reached': max(per_env['height']) <= metrics['height_target'] + .015,
        'roll_absolute_bound': row['max_abs_metrics']['abs_roll'] <= ROLL_BOUND_RAD,
        'pitch': max(per_env['abs_pitch']) <= .10,
        'lateral_velocity': max(per_env['abs_lateral_velocity']) <= .10,
        'slip': max(per_env['slip_rms']) <= .10,
        'soft_joint_margin': min(per_env['soft_joint_limit_margin']) >= 0.,
    }
    if active_lean:
        checks['independent_geometry_inward'] = (geometry_case is not None and
            geometry_case.get('status') == 'inward')
    return {'passed': all(checks.values()), 'checks': checks,
        'active_lean_required': active_lean,
        'reference_diagnostics': {
            'roll_target_mae_rad': metrics['roll_target_mae'],
            'excess_over_reference_rad': metrics['roll_overshoot'],
            'roll_target_rad': metrics['roll_target'],
            'mean_roll_rad': metrics['roll'],
            'peak_abs_roll_rad': row['max_abs_metrics']['abs_roll'],
        }}


def transition_verdict(row, raw_row):
    final = row['phases']['recovery' if row['protocol'] == 'entry_exit'
                          else 'reverse_hold']
    checks = {'no_failure': row['failure_count'] == 0,
        'no_timeout': raw_row['timeout_count'] == 0,
        'nonwheel_contact': raw_row['metrics']['nonwheel_contact'] == 0,
        'torque_saturation': raw_row['metrics']['torque_saturation'] <= .01,
        'vx_recovery': final['vx_mae'] <= .10,
        'yaw_recovery': final['yaw_mae'] <= .10,
        'height_recovery': final['height_mae'] <= .03,
        'contact_recovery': final['wheel_contact'] >= .99,
        'roll_to_level_recovery': final['roll_target_mae'] <= .035}
    return {'passed': all(checks.values()), 'checks': checks}
