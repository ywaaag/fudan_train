"""Compact scalar summaries with historical slicing and rounding semantics."""
METRIC_KEYS = (
    'Train/mean_reward', 'Train/mean_episode_length', 'Loss/policy_symmetry',
    'Loss/policy_reference', 'Loss/policy_reference_fraction',
    'Loss/encoder_parameter_delta', 'Loss/learning_rate', 'Policy/mean_kl',
    'Episode/rew_base_height', 'Episode/rew_orientation',
    'Episode/tracking_abs_error', 'Episode/wheel_slip_rms_m_s',
    'Episode/rew_stand_bilateral_geometry', 'Episode/zero_abs_vx',
    'Episode/zero_abs_yaw_rate', 'Episode/zero_abs_wheel_speed',
    'Episode/wheel_contact_fraction', 'Episode/preclip_torque_saturation_fraction',
)


def through_iteration(scalars, checkpoint_iteration):
    """Keep only events written before the checkpoint after that iteration."""
    result = {}
    for tag, values in scalars.items():
        retained = [(step,value) for step,value in values if step < checkpoint_iteration]
        if retained:
            result[tag] = retained
    return result


def summarize_scalars(run, scalars, window=50):
    """scalars maps tags to ordered (step, value) samples; input is not modified."""
    result = {'run': str(run), 'metrics': {}}
    for key in METRIC_KEYS:
        if key in scalars:
            values = scalars[key][-window:]
            result['metrics'][key] = {
                'step': values[-1][0],
                'last': round(values[-1][1], 6),
                'tail_mean': round(sum(value for _, value in values) / len(values), 6),
            }
    return result
