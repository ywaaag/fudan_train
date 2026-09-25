"""Reject false passes from directional bias, contact loss and reset trajectories."""
from wheel_legged_gym.evaluation import gates as module


def valid_row(vx=.5):
    return {'command':[vx, 0, .4], 'failure_count':0, 'timeout_count':0,
            'metrics':{'vx_mae':.04, 'abs_yaw':.03, 'height_mae':.01,
                       'left_contact':1., 'right_contact':1., 'nonwheel_contact':0.,
                       'torque_saturation':0.}}


def test_stationary_requires_stricter_velocity_error():
    r = valid_row(0)
    r['metrics']['vx_mae'] = .08
    assert module.gate(r)['failed_checks'] == ['vx_mae']
    r['command'][0] = .5
    assert module.gate(r)['passed']


def test_wrong_direction_and_yaw_cannot_be_hidden_by_survival():
    r = valid_row(-.5)
    r['metrics'].update(vx_mae=.7, abs_yaw=.2)
    assert set(module.gate(r)['failed_checks']) == {'vx_mae','yaw_mae'}


def test_reset_or_contact_loss_fails_even_with_perfect_tracking():
    r = valid_row()
    r['failure_count'] = 1
    assert not module.gate(r)['passed']
    r['failure_count'], r['timeout_count'] = 0, 1
    assert module.gate(r)['failed_checks'] == ['timeouts']
    r['timeout_count'] = 0
    r['metrics']['left_contact'] = .98
    assert module.gate(r)['failed_checks'] == ['wheel_contact']


def test_nonfinite_or_saturated_metrics_fail():
    r = valid_row()
    r['metrics']['vx_mae'] = float('nan')
    assert not module.gate(r)['passed']
    r['metrics'].update(vx_mae=.01, torque_saturation=.1)
    assert module.gate(r)['failed_checks'] == ['torque_saturation']
