from wheel_legged_gym.evaluation.cornering_task import turn_verdict


def row():
    metrics = {'roll_target':-.052, 'roll':-.105, 'roll_target_mae':.053,
        'roll_overshoot':.053, 'height_target':.4, 'nonwheel_contact':0,
        'left_contact':1., 'right_contact':1., 'height_mae':0.,
        'vx_mae':0., 'yaw_mae':0., 'torque_saturation':0.}
    per_env = {name:[value] for name,value in {
        'vx_mae':0.,'yaw_mae':0.,'height_mae':0.,'height':.4,
        'left_contact':1.,'right_contact':1.,'abs_pitch':0.,
        'abs_lateral_velocity':0.,'slip_rms':0.,
        'soft_joint_limit_margin':.1}.items()}
    return {'command':[3.,1.,.4],'metrics':metrics,'per_env_metrics':per_env,
        'max_abs_metrics':{'abs_roll':.11},'failure_count':0,'timeout_count':0,
        'survival_fraction':1.}


def test_reference_excess_is_diagnostic_but_inward_and_motion_are_required():
    source = row()
    assert turn_verdict(source, {'status':'inward'})['passed']
    assert not turn_verdict(source, {'status':'outward'})['passed']
    source['per_env_metrics']['yaw_mae'] = [.06]
    assert not turn_verdict(source, {'status':'inward'})['passed']


def test_absolute_roll_bound_and_safety_are_not_relaxed():
    source = row()
    source['max_abs_metrics']['abs_roll'] = .251
    assert not turn_verdict(source, {'status':'inward'})['passed']
    source['max_abs_metrics']['abs_roll'] = .11
    source['per_env_metrics']['slip_rms'] = [.11]
    assert not turn_verdict(source, {'status':'inward'})['passed']
