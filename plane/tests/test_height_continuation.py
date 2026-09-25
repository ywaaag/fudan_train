from wheel_legged_gym.app.continue_height_course import assess


def test_micro_constant_height_and_unsafe_candidates_rejected():
    row={'command':[0,0,.39],'failure_count':0,'timeout_count':0,
         'nonwheel_contact_full_fraction':0,'metrics':{'left_contact':1,'right_contact':1,
         'height_mae':.01,'torque_saturation':0,'vx_mae':0,'yaw_mae':0}}
    safe,score=assess([row],'micro')
    assert safe and score>1
    row['failure_count']=1
    assert not assess([row],'micro')[0]
