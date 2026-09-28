import json

import pytest

from wheel_legged_gym.app.turn_lean_review import command, reusable_result


def test_resume_reuses_complete_physical_failure_but_rejects_wrong_identity(tmp_path):
    output=tmp_path/'turn_train.json'
    spec={'experiment_id':'inward_cornering_r10200_v1','stage':{
        'speed_ramp_seconds':2.,'lean_max_deg':3.}}
    argv=command(tmp_path,tmp_path/'model_100.pt',output,[3.],[1.],[.4],19,4,spec=spec)
    data={'schema':'matched_policy_transition_v1','checkpoint':str(tmp_path/'model_100.pt'),
        'checkpoint_sha256':'source','evaluator_sha256':'evaluator',
        'seed':19,'envs_per_command':4,'randomization_level':1,
        'seconds':18.,'warmup':8.,'geometry_trace':True,'switch_at':1.,
        'transition_ramp_seconds':2.,'yaw_delay_seconds':2.,
        'height_delay_seconds':2.,'lean_reference_max_deg':3.,
        'yaw_exit_at':None,'yaw_exit_ramp_seconds':None,'yaw_exit_factor':None,
        'height_return_at_exit':False,'initial_commands':[[0.,0.,.4]],
        'response_trace':[{'time':0.}],
        'results':[{'command':[3.,1.,.4], 'metrics':{'yaw_mae':.2},
                    'per_env_metrics':{'yaw_mae':[.2]},'failure_count':4}]}
    output.write_text(json.dumps(data))
    assert reusable_result(output,argv,'source','evaluator')['results'][0]['failure_count']==4
    with pytest.raises(ValueError,match='identity'):
        reusable_result(output,argv,'different','evaluator')
    output.write_text('{broken')
    with pytest.raises(ValueError,match='unreadable'):
        reusable_result(output,argv,'source','evaluator')
