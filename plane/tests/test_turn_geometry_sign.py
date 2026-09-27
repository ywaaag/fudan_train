"""Independent quaternion basis and geometric inward-direction checks."""
import math

import torch

from wheel_legged_gym.domain.geometry.rotations import _quat_apply, quat_rotate_inverse
from wheel_legged_gym.domain.rewards.turn_lean import orientation_cost
from tools.summarize_turn_envelope import geometry_sign_audit, require_geometry_direction


def test_known_roll_basis_gravity_evaluator_and_reward_agree():
    gravity_world = torch.tensor([[0.,0.,-1.]],dtype=torch.float64)
    for angle in (-.1,.1):
        q = torch.tensor([[math.sin(angle/2),0.,0.,math.cos(angle/2)]],dtype=torch.float64)
        axes = _quat_apply(q.repeat(3,1),torch.eye(3,dtype=torch.float64))
        assert torch.allclose(axes[0],torch.tensor([1.,0.,0.],dtype=torch.float64))
        assert abs(axes[1,2].item()-math.sin(angle)) < 1e-12
        assert abs(axes[2,1].item()+math.sin(angle)) < 1e-12
        projected = quat_rotate_inverse(q,gravity_world)
        measured = math.atan2(-projected[0,1].item(),-projected[0,2].item())
        assert abs(measured-angle) < 1e-12
    values=[]
    for angle in (-.1,0.,.1):
        q=torch.tensor([[math.sin(angle/2),0.,0.,math.cos(angle/2)]],dtype=torch.float64)
        projected=quat_rotate_inverse(q,gravity_world)
        values.append(orientation_cost(projected,torch.tensor([1.]),
                       torch.tensor([1.]),.14).item())
    assert values[0] < values[1] < values[2]
    for vx in (-1.,1.):
        for yaw in (-.5,.5):
            expected_roll = -math.copysign(1.,vx*yaw)
            angle = .08 * expected_roll
            q=torch.tensor([[math.sin(angle/2),0.,0.,math.cos(angle/2)]],dtype=torch.float64)
            up=_quat_apply(q,torch.tensor([[0.,0.,1.]],dtype=torch.float64))[0]
            inward_body_y=math.copysign(1.,vx*yaw)
            assert up[1].item()*inward_body_y > 0
            projected=quat_rotate_inverse(q,gravity_world)[0]
            assert projected[1].item()*inward_body_y > 0
            cost=orientation_cost(projected[None],torch.tensor([vx]),torch.tensor([yaw]),.14)
            opposite=orientation_cost(projected[None]*torch.tensor([[1.,-1.,1.]],dtype=torch.float64),
                                      torch.tensor([vx]),torch.tensor([yaw]),.14)
            assert cost.item() < opposite.item()


def test_geometry_audit_uses_velocity_turn_and_rejects_missing_contact():
    def frame(time, velocity, contact=True):
        return {'time':time,'velocity_world':[velocity],
            'wheel_contacts':[[contact,contact]],'resets':[0],
            'body_axes_world':[[[1.,0.,0.],[0.,1.,0.],[.0,.1,.995]]],
            'wheel_positions_world':[[[0.,.2,0.],[0.,-.2,0.]]],
            'com_position_world':[[0.,.05,.5]],
            'root_position_world':[[0.,0.,.4]],
            'projected_gravity':[[0.,.1,-.995]],
            'root_quaternion_xyzw':[[0.,0.,0.,1.]]}
    data={'checkpoint':'/tmp/model.pt','checkpoint_sha256':'sha','evaluator_sha256':'e',
          'seed':19,'envs_per_command':1,'warmup':0.,
          'results':[{'command':[1.,.5,.4],
                      'metrics':{'slip_rms':.01,'roll':-.1},
                      'per_env_metrics':{'slip_rms':[.01],'roll':[-.1]},
                      'failure_count':0}],
          'response_trace':[frame(0.,[1.,0.,0.]),frame(.1,[.99,.1,0.])]}
    case=geometry_sign_audit(data)['cases'][0]
    assert case['status']=='inward' and case['valid_windows']==1
    assert case['environments'][0]['inward_window_fraction']==1
    assert case['mean_com_support_inward_m'] > 0
    data['response_trace'][-1]['wheel_contacts']=[[False,True]]
    case=geometry_sign_audit(data)['cases'][0]
    assert case['status']=='unreliable' and case['environments'][0]['rejected_windows']['contact']==1


def test_new_acceptance_requires_independent_inward_geometry():
    for status, expected in [('inward',True),('outward',False),('unreliable',False)]:
        verdict=require_geometry_direction({'passed':True,'checks':{}},{'status':status})
        assert verdict['passed'] is expected
        assert verdict['checks']['independent_geometry_inward'] is expected
    case={'status':'inward','environments':[{'valid_windows':80,
        'rejected_windows':{'speed':20},'mean_up_dot_inward':.01}]}
    row={'per_env_metrics':{'roll_target_mae':[.019]}}
    assert require_geometry_direction({'passed':True,'checks':{}},case,row,
                                      'inward_stage1_v1')['passed']
    case['environments'][0]['mean_up_dot_inward']=.001
    verdict=require_geometry_direction({'passed':True,'checks':{}},case,row,
                                       'inward_stage1_v1')
    assert not verdict['passed'] and not verdict['checks']['inward_projection_0p5deg']
