import isaacgym
import torch
import pytest
from wheel_legged_gym.envs.base.command_sampling import sample_method_v1
from wheel_legged_gym.envs.wheel_legged.wheel_legged_config import WheelLeggedCfg, WheelLeggedCfgPPO
from wheel_legged_gym.envs.wheel_legged.policy_experiments import apply_policy_experiment
from wheel_legged_gym.envs.wheel_legged.h3_speed1 import validate_source


def test_balanced_retention_and_new_endpoints():
    x,yaw,_=sample_method_v1(torch.tensor([[-1.,1.]]).repeat(200,1),torch.zeros(200,2),
                            phase='translate',translation_anchors=(.5,1.))
    for v,n in [(0,40),(-.1,20),(.1,20),(-.5,30),(.5,30),(-1,30),(1,30)]:
        assert (x==v).sum()==n
    assert not yaw.any()
    x,_,_=sample_method_v1(torch.tensor([[-.5,.5]]).repeat(200,1),torch.zeros(200,2),
                          phase='translate',translation_anchors=(.5,1.))
    assert x.abs().max()==.5


def test_reward_and_optimizer_unchanged():
    a,t=WheelLeggedCfg(),WheelLeggedCfgPPO();old=apply_policy_experiment(a,'H3_LOW_SPEED',t)
    b,u=WheelLeggedCfg(),WheelLeggedCfgPPO();new=apply_policy_experiment(b,'H3_SPEED1',u)
    for k in ['reward_scales','reward_parameters','optimizer']:assert old[k]==new[k]
    assert b.commands.ranges.lin_vel_x==[-1.,1.] and b.commands.method_v1_level==1


def test_policy_only_resume_rejected():
    with pytest.raises(ValueError,match='resume_mode=full'):
        validate_source('unused',True,'policy',{})
