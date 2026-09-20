import isaacgym
import torch
from wheel_legged_gym.envs.base.command_sampling import sample_method_v1
from wheel_legged_gym.envs.wheel_legged.wheel_legged_config import WheelLeggedCfg,WheelLeggedCfgPPO
from wheel_legged_gym.envs.wheel_legged.policy_experiments import apply_policy_experiment


def test_zero_slots_change_without_endpoint_loss():
    ranges=torch.tensor([[-1.,1.]]).repeat(200,1)
    x,y,_=sample_method_v1(ranges,torch.zeros(200,2),phase='translate',zero_retention=True)
    assert (x==0).sum()==80
    for v in [-1,-.5,.5,1]:assert (x==v).sum()==30
    assert not y.any()


def test_rewards_and_optimizer_retained():
    c,t=WheelLeggedCfg(),WheelLeggedCfgPPO();a=apply_policy_experiment(c,'ANCHORED_WHEEL_EXPLORE',t)
    c,t=WheelLeggedCfg(),WheelLeggedCfgPPO();b=apply_policy_experiment(c,'EXPLORE_STOP_RETENTION',t)
    for k in ['reward_scales','reward_parameters','optimizer','encoder_action_anchor_coef']:
        assert a[k]==b[k]
    assert c.commands.zero_retention
