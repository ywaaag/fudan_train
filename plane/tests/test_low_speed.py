import isaacgym
import torch
from wheel_legged_gym.envs.wheel_legged.wheel_legged_config import WheelLeggedCfg, WheelLeggedCfgPPO
from wheel_legged_gym.envs.wheel_legged.policy_experiments import apply_policy_experiment
from wheel_legged_gym.envs.base.command_sampling import sample_method_v1


def test_low_speed_contract_and_sampler():
    cfg, train = WheelLeggedCfg(), WheelLeggedCfgPPO()
    manifest = apply_policy_experiment(cfg, 'LOW_SPEED', train)
    assert cfg.commands.ranges.lin_vel_x == [-.5,.5]
    assert cfg.commands.ranges.ang_vel_yaw == [0.,0.]
    assert cfg.rewards.scales.stand_bilateral_geometry == 0
    assert cfg.rewards.scales.stand_still == 0
    assert cfg.domain_rand_level == 1
    assert manifest['optimizer']['extra_learning_rate'] == train.algorithm.extra_learning_rate == 1e-5
    assert cfg.env.num_observations == 25 and cfg.env.num_actions == 6
    x, yaw, _ = sample_method_v1(torch.tensor([[-.5,.5]]).repeat(100,1),
                                torch.zeros(100,2), phase='translate')
    assert (x==0).sum()==20 and (x==-.5).sum()==30 and (x==.5).sum()==30
    assert (x.abs()<.11).sum()==40 and not yaw.any()


def test_tracking_ablation_changes_only_error_scale():
    from wheel_legged_gym.envs.base.reward_terms import capped_tracking_terms
    a, ta = WheelLeggedCfg(), WheelLeggedCfgPPO()
    b, tb = WheelLeggedCfg(), WheelLeggedCfgPPO()
    ma=apply_policy_experiment(a,'LOW_SPEED',ta)
    mb=apply_policy_experiment(b,'LOW_SPEED_TRACKING',tb)
    assert ma['reward_scales']==mb['reward_scales']
    assert ma['optimizer']==mb['optimizer']
    assert a.rewards.tracking_linear_cap==1.
    assert b.rewards.tracking_linear_cap==.5
    assert a.commands.ranges.lin_vel_x==b.commands.ranges.lin_vel_x
    bad=capped_tracking_terms(torch.tensor([.5,-.5]),.5)
    better=capped_tracking_terms(torch.tensor([.2,-.2]),.5)
    assert (better['coarse']>bad['coarse']).all()
    assert (better['gap']<bad['gap']).all()
