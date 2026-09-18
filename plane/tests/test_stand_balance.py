import isaacgym
from wheel_legged_gym.envs.wheel_legged.wheel_legged_config import WheelLeggedCfg, WheelLeggedCfgPPO
from wheel_legged_gym.envs.wheel_legged.policy_experiments import apply_policy_experiment


def test_control_and_treatment_differ_only_in_geometry_reward():
    results = []
    for name in ('STAND_CONTROL', 'STAND_SYMMETRIC'):
        cfg, train = WheelLeggedCfg(), WheelLeggedCfgPPO()
        m = apply_policy_experiment(cfg, name, train)
        assert cfg.rewards.scales.stand_still == -.2
        assert train.algorithm.extra_learning_rate == 0.
        assert train.algorithm.schedule == 'fixed'
        assert cfg.commands.ranges.lin_vel_x == [0., 0.]
        assert cfg.commands.ranges.height == [.4, .4]
        results.append(m)
    a, b = results
    assert a['optimizer'] == b['optimizer']
    differences = [k for k in a['reward_scales'] if a['reward_scales'][k] != b['reward_scales'][k]]
    assert differences == ['stand_bilateral_geometry']
