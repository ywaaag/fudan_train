import isaacgym
import torch
from types import SimpleNamespace
from wheel_legged_gym.envs.base.legged_robot import LeggedRobot
from wheel_legged_gym.envs.wheel_legged.wheel_legged_config import WheelLeggedCfg, WheelLeggedCfgPPO
from wheel_legged_gym.envs.wheel_legged.policy_experiments import apply_policy_experiment


def test_reference_profile_isolated_and_standing():
    cfg, train = WheelLeggedCfg(), WheelLeggedCfgPPO()
    manifest = apply_policy_experiment(cfg, 'FUDAN_STAND', train)
    assert cfg.commands.ranges.lin_vel_x == [0., 0.]
    assert cfg.commands.ranges.ang_vel_yaw == [0., 0.]
    assert cfg.commands.ranges.height == [.4, .4]
    assert cfg.rewards.reward_pipeline == 'legacy_v0'
    assert cfg.rewards.scales.orientation == -500
    assert cfg.rewards.scales.base_height == 2
    assert cfg.rewards.scales.nominal_state == -1
    assert cfg.rewards.scales.zero_wheel_velocity == -1
    assert cfg.rewards.scales.stand_bilateral_geometry == 0
    assert cfg.rewards.scales.track_vx_coarse == 0
    assert train.algorithm.symmetry_loss_coef == 0
    assert train.algorithm.extra_learning_rate == 1e-3
    assert manifest['optimizer']['learning_rate'] == train.algorithm.learning_rate
    assert cfg.env.num_observations == 25 and cfg.env.num_actions == 6
    assert cfg.control.decimation == 2


def test_virtual_leg_angle_uses_actual_mirrored_geometry():
    env = LeggedRobot.__new__(LeggedRobot)
    env.cfg = SimpleNamespace(commands=SimpleNamespace(training_profile='fudan_stand_v1'))
    env.fudan_leg_landmarks = torch.arange(4)
    env.base_quat = torch.tensor([[0., 0., 0., 1.]])
    env.rigid_body_states = torch.zeros(1, 4, 13)
    env.rigid_body_states[0, :, :3] = torch.tensor([[0.,-.17,0.], [0.,.17,0.],
                                                                [.04,-.22,-.34],[.04,.22,-.34]])
    assert env._reward_nominal_state().item() == 0
    env.rigid_body_states[0, 3, 0] += .05
    assert env._reward_nominal_state().item() > 0
