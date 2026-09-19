import isaacgym
from wheel_legged_gym.envs.wheel_legged.wheel_legged_config import WheelLeggedCfg, WheelLeggedCfgPPO
from wheel_legged_gym.envs.wheel_legged.policy_experiments import apply_policy_experiment


def test_legacy_urdf_profile_keeps_current_asset_contract():
    cfg, train = WheelLeggedCfg(), WheelLeggedCfgPPO()
    manifest = apply_policy_experiment(cfg, 'LEGACY_URDF', train)
    assert manifest['physics_equivalence'] is False
    assert cfg.asset.file.endswith('assets/wheel_leg_train.urdf')
    assert cfg.init_state.pos[2] == .4
    assert cfg.commands.sampling_strategy == 'uniform'
    assert cfg.rewards.reward_pipeline == 'legacy_v0'
    assert cfg.rewards.scales.tracking_lin_vel == 1.0
    assert train.algorithm.learning_rate == 1e-3
    assert train.algorithm.extra_learning_rate == 1e-3
    assert train.algorithm.symmetry_loss_coef == 0.0
