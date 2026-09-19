import isaacgym
import pytest
from wheel_legged_gym.envs.wheel_legged.wheel_legged_config import WheelLeggedCfg, WheelLeggedCfgPPO
from wheel_legged_gym.envs.wheel_legged.policy_experiments import apply_policy_experiment
from wheel_legged_gym.envs.wheel_legged.h3_low_speed import validate_source


def test_h3_migration_preserves_low_speed_objective():
    cfg, train = WheelLeggedCfg(), WheelLeggedCfgPPO()
    reference = apply_policy_experiment(cfg, 'LOW_SPEED', train)
    cfg, train = WheelLeggedCfg(), WheelLeggedCfgPPO()
    migration = apply_policy_experiment(cfg, 'H3_LOW_SPEED', train)
    assert reference['reward_scales'] == migration['reward_scales']
    assert reference['optimizer'] == migration['optimizer']
    assert cfg.commands.ranges.lin_vel_x == [-.5,.5]
    assert cfg.domain_rand_level == 1 and not cfg.domain_rand.push_robots
    assert cfg.rewards.tracking_linear_cap == 1.
    assert migration['resume_mode'] == 'policy'


@pytest.mark.parametrize('resume,mode', [(False,'policy'),(True,'full')])
def test_reject_wrong_migration_semantics(resume, mode):
    with pytest.raises(ValueError, match='resume_mode=policy'):
        validate_source('unused.pt', resume, mode)


def test_reject_unverified_checkpoint(tmp_path):
    path = tmp_path/'model_15800.pt'
    path.write_bytes(b'wrong checkpoint')
    with pytest.raises(ValueError, match='SHA256'):
        validate_source(path, True, 'policy')
