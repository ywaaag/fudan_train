import isaacgym
from wheel_legged_gym.envs.wheel_legged.wheel_legged_config import WheelLeggedCfg,WheelLeggedCfgPPO
from wheel_legged_gym.envs.wheel_legged.policy_experiments import apply_policy_experiment
from wheel_legged_gym.utils.helpers import class_to_dict


def test_legacy_command_change_preserves_effective_recipe():
    a,t=WheelLeggedCfg(),WheelLeggedCfgPPO();apply_policy_experiment(a,'LEGACY_URDF',t)
    b,u=WheelLeggedCfg(),WheelLeggedCfgPPO();m=apply_policy_experiment(b,'LEGACY_ANCHORS',u)
    for k in ['rewards','control','noise','domain_rand','asset','init_state']:
        assert class_to_dict(getattr(a,k))==class_to_dict(getattr(b,k)),k
    assert class_to_dict(t.algorithm)==class_to_dict(u.algorithm)
    assert not b.commands.curriculum and b.commands.hold_command_until_reset
    assert b.commands.ranges.lin_vel_x==[-1.,1.] and b.commands.ranges.ang_vel_yaw==[0.,0.]
    assert b.commands.translation_retention_anchors==(.5,1.)
    assert m['resume_mode']=='full'
