import isaacgym
import torch
from wheel_legged_gym.envs.wheel_legged.wheel_legged_config import WheelLeggedCfg,WheelLeggedCfgPPO
from wheel_legged_gym.envs.wheel_legged.policy_experiments import apply_policy_experiment
from wheel_legged_gym.envs.base.command_sampling import sample_method_v1
from wheel_legged_gym.utils.helpers import class_to_dict


def test_stage2_preserves_recipe_and_covers_retained_endpoints():
    c,t=WheelLeggedCfg(),WheelLeggedCfgPPO();apply_policy_experiment(c,'LEGACY_ANCHORS',t)
    d,u=WheelLeggedCfg(),WheelLeggedCfgPPO();apply_policy_experiment(d,'LEGACY_SPEED2',u)
    for k in ['rewards','noise','domain_rand','control','asset','init_state']:
        assert class_to_dict(getattr(c,k))==class_to_dict(getattr(d,k)),k
    assert class_to_dict(t.algorithm)==class_to_dict(u.algorithm)
    x,y,_=sample_method_v1(torch.tensor([[-2.,2.]]).repeat(400,1),torch.zeros(400,2),
        phase='translate',translation_anchors=d.commands.translation_retention_anchors)
    assert (x==0).sum()==80
    for v in [.5,1.,1.5,2.]:
        for s in [-1,1]:assert (x==v*s).sum()==30
    assert not y.any()
