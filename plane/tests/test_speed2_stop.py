import isaacgym
import torch
from wheel_legged_gym.envs.base.command_sampling import sample_method_v1
from wheel_legged_gym.envs.wheel_legged.wheel_legged_config import WheelLeggedCfg,WheelLeggedCfgPPO
from wheel_legged_gym.envs.wheel_legged.policy_experiments import apply_policy_experiment


def test_stop_retains_all_eight_endpoints_over_complete_cycles():
    cfg,train=WheelLeggedCfg(),WheelLeggedCfgPPO()
    m=apply_policy_experiment(cfg,'LEGACY_SPEED2_STOP',train)
    ranges=torch.tensor([[-2.,2.]]).repeat(400,1)
    for offset in [0,7,31,500]:
        x,y,_=sample_method_v1(ranges,torch.zeros(400,2),phase='translate',
            slot_ids=torch.arange(400)+offset,zero_retention=True,
            translation_anchors=cfg.commands.translation_retention_anchors)
        assert (x==0).sum()==160
        for v in [.5,1.,1.5,2.]:
            assert (x==v).sum()==30 and (x==-v).sum()==30
        assert not y.any()
    assert m['profile']=='legacy_speed2_stop_v2'
