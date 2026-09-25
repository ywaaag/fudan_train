import isaacgym  # must precede torch
import torch
from wheel_legged_gym.domain.rewards.terms import bilateral_geometry_cost
from wheel_legged_gym.app.experiment_inputs import apply_training_profile
from wheel_legged_gym.contracts.wheel_legged_config import WheelLeggedCfg, WheelLeggedCfgPPO


def test_geometry_reflection_and_increasing_error():
    p = torch.tensor([[[-.1,-.2,-.1],[-.1,.2,-.1],[0.,-.22,-.34],[0.,.22,-.34]]])
    assert bilateral_geometry_cost(p).item() == 0
    p[:, 1, 0] += .003
    assert bilateral_geometry_cost(p).item() == 0
    p[:, 1, 0] += .03
    small = bilateral_geometry_cost(p)
    p[:, 1, 0] += .15
    assert bilateral_geometry_cost(p) > small
    mirrored = p[:, [1, 0, 3, 2]] * torch.tensor([1.,-1.,1.])
    torch.testing.assert_close(bilateral_geometry_cost(p), bilateral_geometry_cost(mirrored))


def test_geometry_term_only_in_stand():
    cfg, train = WheelLeggedCfg(), WheelLeggedCfgPPO()
    apply_training_profile(cfg, train, phase='stand', level=0)
    assert cfg.rewards.scales.stand_bilateral_geometry < 0
    assert cfg.rewards.scales.stand_still == 0
    apply_training_profile(cfg, train, phase='translate', level=0)
    assert cfg.rewards.scales.stand_bilateral_geometry == 0
