import isaacgym
import torch
from wheel_legged_gym.domain.commands.height_commands import alternate_height


def test_switch_preserves_motion_and_retention():
    c=torch.tensor([[1.,-.5,.39],[-1.,.5,.41],[4.,0,.4],[0.,0,.39],[0.,0,.41]])
    before=c.clone()
    steps=torch.tensor([500,500,500,499,0])
    alternate_height(c,steps,.01,5.,.39,.41)
    assert torch.equal(c[:,:2],before[:,:2])
    assert torch.allclose(c[:,2],torch.tensor([.41,.39,.4,.39,.41]))
    alternate_height(c,torch.tensor([1000,1000,1000,499,0]),.01,5.,.39,.41)
    assert torch.equal(c,before)
