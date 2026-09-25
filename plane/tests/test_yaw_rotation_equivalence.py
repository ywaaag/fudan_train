"""Match installed Isaac quaternion arithmetic without a domain simulator dependency."""
from isaacgym.torch_utils import normalize, quat_apply
import torch
import pytest
from wheel_legged_gym.domain.geometry.rotations import quat_apply_yaw


@pytest.mark.parametrize('shape', [(7,3),(2,5,3)])
def test_yaw_rotation_exact_and_nonmutating(shape):
    torch.manual_seed(19)
    vectors=torch.randn(shape)
    quaternions=torch.randn(vectors.numel()//3,4)
    quaternions[0]=0
    before=quaternions.clone();vector_before=vectors.clone()
    yaw=quaternions.clone().view(-1,4);yaw[:,:2]=0.
    expected=quat_apply(normalize(yaw),vectors)
    assert torch.equal(expected,quat_apply_yaw(quaternions,vectors))
    assert torch.equal(before,quaternions) and torch.equal(vector_before,vectors)
