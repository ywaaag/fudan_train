import isaacgym
import torch
from wheel_legged_gym.learning.modules.policy_retention import reference_loss
from wheel_legged_gym.learning.storage.rollout_storage import RolloutStorage


def test_only_retention_cohort_receives_reference_gradient():
    mean=torch.ones(8,6,requires_grad=True)
    teacher=torch.zeros(8,6,requires_grad=True)
    std=torch.ones(6,requires_grad=True)
    loss,fraction=reference_loss(mean,teacher,std,torch.arange(8),4)
    assert float(fraction)==.75 and float(loss)==.5
    loss.backward()
    assert not mean.grad[[0,4]].any()
    assert mean.grad[[1,2,3,5,6,7]].abs().sum()>0
    assert teacher.grad is None and std.grad is None


def test_minibatch_environment_ids_follow_shuffled_samples():
    storage=RolloutStorage(8,3,[25],[28],[125],[6])
    storage.observations[:,:,0]=torch.arange(8)
    storage.track_minibatch_env_ids=True
    for batch in storage.mini_batch_generator(3,2):
        assert torch.equal(batch[0][:,0].long(),storage.minibatch_env_ids)


def test_no_retention_samples_is_finite_zero():
    mean=torch.ones(2,6,requires_grad=True)
    loss,fraction=reference_loss(mean,torch.zeros_like(mean),torch.ones(6),torch.tensor([0,4]),4)
    assert loss==0 and fraction==0
    loss.backward();assert not mean.grad.any()
