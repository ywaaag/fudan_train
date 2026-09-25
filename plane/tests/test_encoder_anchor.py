import isaacgym
import torch
from wheel_legged_gym.learning.storage.rollout_storage import RolloutStorage
from wheel_legged_gym.contracts.wheel_legged_config import WheelLeggedCfg, WheelLeggedCfgPPO
from wheel_legged_gym.app.experiment_inputs import apply_policy_experiment


def test_encoder_anchor_is_single_ablation_variable():
    a,t=WheelLeggedCfg(),WheelLeggedCfgPPO(); anchored=apply_policy_experiment(a,'ENCODER_ANCHORED',t)
    b,u=WheelLeggedCfg(),WheelLeggedCfgPPO(); updating=apply_policy_experiment(b,'ENCODER_UPDATING',u)
    assert anchored['encoder_action_anchor_coef']==1.0
    assert anchored['freeze_encoder_updates'] is False
    assert updating['freeze_encoder_updates'] is False
    for key in set(anchored)|set(updating):
        if key not in {'name','profile','encoder_action_anchor_coef','ablation_variable'}:
            assert anchored[key]==updating[key], key


def test_anchor_batch_uses_current_observation_aligned_to_history():
    storage=RolloutStorage(2,2,[25],[28],[125],[6],device='cpu')
    ids=torch.arange(4.).reshape(2,2,1)
    storage.observations[:]=ids
    storage.next_observations[:]=ids+100
    storage.observation_history[:]=ids+200
    storage.privileged_observations[:]=ids+300
    for nxt,critic,history,current in storage.encoder_mini_batch_generator(2,1,include_current_obs=True):
        assert torch.equal(nxt[:,0]-100,current[:,0])
        assert torch.equal(history[:,0]-200,current[:,0])
        assert torch.equal(critic[:,0]-300,current[:,0])


def test_wheel_exploration_changes_only_initialization_setting():
    c,t=WheelLeggedCfg(),WheelLeggedCfgPPO();a=apply_policy_experiment(c,'ENCODER_ANCHORED',t)
    c,t=WheelLeggedCfg(),WheelLeggedCfgPPO();b=apply_policy_experiment(c,'ANCHORED_WHEEL_EXPLORE',t)
    for k in set(a)|set(b):
        if k not in {'name','profile','ablation_variable','wheel_exploration_initialization'}:
            assert a[k]==b[k],k
