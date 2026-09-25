import isaacgym
from wheel_legged_gym.contracts.wheel_legged_config import WheelLeggedCfg,WheelLeggedCfgPPO
from wheel_legged_gym.app.experiment_inputs import apply_policy_experiment


def test_clean_observations_keep_randomization_rewards_and_sampling():
    a,t=WheelLeggedCfg(),WheelLeggedCfgPPO();old=apply_policy_experiment(a,'ANCHORED_WHEEL_EXPLORE',t)
    b,u=WheelLeggedCfg(),WheelLeggedCfgPPO();new=apply_policy_experiment(b,'EXPLORE_CLEAN_OBS',u)
    assert a.noise.add_noise and not b.noise.add_noise
    assert b.domain_rand.randomize_friction and b.domain_rand.randomize_base_mass
    assert b.commands.mixture_zero_fraction==.2 and b.commands.mixture_small_fraction==.2
    assert not getattr(b.commands,'zero_retention',False)
    for k in ['reward_scales','reward_parameters','optimizer','encoder_action_anchor_coef']:
        assert old[k]==new[k]
