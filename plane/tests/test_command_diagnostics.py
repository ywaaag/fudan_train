import isaacgym
import torch
import pytest
from types import SimpleNamespace
import wheel_legged_gym.envs
from wheel_legged_gym.utils.command_diagnostics import CommandDiagnostics
from wheel_legged_gym.contracts.wheel_legged_config import WheelLeggedCfg, WheelLeggedCfgPPO
from wheel_legged_gym.app.experiment_inputs import apply_policy_experiment


def test_time_weighting_empty_bins_and_reward_deltas(tmp_path):
    env=SimpleNamespace(device='cpu',dt=.01,commands=torch.tensor([[.5,0,.4]]*3+[[0,0,.4]]),
        obs_scales=SimpleNamespace(lin_vel=2.),base_lin_vel=torch.tensor([[.3,0,0],[.4,0,0],[.5,0,0],[.1,0,0]]),
        base_ang_vel=torch.zeros(4,3),base_height=torch.full((4,),.4),
        episode_sums={'track':torch.zeros(4)},reset_buf=torch.tensor([0,1,0,0]),time_out_buf=torch.tensor([0,0,0,0]))
    def reward():env.episode_sums['track']+=torch.tensor([.01,.02,.03,.04])
    env.compute_reward=reward
    d=CommandDiagnostics(env,tmp_path)
    d.before_step(env.base_lin_vel*2+.02)
    env.compute_reward()
    # A reset clears episode counters, but the diagnostic already captured its reward.
    env.episode_sums['track'][1]=0
    row=d.flush(0)['rows']
    forward=next(x for x in row if x['command_vx']==.5)
    assert forward['samples']==3 and forward['env_seconds']==.03
    assert forward['vx_mae']==pytest.approx(.1)
    assert forward['encoder_vx_bias']==pytest.approx(.01)
    assert forward['failures']==1
    assert forward['reward_per_env_second']['track']==pytest.approx(2.)
    missing=next(x for x in row if x['command_vx']==-1)
    assert missing['samples']==0 and missing['vx_mae'] is None
    assert sum(x['samples'] for x in d.flush(1)['rows'])==0


def test_profiles_only_differ_in_encoder_step():
    configs=[]; manifests=[]
    for name in ['ENCODER_FROZEN','ENCODER_UPDATING']:
        c,t=WheelLeggedCfg(),WheelLeggedCfgPPO()
        manifests.append(apply_policy_experiment(c,name,t))
        configs.append((c,t))
    a,b=manifests
    assert a['freeze_encoder_updates'] and not b['freeze_encoder_updates']
    for k in set(a)|set(b):
        if k not in ['name','freeze_encoder_updates']:assert a[k]==b[k],k
