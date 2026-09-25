import isaacgym
import torch
import json
from wheel_legged_gym.learning.modules.policy_symmetry import mirror_observations,mirror_actions,mirror_history


def test_mirror_contract_signs_and_history_order():
    obs=torch.arange(25,dtype=torch.float).unsqueeze(0)+1
    mirrored=mirror_observations(obs)
    assert torch.equal(mirrored[:,:3],torch.tensor([[-1.,2.,-3.]]))
    assert torch.equal(mirrored[:,6:9],torch.tensor([[7.,-8.,9.]]))
    assert torch.equal(mirrored[:,9:13],obs[:,[11,12,9,10]])
    assert torch.equal(mirror_observations(mirrored),obs)
    history=torch.cat([obs+100*i for i in range(5)],dim=-1)
    assert torch.equal(mirror_history(history).reshape(1,5,25)[:,3],mirror_observations(obs+300))
    actions=torch.tensor([[1.,2.,3.,4.,5.,6.]])
    assert torch.equal(mirror_actions(actions),torch.tensor([[4.,5.,6.,1.,2.,3.]]))
    assert torch.equal(mirror_actions(mirror_actions(actions)),actions)


def test_static_control_disables_only_dynamic_commands(tmp_path,monkeypatch):
    from wheel_legged_gym.app.experiment_inputs import apply_motion_goal
    from wheel_legged_gym.contracts.wheel_legged_config import WheelLeggedCfg,WheelLeggedCfgPPO
    from wheel_legged_gym.contracts.config_serialization import class_to_dict
    path=tmp_path/'spec.json';monkeypatch.setenv('FUDAN_MOTION_GOAL_SPEC',str(path))
    spec={'stage':'basic_motion','geometry_symmetry':True,'geometry_weight':-.2,'freeze_motion_encoder':True,
          'start_stop_ramp_seconds':1.,'start_stop_speed':2.,'start_stop_fraction':.25,
          'dynamic_fixed_lr':1e-6,'dynamic_equivariance':.01}
    path.write_text(json.dumps(spec));cfg,train=WheelLeggedCfg(),WheelLeggedCfgPPO();apply_motion_goal(cfg,train)
    before=class_to_dict(cfg);optim=class_to_dict(train)
    spec['start_stop_fraction']=0.;path.write_text(json.dumps(spec))
    cfg,train=WheelLeggedCfg(),WheelLeggedCfgPPO();manifest=apply_motion_goal(cfg,train);after=class_to_dict(cfg)
    assert cfg.commands.start_stop_ramp_seconds==0 and cfg.commands.hold_command_until_reset
    assert manifest['start_stop']['static_control'] and manifest['start_stop']['dynamic_fraction']==0
    for key in ['start_stop_ramp_seconds','start_stop_stride']:after['commands'][key]=before['commands'][key]
    assert before==after and optim==class_to_dict(train)
