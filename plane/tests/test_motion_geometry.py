import isaacgym
import torch
import json
from types import SimpleNamespace as NS
from wheel_legged_gym.envs.base.legged_robot import LeggedRobot
from wheel_legged_gym.app.experiment_inputs import apply_motion_goal
from wheel_legged_gym.contracts.wheel_legged_config import WheelLeggedCfg,WheelLeggedCfgPPO
from wheel_legged_gym.contracts.config_serialization import class_to_dict


def test_geometry_reward_straight_only_and_body_rotation_invariant():
    env=LeggedRobot.__new__(LeggedRobot)
    env.cfg=NS(rewards=NS(bilateral_geometry_tolerance_m=.005,bilateral_geometry_scale_m=.1,
                          straight_bilateral_geometry=True))
    env.bilateral_body_indices=torch.arange(4)
    points=torch.tensor([[.1,-.2,-.2],[.1,.2,-.2],[0,-.2,-.34],[0,.2,-.34]])
    env.rigid_body_states=torch.zeros(3,4,13);env.rigid_body_states[:,:,:3]=points
    env.root_states=torch.zeros(3,13)
    env.base_quat=torch.tensor([[0.,0.,0.,1.]]).repeat(3,1)
    env.commands=torch.tensor([[0.,0.,.4],[1.,0.,.4],[1.,.5,.4]])
    assert torch.equal(env._reward_stand_bilateral_geometry(),torch.zeros(3))
    env.rigid_body_states[:,0,0]+=.10
    penalty=env._reward_stand_bilateral_geometry()
    assert penalty[0]>0 and penalty[1]==penalty[0] and penalty[2]==0
    # Rotate all body landmarks and base together 90 degrees around Z.
    xyz=env.rigid_body_states[:,:,:3].clone()
    env.rigid_body_states[:,:,0]=-xyz[:,:,1];env.rigid_body_states[:,:,1]=xyz[:,:,0]
    env.base_quat[:]=torch.tensor([0.,0.,2**-.5,2**-.5])
    assert torch.allclose(env._reward_stand_bilateral_geometry(),penalty,atol=1e-6)


def test_geometry_ablation_preserves_command_and_optimizer(tmp_path,monkeypatch):
    path=tmp_path/'spec.json';monkeypatch.setenv('FUDAN_MOTION_GOAL_SPEC',str(path))
    path.write_text(json.dumps({'stage':'turn1'}))
    cfg,train=WheelLeggedCfg(),WheelLeggedCfgPPO();apply_motion_goal(cfg,train)
    old=class_to_dict(cfg);optim=class_to_dict(train)
    path.write_text(json.dumps({'stage':'turn1','geometry_symmetry':True}))
    cfg,train=WheelLeggedCfg(),WheelLeggedCfgPPO();apply_motion_goal(cfg,train)
    new=class_to_dict(cfg)
    assert old['commands']==new['commands'] and optim==class_to_dict(train)
    changed={k for k,v in new['rewards']['scales'].items() if old['rewards']['scales'].get(k)!=v}
    assert changed=={'nominal_state','stand_bilateral_geometry'}
    assert cfg.rewards.scales.nominal_state==0 and cfg.rewards.scales.stand_bilateral_geometry==-.1


def test_recovery_does_not_trade_back_good_posture_for_speed():
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
    from wheel_legged_gym.evaluation.motion_candidates import geometry_retained
    reference={'0.0':.023,'1.0':.029,'-4.0':.116}
    assert geometry_retained({'0.0':.025,'1.0':.03,'-4.0':.12},reference)
    assert not geometry_retained({'0.0':.04,'1.0':.029,'-4.0':.1},reference)
    assert not geometry_retained({'0.0':.023,'1.0':.029,'-4.0':.13},reference)
    assert not geometry_retained({'0.0':.023},reference)


def test_reviewed_high_speed_tolerance_keeps_low_speed_strict():
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
    from wheel_legged_gym.evaluation.motion_candidates import geometry_retained
    ref={'0.0':.023,'1.0':.02,'4.0':.027}
    probe={'0.0':.025,'1.0':.021,'4.0':.034}
    assert not geometry_retained(probe,ref)
    assert geometry_retained(probe,ref,.035)
    probe['1.0']=.034
    assert not geometry_retained(probe,ref,.035)
