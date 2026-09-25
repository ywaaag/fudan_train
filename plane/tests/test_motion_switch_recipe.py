import isaacgym
import json
from wheel_legged_gym.app.experiment_inputs import apply_motion_goal
from wheel_legged_gym.contracts.wheel_legged_config import WheelLeggedCfg,WheelLeggedCfgPPO
from wheel_legged_gym.contracts.config_serialization import class_to_dict


def test_switch_changes_only_command_timing(tmp_path,monkeypatch):
    p=tmp_path/'spec.json';monkeypatch.setenv('FUDAN_MOTION_GOAL_SPEC',str(p))
    spec={'stage':'basic_motion','geometry_symmetry':True}
    p.write_text(json.dumps(spec));cfg,train=WheelLeggedCfg(),WheelLeggedCfgPPO()
    apply_motion_goal(cfg,train);before=class_to_dict(cfg);optimizer=class_to_dict(train)
    spec['command_switch_seconds']=5;p.write_text(json.dumps(spec))
    cfg,train=WheelLeggedCfg(),WheelLeggedCfgPPO();apply_motion_goal(cfg,train);after=class_to_dict(cfg)
    assert not cfg.commands.hold_command_until_reset and cfg.commands.resampling_time==5
    for key in ['hold_command_until_reset','resampling_time']:after['commands'][key]=before['commands'][key]
    assert before==after and optimizer==class_to_dict(train)
