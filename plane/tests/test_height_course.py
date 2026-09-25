import isaacgym
import torch
import pytest
from wheel_legged_gym.domain.commands.command_sampling import sample_height_bank
from wheel_legged_gym.experiments.recipes.height_course import STAGES,height_bank
from wheel_legged_gym.experiments.recipes.motion_goal import stage_bank


def test_height_bank_keeps_full_motion_and_paired_height_targets():
    previous=set()
    for stage in STAGES:
        bank=height_bank(stage);commands=set(bank)
        assert {(v,w,.4) for v,w in stage_bank('yaw4')}<=commands
        assert previous<=commands
        previous=commands
        n=len(bank)*3
        for offset in [0,7,31]:
            slots=torch.arange(n)+offset
            selected=sample_height_bank(torch.tensor([[-4.,4.]]).repeat(n,1),
                torch.tensor([[-4.,4.]]).repeat(n,1),torch.tensor([[.3,.45]]).repeat(n,1),slots,bank)
            assert torch.equal(selected,torch.tensor(bank)[slots%len(bank)])
    assert (0,0,.30) in previous and (0,0,.45) in previous


def test_out_of_range_height_rejected():
    with pytest.raises(ValueError):
        sample_height_bank(torch.tensor([[-4.,4.]]),torch.tensor([[-4.,4.]]),
            torch.tensor([[.375,.425]]),torch.tensor([0]),[(0,0,.45)])


def test_dual_height_has_equal_exposure_in_one_policy():
    bank=height_bank('dual')
    assert bank==[(0.,0.,.35),(0.,0.,.45)]
    selected=sample_height_bank(torch.tensor([[-4.,4.]]).repeat(80,1),
        torch.tensor([[-4.,4.]]).repeat(80,1),torch.tensor([[.35,.45]]).repeat(80,1),
        torch.arange(80)+19,bank)
    assert (selected[:,2]==.35).sum()==40 and (selected[:,2]==.45).sum()==40
    assert not selected[:,:2].any()


def test_fixed_height_diagnostics_sample_only_requested_parking_target(tmp_path,monkeypatch):
    import json
    from wheel_legged_gym.app.experiment_inputs import apply_height_course
    from wheel_legged_gym.contracts.wheel_legged_config import WheelLeggedCfg,WheelLeggedCfgPPO
    path=tmp_path/'spec.json';monkeypatch.setenv('FUDAN_HEIGHT_SPEC',str(path))
    for stage,h in [('fixed35',.35),('fixed45',.45)]:
        path.write_text(json.dumps({'stage':stage}))
        cfg,train=WheelLeggedCfg(),WheelLeggedCfgPPO();m=apply_height_course(cfg,train)
        assert cfg.commands.height_bank==[(0.,0.,h)]
        assert cfg.commands.ranges.height==[h,h]
        assert cfg.commands.height_switch_interval==0
        assert cfg.rewards.scales.base_height==1
        assert 'Diagnostic branch only' in m['scope']
        selected=sample_height_bank(torch.tensor([[-4.,4.]]).repeat(80,1),
            torch.tensor([[-4.,4.]]).repeat(80,1),torch.tensor([[h,h]]).repeat(80,1),
            torch.arange(80)+501,cfg.commands.height_bank)
        assert torch.equal(selected,torch.tensor([[0.,0.,h]]).repeat(80,1))


def test_micro_retains_motion_and_requires_distinct_heights():
    bank=height_bank('micro',4)
    assert {c[2] for c in bank}=={.39,.4,.41}
    assert len(bank)==128
    assert {(v,w,.4) for v,w in stage_bank('yaw4')}<=set(bank)
    # A policy always at .40 must not pass the tighter .005m micro tolerance.
    assert all(abs(h-.4)>.005 for h in [.39,.41])


def test_reweight_changes_only_variable_height_exposure():
    old=height_bank('near');new=height_bank('near',4)
    assert set(old)==set(new)
    assert [c for c in old if c[2]==.4]==[c for c in new if c[2]==.4]
    assert len(new)==128
    assert sum(c[2]!=.4 for c in new)==80
    for c in set(old):
        assert new.count(c)==old.count(c)*(1 if c[2]==.4 else 4)


def test_height_gain_changes_only_height_contribution(tmp_path,monkeypatch):
    import json
    from wheel_legged_gym.app.experiment_inputs import apply_height_course
    from wheel_legged_gym.contracts.wheel_legged_config import WheelLeggedCfg,WheelLeggedCfgPPO
    from wheel_legged_gym.contracts.config_serialization import class_to_dict
    path=tmp_path/'spec.json';monkeypatch.setenv('FUDAN_HEIGHT_SPEC',str(path))
    path.write_text(json.dumps({'stage':'near','variable_repeats':4}))
    before,bt=WheelLeggedCfg(),WheelLeggedCfgPPO();apply_height_course(before,bt)
    old=class_to_dict(before);old_train=class_to_dict(bt)
    path.write_text(json.dumps({'stage':'near','variable_repeats':4,'height_reward_gain':4}))
    after,at=WheelLeggedCfg(),WheelLeggedCfgPPO();apply_height_course(after,at)
    new=class_to_dict(after)
    assert new['commands']==old['commands']
    assert class_to_dict(at)==old_train
    assert new['rewards']['scales']['base_height']==4
    new['rewards']['scales']['base_height']=old['rewards']['scales']['base_height']
    new['rewards'].pop('unclipped_reward_names',None)
    old['rewards'].pop('unclipped_reward_names',None)
    assert new==old
