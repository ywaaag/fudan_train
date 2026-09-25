import isaacgym
import sys
from pathlib import Path
import torch
import pytest
from wheel_legged_gym.domain.commands.command_sampling import sample_fixed_bank
from wheel_legged_gym.contracts.wheel_legged_config import WheelLeggedCfg, WheelLeggedCfgPPO
from wheel_legged_gym.app.experiment_inputs import apply_policy_experiment

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from run_h3_low_speed import audit_grid
from compare_policy_versions import gate


def test_yaw_bank_preserves_zero_and_all_motion_anchors():
    cfg, train = WheelLeggedCfg(), WheelLeggedCfgPPO()
    apply_policy_experiment(cfg, 'LEGACY_YAW', train)
    lin = torch.tensor([[-2., 2.]]).repeat(400, 1)
    yaw = torch.tensor([[-.5, .5]]).repeat(400, 1)
    for offset in [0, 7, 31, 500]:
        samples = sample_fixed_bank(lin, yaw, torch.arange(400)+offset, cfg.commands.fixed_bank)
        assert ((samples == 0).all(dim=1)).sum() == 160
        for vx in [-.5,.5,-1,1,-1.5,1.5,-2,2]:
            assert ((samples[:,0] == vx) & (samples[:,1] == 0)).sum() == 20
        for w in [-.5,.5]:
            assert ((samples[:,0] == 0) & (samples[:,1] == w)).sum() == 40
    with pytest.raises(ValueError):
        sample_fixed_bank(lin, yaw*0, torch.arange(400), cfg.commands.fixed_bank)


def test_runner_grid_contains_all_training_endpoints():
    assert len(audit_grid('LEGACY_SPEED2_STOP')[0]) == 9
    vx, yaw = audit_grid('LEGACY_YAW')
    assert len(vx) == len(yaw) == 11
    assert set(zip(vx,yaw)) == {(0.,0.),(0.,-.5),(0.,.5)} | {
        (v,0.) for v in [-.5,.5,-1,1,-1.5,1.5,-2,2]}


def test_yaw_gate_uses_tracking_error_not_absolute_rate():
    row = dict(command=[0,.5,.4], failure_count=0, timeout_count=0,
        metrics=dict(nonwheel_contact=0, left_contact=1, right_contact=1,
                     height_mae=0, vx_mae=0, abs_yaw=.5, yaw_mae=0, torque_saturation=0))
    assert gate(row)['passed']
    row['metrics'].update(abs_yaw=0, yaw_mae=.5)
    assert gate(row)['failed_checks'] == ['yaw_mae']
