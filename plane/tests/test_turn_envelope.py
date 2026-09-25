import math

import isaacgym
import torch

from wheel_legged_gym.domain.commands.turn_envelope import TurnEnvelope
from wheel_legged_gym.domain.commands.resampling import CommandBuffers, resample
from wheel_legged_gym.domain.rewards.turn_lean import orientation_cost, roll_reference
from wheel_legged_gym.experiments.recipes.turn_envelope import LOW_TURNS, apply_turn_envelope
from wheel_legged_gym.contracts.wheel_legged_config import WheelLeggedCfg, WheelLeggedCfgPPO
from wheel_legged_gym.learning.modules.policy_retention import reference_loss


def test_lean_reference_sign_cap_mirror_and_zero():
    vx = torch.tensor([0., 1., 1., -1., -1., 4.])
    yaw = torch.tensor([1., 1., -1., 1., -1., 4.])
    cap = math.radians(2.)
    roll = roll_reference(vx, yaw, cap)
    assert roll[0] == 0 and roll[1] > 0 and roll[2] < 0
    assert roll[3] < 0 and roll[4] > 0 and roll.abs().max() <= cap
    torch.testing.assert_close(roll_reference(vx, -yaw, cap), -roll)
    gravity = torch.stack((torch.zeros_like(roll), -roll.sin(), -roll.cos()), dim=1)
    original = gravity.clone()
    torch.testing.assert_close(orientation_cost(gravity, vx, yaw, cap), torch.zeros_like(roll), atol=1e-7, rtol=0)
    torch.testing.assert_close(gravity, original)
    pitched = gravity.clone()
    pitched[:, 0] = .2
    torch.testing.assert_close(orientation_cost(pitched, vx, yaw, cap), torch.full_like(roll, .04), atol=1e-7, rtol=0)
    flat = torch.tensor([[.1, .2, -.97]])
    torch.testing.assert_close(orientation_cost(flat, torch.zeros(1), torch.zeros(1), cap),
                               flat[:, :2].square().sum(dim=1))
    assert gravity.shape == (6, 3)


def test_turn_bank_and_recipe_match_except_target():
    assert len(LOW_TURNS) == 24
    assert set(LOW_TURNS) == {(sv*v, sw*w) for v in (.5, 1., 2.) for w in (.25, .5)
                              for sv in (-1., 1.) for sw in (-1., 1.)}
    source = dict(source_checkpoint='/tmp/source.pt', source_sha256='a8b9dc01879ddba41c54289c0367c6a3c93325bc2982357173b1899be059b790', source_iteration=10200)
    configs = []
    for branch in ('A', 'B'):
        cfg, train = WheelLeggedCfg(), WheelLeggedCfgPPO()
        manifest = apply_turn_envelope(cfg, train, spec=dict(source, branch=branch))
        assert cfg.commands.ranges.height == [.4, .4]
        assert cfg.commands.turn_stride == 4
        assert cfg.rewards.scales.stand_bilateral_geometry == -.2
        assert train.algorithm.learning_rate == 1e-6
        assert train.algorithm.symmetry_loss_coef == .01
        assert manifest['dynamic_reference_coef'] == 1.
        configs.append((cfg, train, manifest))
    assert configs[0][0].rewards.turn_lean_max_rad == 0.
    assert configs[1][0].rewards.turn_lean_max_rad == math.radians(2.)
    assert configs[0][2]['retention_bank'] == configs[1][2]['retention_bank']


def test_turn_scheduler_changes_only_cohort():
    schedule = TurnEnvelope(8, 'cpu', .01)
    commands = torch.full((8, 3), 9.)
    ids = torch.tensor([0, 4])
    targets = torch.tensor([[1., .5], [-1., -.5]])
    schedule.reset(commands, ids, targets)
    for _ in range(450):
        schedule.advance(commands)
    assert commands[0, 0] > 0 and commands[0, 1] > 0
    assert commands[4, 0] < 0 and commands[4, 1] < 0
    assert torch.all(commands[torch.tensor([1, 2, 3, 5, 6, 7])] == 9.)


def test_resampling_and_reference_are_separated_by_environment_id():
    cfg, train = WheelLeggedCfg(), WheelLeggedCfgPPO()
    source = dict(source_checkpoint='/tmp/source.pt', source_sha256='a8b9dc01879ddba41c54289c0367c6a3c93325bc2982357173b1899be059b790', source_iteration=10200, branch='B')
    apply_turn_envelope(cfg, train, spec=source)
    ids = torch.arange(64)
    state = CommandBuffers(torch.zeros(64, 4), torch.full((64,), -1), ids.clone())
    ranges = {key: torch.tensor(getattr(cfg.commands.ranges, key)).repeat(64, 1)
              for key in ('lin_vel_x', 'ang_vel_yaw', 'height')}
    scheduler = TurnEnvelope(64, 'cpu', .01)
    resample(ids, state=state, ranges=ranges, config=cfg.commands, device='cpu',
             reset_start_stop=lambda *_: None, sample_heading=lambda *_args, **_kwargs: None,
             reset_turn=scheduler.reset)
    assert torch.all(state.commands[ids[ids % 4 == 0], :2] == 0.)
    assert torch.all(state.commands[:, 2] == .4)
    assert set(map(tuple, scheduler.target[ids[ids % 4 == 0]].tolist())).issubset(set(LOW_TURNS))
    mean = torch.ones(64, 6)
    reference = torch.zeros_like(mean)
    _, fraction = reference_loss(mean, reference, torch.ones(6), ids, 4)
    assert fraction == .75
    mean[ids % 4 == 0] = 1000.
    penalty, fraction = reference_loss(mean, reference, torch.ones(6), ids, 4)
    assert fraction == .75 and penalty == .5
