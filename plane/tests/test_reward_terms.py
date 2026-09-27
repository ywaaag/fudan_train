from __future__ import annotations

import isaacgym  # noqa: F401  # Isaac Gym must initialize before torch.
import torch
from types import SimpleNamespace

from wheel_legged_gym.domain.rewards.terms import (
    capped_tracking_terms,
    wheel_rolling_terms,
)
from wheel_legged_gym.envs.base.legged_robot import LeggedRobot


def test_capped_tracking_keeps_a_gap_gradient_outside_cap():
    error = torch.tensor([0.0, 0.5, 1.0, 3.0], requires_grad=True)
    terms = capped_tracking_terms(error, cap=1.0)
    loss = terms["coarse"].sum() + terms["fine"].sum() - terms["gap"].sum()
    loss.backward()
    assert torch.isfinite(error.grad).all()
    assert float(torch.abs(error.grad[-1])) > 0.0
    assert float(terms["coarse"][0]) == 1.0


def test_wheel_rolling_terms_account_for_yaw_differential_and_sign():
    base_vx = torch.tensor([1.0])
    yaw_rate = torch.tensor([2.0])
    wheel_y = torch.tensor([[-0.2, 0.2]])
    # v_expected = vx - wz*y = [1.4, 0.6], and signed speed is -omega*r.
    wheel_ang_vel = torch.tensor([[-23.333333, -10.0]])
    contact = torch.ones((1, 2), dtype=torch.bool)
    terms = wheel_rolling_terms(
        base_vx,
        yaw_rate,
        wheel_ang_vel,
        wheel_y,
        wheel_radius=0.06,
        contact=contact,
    )
    torch.testing.assert_close(
        terms["expected_speed"], torch.tensor([[1.4, 0.6]]), atol=1e-5, rtol=1e-5
    )
    assert float(terms["residual_rms"]) < 1e-5
    assert float(terms["contact_slip"]) < 1e-5


def test_airborne_spin_is_separate_from_contact_slip():
    terms = wheel_rolling_terms(
        torch.tensor([0.0]),
        torch.tensor([0.0]),
        torch.tensor([[20.0, 0.0]]),
        torch.tensor([[-0.2, 0.2]]),
        wheel_radius=0.06,
        contact=torch.tensor([[False, True]]),
    )
    assert float(terms["airborne_spin"]) > 0.0
    assert float(terms["contact_slip"]) < 1e-6


def test_reward_activation_clipping_and_terminal_samples_stay_per_environment():
    rewards = SimpleNamespace(reward_pipeline='legacy_v0',
        unclipped_reward_names=(), clip_single_reward=1., only_positive_rewards=False)
    fake = SimpleNamespace(cfg=SimpleNamespace(rewards=rewards), dt=.01,
        reward_scales={'wheel_slip':-1.,'wheel_contact_loss':-1.,'termination':-1.},
        num_envs=3, device='cpu', rew_buf=torch.zeros(3),
        _reward_wheel_slip=lambda:torch.tensor([0.,.5,4.]),
        _reward_wheel_contact_loss=lambda:torch.tensor([0.,.5,1.]),
        _reward_termination=lambda:torch.tensor([0.,0.,1.]))
    LeggedRobot._prepare_reward_function(fake)
    assert fake.reward_names == ['wheel_slip','wheel_contact_loss']
    assert fake.reward_scales == {'wheel_slip':-.01,
                                  'wheel_contact_loss':-.01,'termination':-.01}
    LeggedRobot.compute_reward(fake)
    torch.testing.assert_close(fake.rew_buf,torch.tensor([0.,-.01,-.03]))
    torch.testing.assert_close(fake.episode_sums['wheel_slip'],
                               torch.tensor([0.,-.005,-.01]))
    torch.testing.assert_close(fake.episode_sums['wheel_contact_loss'],
                               torch.tensor([0.,-.005,-.01]))
    assert fake.episode_sums['termination'][-1] == -.01

    fake.reward_scales = {'wheel_slip':0.,'wheel_contact_loss':0.,'termination':0.}
    LeggedRobot._prepare_reward_function(fake)
    assert fake.reward_names == []


def test_turning_rolling_residual_is_per_environment_and_not_legal_yaw_differential():
    terms = wheel_rolling_terms(
        torch.tensor([1.,1.]), torch.tensor([2.,2.]),
        torch.tensor([[-23.333333,-10.],[-23.333333,0.]]),
        torch.tensor([[-.2,.2],[-.2,.2]]), .06,
        torch.ones((2,2),dtype=torch.bool))
    assert terms['contact_slip'][0] < 1e-4
    assert terms['contact_slip'][1] > .01
