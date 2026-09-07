from __future__ import annotations

import isaacgym  # noqa: F401  # Isaac Gym must initialize before torch.
import torch

from wheel_legged_gym.envs.base.reward_terms import (
    capped_tracking_terms,
    wheel_rolling_terms,
)


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
