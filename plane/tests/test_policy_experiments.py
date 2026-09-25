from __future__ import annotations

import unittest

import isaacgym  # must precede torch in the Isaac Gym environment
import torch

from wheel_legged_gym.domain.commands.command_sampling import (
    MODE_FORWARD,
    MODE_REVERSE,
    MODE_SMALL,
    MODE_ZERO,
    sample_zero_reverse_mixture,
)
from wheel_legged_gym.app.experiment_inputs import apply_policy_experiment
from wheel_legged_gym.app.experiment_inputs import apply_training_profile
from wheel_legged_gym.contracts.wheel_legged_config import (
    WheelLeggedCfg,
    WheelLeggedCfgPPO,
)
from wheel_legged_gym.scripts.select_policy_checkpoint import score_grid


class PolicyExperimentTest(unittest.TestCase):
    def test_stand_entropy_survives_manifest_optimizer_overrides(self):
        from types import SimpleNamespace
        from wheel_legged_gym.app.optimizer_overrides import enforce_optimizer_overrides
        cfg, train = WheelLeggedCfg(), WheelLeggedCfgPPO()
        manifest = apply_training_profile(cfg, train, phase="stand", level=0)
        alg = SimpleNamespace(entropy_coef=train.algorithm.entropy_coef,
            optimizer=SimpleNamespace(param_groups=[{"lr": 1.0}]), extra_optimizer=None)
        enforce_optimizer_overrides(SimpleNamespace(alg=alg), manifest)
        self.assertEqual(alg.entropy_coef, 0.001)
        self.assertEqual(manifest["optimizer"]["entropy_coef"], train.algorithm.entropy_coef)

    def test_mixture_has_requested_modes_signs_and_exact_zero(self) -> None:
        torch.manual_seed(7)
        count = 200_000
        linear_ranges = torch.tensor([[-0.5, 0.5]]).repeat(count, 1)
        yaw_ranges = torch.tensor([[-0.8, 0.8]]).repeat(count, 1)
        linear, yaw, mode = sample_zero_reverse_mixture(
            linear_ranges,
            yaw_ranges,
            zero_fraction=0.20,
            small_fraction=0.20,
            reverse_fraction=0.30,
            forward_fraction=0.30,
            small_linear_limit=0.10,
            small_yaw_limit=0.10,
            small_yaw_only_fraction=0.50,
            linear_yaw_zero_fraction=0.50,
        )
        expected = {MODE_ZERO: 0.20, MODE_SMALL: 0.20, MODE_REVERSE: 0.30, MODE_FORWARD: 0.30}
        for mode_id, fraction in expected.items():
            actual = float(torch.mean((mode == mode_id).float()))
            self.assertAlmostEqual(actual, fraction, delta=0.004)
        self.assertTrue(torch.all(linear[mode == MODE_ZERO] == 0.0))
        self.assertTrue(torch.all(yaw[mode == MODE_ZERO] == 0.0))
        self.assertTrue(torch.all(linear[mode == MODE_REVERSE] <= 0.0))
        self.assertTrue(torch.all(linear[mode == MODE_FORWARD] >= 0.0))
        self.assertLessEqual(float(torch.max(torch.abs(linear[mode == MODE_SMALL]))), 0.10)
        self.assertLessEqual(float(torch.max(torch.abs(yaw[mode == MODE_SMALL]))), 0.10)

    def test_experiments_do_not_change_interface_or_baseline_defaults(self) -> None:
        baseline = WheelLeggedCfg()
        self.assertEqual(baseline.commands.sampling_strategy, "uniform")
        self.assertEqual(baseline.rewards.scales.zero_base_velocity, -1.0)
        self.assertEqual(baseline.rewards.scales.zero_wheel_velocity, -1.0)
        cfg = WheelLeggedCfg()
        manifest = apply_policy_experiment(cfg, "H7", WheelLeggedCfgPPO())
        self.assertEqual(manifest["name"], "H7")
        self.assertEqual(cfg.env.num_observations, 25)
        self.assertEqual(cfg.env.num_actions, 6)
        self.assertEqual(cfg.env.obs_history_length, 5)
        self.assertEqual(cfg.commands.sampling_strategy, "zero_reverse_mixture")
        self.assertLess(cfg.rewards.scales.zero_base_velocity, 0.0)
        self.assertLess(cfg.rewards.scales.zero_wheel_velocity, 0.0)

        self.assertEqual(cfg.commands.curriculum_stages, ((3.0, 3.0),))
        self.assertEqual(cfg.commands.curriculum_required_passes, 5)
        self.assertEqual(cfg.rewards.scales.high_speed_slip, -7.0)
        self.assertEqual(cfg.rewards.scales.high_speed_yaw_penalty, -2.0)

    def test_small_anchor_commands_are_discrete_and_axis_aligned(self) -> None:
        torch.manual_seed(19)
        count = 20_000
        ranges = torch.tensor([[-0.5, 0.5]]).repeat(count, 1)
        yaw_ranges = torch.tensor([[-0.8, 0.8]]).repeat(count, 1)
        linear, yaw, mode = sample_zero_reverse_mixture(
            ranges,
            yaw_ranges,
            zero_fraction=0.0,
            small_fraction=1.0,
            reverse_fraction=0.0,
            forward_fraction=0.0,
            small_linear_limit=0.10,
            small_yaw_limit=0.10,
            small_yaw_only_fraction=0.50,
            linear_yaw_zero_fraction=0.50,
            small_linear_anchors=(-0.10, -0.05, 0.05, 0.10),
            small_yaw_anchors=(-0.05, 0.05),
        )
        self.assertTrue(torch.all(mode == MODE_SMALL))
        self.assertTrue(torch.all((linear == 0.0) ^ (yaw == 0.0)))
        expected_linear = torch.tensor([-0.10, -0.05, 0.05, 0.10])
        expected_yaw = torch.tensor([-0.05, 0.05])
        for value in torch.unique(linear[linear != 0.0]):
            self.assertLess(float(torch.min(torch.abs(expected_linear - value))), 1e-6)
        for value in torch.unique(yaw[yaw != 0.0]):
            self.assertLess(float(torch.min(torch.abs(expected_yaw - value))), 1e-6)

    def test_endpoint_anchors_preserve_signed_forward_reverse_modes(self) -> None:
        torch.manual_seed(23)
        count = 20_000
        ranges = torch.tensor([[-3.5, 3.5]]).repeat(count, 1)
        yaw_ranges = torch.zeros_like(ranges)
        linear, yaw, mode = sample_zero_reverse_mixture(
            ranges,
            yaw_ranges,
            zero_fraction=0.0,
            small_fraction=0.0,
            reverse_fraction=0.5,
            forward_fraction=0.5,
            small_linear_limit=0.1,
            small_yaw_limit=0.1,
            small_yaw_only_fraction=0.0,
            linear_yaw_zero_fraction=1.0,
            endpoint_anchor_fraction=1.0,
            endpoint_linear_anchors=(-3.5, -3.25, -3.0, -2.5, 2.5, 3.0, 3.25, 3.5),
        )
        self.assertTrue(torch.all(linear[mode == MODE_REVERSE] < 0.0))
        self.assertTrue(torch.all(linear[mode == MODE_FORWARD] > 0.0))
        self.assertTrue(torch.all(yaw == 0.0))
        self.assertGreaterEqual(float(torch.min(torch.abs(linear))), 2.5)

    def test_h7_enables_narrow_high_speed_bridge(self) -> None:
        cfg = WheelLeggedCfg()
        train_cfg = WheelLeggedCfgPPO()
        manifest = apply_policy_experiment(cfg, "H7", train_cfg)
        self.assertEqual(manifest["name"], "H7")
        self.assertEqual(cfg.commands.mixture_zero_fraction, 0.25)
        self.assertEqual(cfg.commands.mixture_small_fraction, 0.25)
        self.assertEqual(cfg.commands.mixture_endpoint_anchor_fraction, 0.50)
        self.assertEqual(cfg.commands.mixture_endpoint_linear_anchors[-1], 3.00)
        self.assertEqual(cfg.rewards.scales.zero_yaw_wheel_symmetry, 0.0)

    def test_h3_uses_conservative_optimizer_and_balanced_penalties(self) -> None:
        cfg = WheelLeggedCfg()
        train_cfg = WheelLeggedCfgPPO()
        apply_policy_experiment(cfg, "H3", train_cfg)
        self.assertEqual(train_cfg.algorithm.learning_rate, 2.0e-6)
        self.assertEqual(train_cfg.algorithm.entropy_coef, 0.0002)
        self.assertEqual(cfg.commands.mixture_zero_fraction, 0.35)
        self.assertEqual(cfg.commands.mixture_endpoint_anchor_fraction, 0.50)
        self.assertEqual(cfg.rewards.scales.zero_base_velocity, -8.0)
        self.assertEqual(cfg.rewards.scales.zero_wheel_velocity, -5.0)

    def test_checkpoint_score_uses_command_specific_tolerances(self) -> None:
        commands = (
            (0.0, 0.0, 0.4, 0.02, 0.00),
            (0.05, 0.0, 0.4, 0.06, 0.00),
            (-0.05, 0.0, 0.4, -0.04, 0.00),
            (0.10, 0.0, 0.4, 0.11, 0.00),
            (-0.10, 0.0, 0.4, -0.09, 0.00),
            (0.0, 0.05, 0.4, 0.00, 0.04),
            (0.0, -0.05, 0.4, 0.00, -0.04),
        )
        payload = {
            "passed": True,
            "results": [
                {
                    "command": [forward, yaw, height],
                    "mean_vx_m_s": vx,
                    "mean_yaw_rate_rad_s": yaw_rate,
                    "termination_count": 0,
                    "passed": True,
                }
                for forward, yaw, height, vx, yaw_rate in commands
            ],
        }
        score = score_grid(payload)
        self.assertTrue(score["passed"])
        self.assertEqual(score["passed_commands"], 7)
        self.assertAlmostEqual(score["max_normalized_error"], 2.0 / 3.0)
        self.assertEqual(score["termination_count"], 0)

    def test_method_v1_profile_preserves_contract_and_enables_normalized_terms(self):
        cfg = WheelLeggedCfg()
        train_cfg = WheelLeggedCfgPPO()
        manifest = apply_training_profile(
            cfg, train_cfg, profile="method_v1", phase="translate", level=0
        )
        self.assertEqual(manifest["reward_version"], "normalized_v1")
        self.assertEqual(cfg.env.num_observations, 25)
        self.assertEqual(cfg.env.num_actions, 6)
        self.assertEqual(cfg.env.obs_history_length, 5)
        self.assertEqual(cfg.commands.ranges.lin_vel_x, [-0.5, 0.5])
        self.assertEqual(cfg.commands.ranges.ang_vel_yaw, [0.0, 0.0])
        self.assertEqual(cfg.rewards.scales.track_vx_coarse, 1.0)
        self.assertEqual(cfg.rewards.scales.method_termination, -5.0)
        self.assertEqual(train_cfg.algorithm.extra_learning_rate, 1.0e-5)

    def test_method_v1_rejects_invalid_command_level(self):
        cfg = WheelLeggedCfg()
        with self.assertRaises(ValueError):
            apply_training_profile(
                cfg, WheelLeggedCfgPPO(), profile="method_v1", phase="yaw", level=99
            )

    def test_method_v1_command_sampling_uses_slot_ids(self):
        ranges = torch.tensor([[-1.0, 1.0]])
        yaw_ranges = torch.tensor([[-1.0, 1.0]])
        from wheel_legged_gym.domain.commands.command_sampling import sample_method_v1

        linear, _, _ = sample_method_v1(
            ranges, yaw_ranges, phase="translate", slot_ids=torch.tensor([7])
        )
        self.assertGreater(float(torch.abs(linear[0])), 0.0)
        linear, _, _ = sample_method_v1(
            ranges.repeat(10, 1),
            yaw_ranges.repeat(10, 1),
            phase="translate",
            slot_ids=torch.arange(10),
        )
        self.assertTrue(torch.any(linear < 0.0))
        self.assertTrue(torch.any(linear > 0.0))


if __name__ == "__main__":
    unittest.main()
