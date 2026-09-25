# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin


"""Frozen curriculum control flow before extraction; do not regenerate."""
from wheel_legged_gym.domain.commands.command_curriculum import evaluate_curriculum_window

class OriginalProgress:
    def _maybe_advance_staged_curriculum(self):
        if not self._uses_staged_curriculum():
            return
        interval = int(self.cfg.commands.curriculum_check_interval_steps)
        if self.common_step_counter - self.command_curriculum_last_check_step < interval:
            return
        if (
            self.command_curriculum_window_episodes
            < self.cfg.commands.curriculum_min_episodes
        ):
            return
        stats = {
            "episodes": self.command_curriculum_window_episodes,
            **{
                name: float(value.item())
                for name, value in self.command_curriculum_window.items()
            },
        }
        metrics = evaluate_curriculum_window(stats, self.cfg.commands)
        self.command_curriculum_last_metrics = metrics
        self.command_curriculum_last_check_step = self.common_step_counter
        if metrics["passed"]:
            self.command_curriculum_pass_streak += 1
        else:
            self.command_curriculum_pass_streak = 0
        final_stage = len(self.command_curriculum_stages) - 1
        if (
            self.command_curriculum_stage < final_stage
            and self.command_curriculum_pass_streak
            >= int(self.cfg.commands.curriculum_required_passes)
        ):
            self.command_curriculum_stage += 1
            self.command_curriculum_pass_streak = 0
            self._apply_staged_curriculum_ranges()
        self._reset_staged_curriculum_window()


    def _staged_curriculum_log_metrics(self):
        linear_limit, yaw_limit = self.command_curriculum_stages[
            self.command_curriculum_stage
        ]
        metrics = {
            "curriculum_stage": float(self.command_curriculum_stage),
            "curriculum_linear_limit": linear_limit,
            "curriculum_yaw_limit": yaw_limit,
            "curriculum_pass_streak": float(self.command_curriculum_pass_streak),
        }
        metrics.update(
            {
                f"curriculum_{name}": float(value)
                for name, value in self.command_curriculum_last_metrics.items()
                if name not in ("passed", "enough_samples")
            }
        )
        metrics["curriculum_window_passed"] = float(
            self.command_curriculum_last_metrics["passed"]
        )
        return metrics


