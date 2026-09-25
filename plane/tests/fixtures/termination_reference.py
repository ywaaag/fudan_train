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

"""Frozen termination logic before domain extraction."""
import torch

class OriginalTermination:
    def check_termination(self):
        """Check if environments need to be reset"""
        if self._method_v1:
            forbidden = torch.any(
                torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1)
                > float(self.cfg.rewards.forbidden_contact_force_threshold),
                dim=1,
            ) if self.termination_contact_indices.numel() else torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
            upright_bad = self.projected_gravity[:, 2] > -0.1
            warmup = self.episode_length_buf < (
                float(self.cfg.rewards.contact_warmup_s) / self.dt
            )
            wheel_loss = (
                self.wheel_contact_seen
                & ~warmup
                & (self.wheel_contact_history[:, 0].sum(dim=1) < 1.0)
                & (getattr(self.cfg.commands, "training_phase", "legacy") != "stand")
            )
            self.forbidden_contact_streak = torch.where(
                forbidden, self.forbidden_contact_streak + 1.0, torch.zeros_like(self.forbidden_contact_streak)
            )
            self.upright_streak = torch.where(
                upright_bad, self.upright_streak + 1.0, torch.zeros_like(self.upright_streak)
            )
            self.wheel_loss_streak = torch.where(
                wheel_loss, self.wheel_loss_streak + 1.0, torch.zeros_like(self.wheel_loss_streak)
            )
            forbidden_limit = float(self.cfg.rewards.forbidden_contact_grace_s) / self.dt
            wheel_limit = float(self.cfg.rewards.wheel_loss_grace_s) / self.dt
            fail_buf = (
                (self.forbidden_contact_streak >= forbidden_limit)
                | (self.upright_streak >= forbidden_limit)
                | (self.wheel_loss_streak >= wheel_limit)
            )
        else:
            fail_buf = torch.any(
                torch.norm(
                    self.contact_forces[:, self.termination_contact_indices, :], dim=-1
                )
                > 10.0,
                dim=1,
            )
            fail_buf |= self.projected_gravity[:, 2] > -0.1
        self.fail_buf *= fail_buf
        self.fail_buf += fail_buf
        self.time_out_buf = (
            self.episode_length_buf > self.max_episode_length
        )  # no terminal reward for time-outs
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.edge_reset_buf = self.base_position[:, 0] > self.terrain_x_max - 1
            self.edge_reset_buf |= self.base_position[:, 0] < self.terrain_x_min + 1
            self.edge_reset_buf |= self.base_position[:, 1] > self.terrain_y_max - 1
            self.edge_reset_buf |= self.base_position[:, 1] < self.terrain_y_min + 1
        if self._method_v1:
            self.reset_buf = fail_buf | self.time_out_buf | self.edge_reset_buf
        else:
            self.reset_buf = (
                (self.fail_buf > self.cfg.env.fail_to_terminal_time_s / self.dt)
                | self.time_out_buf
                | self.edge_reset_buf
            )

