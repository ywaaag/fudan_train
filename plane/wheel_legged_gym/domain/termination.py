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

"""Termination state transitions, independent of simulator/reset side effects."""
from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class TerminationState:
    forbidden_contact_streak: Optional[torch.Tensor]
    upright_streak: Optional[torch.Tensor]
    wheel_loss_streak: Optional[torch.Tensor]
    fail_buf: Optional[torch.Tensor]
    time_out_buf: Optional[torch.Tensor]
    edge_reset_buf: Optional[torch.Tensor]
    reset_buf: Optional[torch.Tensor]


def check_termination(state, *, method_v1, contact_forces, contact_indices,
                      projected_gravity, episode_lengths, max_episode_length,
                      wheel_contact_seen, wheel_contact_history, base_position,
                      terrain_bounds, cfg, dt, num_envs, device):
    """Check if environments need to be reset"""
    if method_v1:
        forbidden = torch.any(
            torch.norm(contact_forces[:, contact_indices, :], dim=-1)
            > float(cfg.rewards.forbidden_contact_force_threshold),
            dim=1,
        ) if contact_indices.numel() else torch.zeros(
            num_envs, dtype=torch.bool, device=device
        )
        upright_bad = projected_gravity[:, 2] > -0.1
        warmup = episode_lengths < (
            float(cfg.rewards.contact_warmup_s) / dt
        )
        wheel_loss = (
            wheel_contact_seen
            & ~warmup
            & (wheel_contact_history[:, 0].sum(dim=1) < 1.0)
            & (getattr(cfg.commands, "training_phase", "legacy") != "stand")
        )
        state.forbidden_contact_streak = torch.where(
            forbidden, state.forbidden_contact_streak + 1.0, torch.zeros_like(state.forbidden_contact_streak)
        )
        state.upright_streak = torch.where(
            upright_bad, state.upright_streak + 1.0, torch.zeros_like(state.upright_streak)
        )
        state.wheel_loss_streak = torch.where(
            wheel_loss, state.wheel_loss_streak + 1.0, torch.zeros_like(state.wheel_loss_streak)
        )
        forbidden_limit = float(cfg.rewards.forbidden_contact_grace_s) / dt
        wheel_limit = float(cfg.rewards.wheel_loss_grace_s) / dt
        fail_buf = (
            (state.forbidden_contact_streak >= forbidden_limit)
            | (state.upright_streak >= forbidden_limit)
            | (state.wheel_loss_streak >= wheel_limit)
        )
    else:
        fail_buf = torch.any(
            torch.norm(
                contact_forces[:, contact_indices, :], dim=-1
            )
            > 10.0,
            dim=1,
        )
        fail_buf |= projected_gravity[:, 2] > -0.1
    state.fail_buf *= fail_buf
    state.fail_buf += fail_buf
    state.time_out_buf = (
        episode_lengths > max_episode_length
    )  # no terminal reward for time-outs
    if cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
        state.edge_reset_buf = base_position[:, 0] > terrain_bounds.x_max - 1
        state.edge_reset_buf |= base_position[:, 0] < terrain_bounds.x_min + 1
        state.edge_reset_buf |= base_position[:, 1] > terrain_bounds.y_max - 1
        state.edge_reset_buf |= base_position[:, 1] < terrain_bounds.y_min + 1
    if method_v1:
        state.reset_buf = fail_buf | state.time_out_buf | state.edge_reset_buf
    else:
        state.reset_buf = (
            (state.fail_buf > cfg.env.fail_to_terminal_time_s / dt)
            | state.time_out_buf
            | state.edge_reset_buf
        )

