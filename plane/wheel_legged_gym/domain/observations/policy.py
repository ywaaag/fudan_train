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

"""25D policy input in the frozen deployment channel order."""
import torch


def proprioception(*, angular_velocity, projected_gravity, commands, command_scale,
                   position, default_position, velocity, actions, scales):
    """Build a clean observation; previous actions are already in policy units."""
    return torch.cat(
        (
            angular_velocity * scales.ang_vel,
            projected_gravity,
            commands[:, :3] * command_scale,
            (position[:, [0, 1, 3, 4]] - default_position[:, [0, 1, 3, 4]])
            * scales.dof_pos,
            velocity * scales.dof_vel,
            actions,
        ),
        dim=-1,
    )


def noise_scale_vector(template, *, noise_scales, noise_level, observation_scales,
                       measure_heights):
    """Preserve historical slices, including the inactive 48:235 height slice.

    Height noise is empty for the 25D actor input. Changing that contract is a
    separate behavior change, not part of the architecture migration.
    """
    noise_vec = torch.zeros_like(template)
    noise_vec[:3] = noise_scales.ang_vel * noise_level * observation_scales.ang_vel
    noise_vec[3:6] = noise_scales.gravity * noise_level
    noise_vec[6:9] = 0.0
    noise_vec[9:13] = noise_scales.dof_pos * noise_level * observation_scales.dof_pos
    noise_vec[13:19] = noise_scales.dof_vel * noise_level * observation_scales.dof_vel
    noise_vec[19:25] = 0.0
    if measure_heights:
        noise_vec[48:235] = (
            noise_scales.height_measurements * noise_level
            * observation_scales.height_measurements
        )
    return noise_vec


def update_history(history, observation, *, environment_steps, policy_decimation,
                   history_decimation, observation_width):
    """Update selected rows in place, oldest to newest; return value is unused."""
    update_idx = (
        (environment_steps / policy_decimation) % history_decimation
    ) == 0
    history[update_idx, :] = torch.cat(
        (history[update_idx, observation_width:], observation[update_idx, :]),
        dim=-1,
    )
