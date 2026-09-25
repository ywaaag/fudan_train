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

"""Training-only privileged observation; never the deployed actor input."""
import torch


def privileged_observation(*, root_height, measured_heights, linear_velocity,
                           clean_policy_observation, action_history, acceleration,
                           torques, mass, center_of_mass, default_position,
                           raw_default_position, friction, restitution, scales,
                           num_environments):
    """Assemble critic channels before policy noise, preserving batch mass mean.

    Inputs remain owned by the environment. Concatenation creates a separate
    tensor, so later in-place noise on actor observations cannot affect critic.
    """
    heights = (
        torch.clip(root_height.unsqueeze(1) - 0.5 - measured_heights, -1, 1.0)
        * scales.height_measurements
    )
    return torch.cat(
        (
            linear_velocity * scales.lin_vel,
            clean_policy_observation,
            action_history[:, :, 0],
            action_history[:, :, 1],
            acceleration * scales.dof_acc,
            heights,
            torques * scales.torque,
            (mass - mass.mean()).view(num_environments, 1),
            center_of_mass,
            default_position - raw_default_position,
            friction.view(num_environments, 1),
            restitution.view(num_environments, 1),
        ),
        dim=-1,
    )
