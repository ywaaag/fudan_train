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

"""Read configured joint limits and apply explicit asset armature name rules."""
import torch


def read_dof_limits(props, *, num_dof, device, soft_position_limit):
    position_limits = torch.zeros(
        num_dof, 2, dtype=torch.float, device=device, requires_grad=False
    )
    velocity_limits = torch.zeros(
        num_dof, dtype=torch.float, device=device, requires_grad=False
    )
    torque_limits = torch.zeros(
        num_dof, dtype=torch.float, device=device, requires_grad=False
    )

    for i in range(len(props)):
        position_limits[i, 0] = props['lower'][i].item()
        position_limits[i, 1] = props['upper'][i].item()
        velocity_limits[i] = props['velocity'][i].item()
        torque_limits[i] = props['effort'][i].item()

        m = (position_limits[i, 0] + position_limits[i, 1]) / 2
        r = position_limits[i, 1] - position_limits[i, 0]
        position_limits[i, 0] = m - 0.5 * r * soft_position_limit
        position_limits[i, 1] = m + 0.5 * r * soft_position_limit
    return position_limits, velocity_limits, torque_limits


def apply_armature(props, dof_names, armature_by_name):
    for i in range(len(props)):
        name = dof_names[i]
        props["armature"][i] = 0.0
        for key, val in armature_by_name.items():
            if key in name:
                props["armature"][i] = val
                print(props["armature"][i])
                break

    return props
