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

"""Per-task rigid-body randomization; preserve Torch/NumPy sampling order."""
from dataclasses import dataclass
from typing import Optional
import numpy as np
import torch


@dataclass
class BodyRandomizationState:
    mass: torch.Tensor
    com: torch.Tensor
    added_mass: Optional[torch.Tensor] = None


def randomize_body_properties(props, env_id, *, config, num_envs, device, state):
    # if env_id==0:
    #     sum = 0
    #     for i, p in enumerate(props):
    #         sum += p.mass
    #         print(f"Mass of body {i}: {p.mass} (before randomization)")
    #     print(f"Total mass {sum} (before randomization)")
    # randomize base mass
    if config.randomize_base_mass:
        if env_id == 0:
            min_add_mass, max_add_mass = config.added_mass_range
            state.added_mass = (
                torch.rand(
                    num_envs,
                    dtype=torch.float,
                    device=device,
                    requires_grad=False,
                )
                * (max_add_mass - min_add_mass)
                + min_add_mass
            )
            state.mass = props[0].mass + state.added_mass
        props[0].mass += state.added_mass[env_id]
    else:
        state.mass[:] = props[0].mass
    if config.randomize_base_com:
        if env_id == 0:
            com_x, com_y, com_z = config.rand_com_vec
            state.com[:, 0] = (
                torch.rand(
                    num_envs,
                    dtype=torch.float,
                    device=device,
                    requires_grad=False,
                )
                * (com_x * 2)
                - com_x
            )
            state.com[:, 1] = (
                torch.rand(
                    num_envs,
                    dtype=torch.float,
                    device=device,
                    requires_grad=False,
                )
                * (com_y * 2)
                - com_y
            )
            state.com[:, 2] = (
                torch.rand(
                    num_envs,
                    dtype=torch.float,
                    device=device,
                    requires_grad=False,
                )
                * (com_z * 2)
                - com_z
            )
        props[0].com.x += state.com[env_id, 0]
        props[0].com.y += state.com[env_id, 1]
        props[0].com.z += state.com[env_id, 2]
    if config.randomize_inertia:
        for i in range(len(props)):
            low_bound, high_bound = config.randomize_inertia_range
            inertia_scale = np.random.uniform(low_bound, high_bound)
            props[i].mass *= inertia_scale
            props[i].inertia.x.x *= inertia_scale
            props[i].inertia.y.y *= inertia_scale
            props[i].inertia.z.z *= inertia_scale
    return props

