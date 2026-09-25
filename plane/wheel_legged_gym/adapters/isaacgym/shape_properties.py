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

"""Shape material randomization with explicit per-task coefficient storage."""
from dataclasses import dataclass
from isaacgym.torch_utils import torch_rand_float
import torch


@dataclass
class ShapeRandomizationState:
    friction: torch.Tensor
    restitution: torch.Tensor


def randomize_shape_properties(props, env_id, *, config, num_envs, device, state):
    """Callback allowing to store/change/randomize the rigid shape properties of each environment.
        Called During environment creation.
        Base behavior: randomizes the friction of each environment

    Args:
        props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
        env_id (int): Environment id

    Returns:
        [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
    """
    if config.randomize_friction:
        if env_id == 0:
            # prepare friction randomization
            friction_range = config.friction_range
            num_buckets = 64
            bucket_ids = torch.randint(0, num_buckets, (num_envs, 1))
            friction_buckets = torch_rand_float(
                friction_range[0],
                friction_range[1],
                (num_buckets, 1),
                device=device,
            )
            state.friction = friction_buckets[bucket_ids]

        for s in range(len(props)):
            props[s].friction = state.friction[env_id]
    if config.randomize_restitution:
        if env_id == 0:
            (
                min_restitution,
                max_restitution,
            ) = config.restitution_range
            state.restitution = (
                torch.rand(
                    num_envs,
                    dtype=torch.float,
                    device=device,
                    requires_grad=False,
                )
                * (max_restitution - min_restitution)
                + min_restitution
            )
        for s in range(len(props)):
            props[s].restitution = state.restitution[env_id]
    return props

