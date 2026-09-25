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

"""Create actor instances with explicitly ordered randomization callbacks."""
from isaacgym import gymapi
from isaacgym.torch_utils import torch_rand_float
import numpy as np


def create_actor_instances(*, gym, sim, robot_asset, num_envs, origins, device,
                           start_pose, env_lower, env_upper, asset_name, self_collisions,
                           rigid_shape_props_asset, dof_props_asset,
                           process_shapes, process_dofs, process_bodies,
                           environments, actors):
    """Append handles after all properties are applied, retaining caller list identity.

    Property callbacks may update per-environment randomization buffers. They run
    at the original points in the loop; do not precompute or batch these calls.
    """
    for i in range(num_envs):
        # create env instance
        env_handle = gym.create_env(
            sim, env_lower, env_upper, int(np.sqrt(num_envs))
        )
        pos = origins[i].clone()
        pos[:2] += torch_rand_float(-1.0, 1.0, (2, 1), device=device).squeeze(
            1
        )
        start_pose.p = gymapi.Vec3(*pos)

        rigid_shape_props = process_shapes(
            rigid_shape_props_asset, i
        )
        gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
        actor_handle = gym.create_actor(
            env_handle,
            robot_asset,
            start_pose,
            asset_name,
            i,
            self_collisions,
            0,
        )
        dof_props = process_dofs(dof_props_asset, i)
        gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
        body_props = gym.get_actor_rigid_body_properties(
            env_handle, actor_handle
        )
        body_props = process_bodies(body_props, i)
        gym.set_actor_rigid_body_properties(
            env_handle, actor_handle, body_props, recomputeInertia=True
        )
        environments.append(env_handle)
        actors.append(actor_handle)

