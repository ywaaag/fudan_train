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

"""Indexed physical-state resets; caller owns buffers and episode bookkeeping."""
from isaacgym import gymtorch
from isaacgym.torch_utils import torch_rand_float
import torch


def reset_dofs(env_ids, *, gym, sim, dof_position, dof_velocity, default_position, dof_state):
    """Resets DOF position and velocities of selected environmments
    Positions use the already-configured per-environment default positions.
    Velocities are set to zero.

    Args:
        env_ids (List[int]): Environemnt ids
    """
    dof_position[env_ids] = default_position[env_ids, :]
    dof_velocity[env_ids] = 0.0

    env_ids_int32 = env_ids.to(dtype=torch.int32)
    gym.set_dof_state_tensor_indexed(
        sim,
        gymtorch.unwrap_tensor(dof_state),
        gymtorch.unwrap_tensor(env_ids_int32),
        len(env_ids_int32),
    )

def reset_root_states(env_ids, *, gym, sim, root_states, initial_state, origins, custom_origins, device):
    """Resets ROOT states position and velocities of selected environmments
        Sets base position based on the curriculum
        Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
    Args:
        env_ids (List[int]): Environemnt ids
    """
    # base position
    if custom_origins:
        root_states[env_ids] = initial_state
        root_states[env_ids, :3] += origins[env_ids]
        root_states[env_ids, :2] += torch_rand_float(
            -1.0, 1.0, (len(env_ids), 2), device=device
        )  # xy position within 1m of the center
    else:
        root_states[env_ids] = initial_state
        root_states[env_ids, :3] += origins[env_ids]
    # base velocities
    root_states[env_ids, 7:13] = torch_rand_float(
        -0.5, 0.5, (len(env_ids), 6), device=device
    )  # [7:10]: lin vel, [10:13]: ang vel
    env_ids_int32 = env_ids.to(dtype=torch.int32)
    gym.set_actor_root_state_tensor_indexed(
        sim,
        gymtorch.unwrap_tensor(root_states),
        gymtorch.unwrap_tensor(env_ids_int32),
        len(env_ids_int32),
    )

