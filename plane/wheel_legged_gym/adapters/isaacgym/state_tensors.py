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

"""Acquire simulator-owned state tensors and retain aliasing of their views."""
from dataclasses import dataclass
from isaacgym import gymtorch
import torch


@dataclass(frozen=True)
class SimulationTensors:
    """Frozen bindings, mutable shared tensors; only dof_acc is independently allocated."""
    root_states: torch.Tensor
    dof_state: torch.Tensor
    dof_pos: torch.Tensor
    dof_vel: torch.Tensor
    dof_acc: torch.Tensor
    base_quat: torch.Tensor
    contact_forces: torch.Tensor
    rigid_body_states: torch.Tensor


def acquire_state_tensors(gym, sim, *, num_envs, num_dof, num_bodies):
    # get gym GPU state tensors
    actor_root_state = gym.acquire_actor_root_state_tensor(sim)
    dof_state_tensor = gym.acquire_dof_state_tensor(sim)
    net_contact_forces = gym.acquire_net_contact_force_tensor(sim)
    rigid_body_state_tensor = gym.acquire_rigid_body_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)
    gym.refresh_actor_root_state_tensor(sim)
    gym.refresh_net_contact_force_tensor(sim)
    gym.refresh_rigid_body_state_tensor(sim)

    # create some wrapper tensors for different slices
    root_states = gymtorch.wrap_tensor(actor_root_state)
    dof_state = gymtorch.wrap_tensor(dof_state_tensor)
    dof_pos = dof_state.view(num_envs, num_dof, 2)[..., 0]
    dof_vel = dof_state.view(num_envs, num_dof, 2)[..., 1]
    dof_acc = torch.zeros_like(dof_vel)
    base_quat = root_states[:, 3:7]

    contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(
        num_envs, -1, 3
    )  # shape: num_envs, num_bodies, xyz axis
    rigid_body_states = gymtorch.wrap_tensor(rigid_body_state_tensor).view(
        num_envs, num_bodies, 13
    )

    return SimulationTensors(
        root_states=root_states,
        dof_state=dof_state,
        dof_pos=dof_pos,
        dof_vel=dof_vel,
        dof_acc=dof_acc,
        base_quat=base_quat,
        contact_forces=contact_forces,
        rigid_body_states=rigid_body_states,
    )
