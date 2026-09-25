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


"""Create Isaac Gym ground geometry; environment owns returned height tensors."""
from isaacgym import gymapi
import torch

def create_trimesh(gym, sim, material, terrain, device):
    """Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
    #"""
    tm_params = gymapi.TriangleMeshParams()
    tm_params.nb_vertices = terrain.vertices.shape[0]
    tm_params.nb_triangles = terrain.triangles.shape[0]

    tm_params.transform.p.x = -terrain.cfg.border_size
    tm_params.transform.p.y = -terrain.cfg.border_size
    tm_params.transform.p.z = 0.0
    tm_params.static_friction = material.static_friction
    tm_params.dynamic_friction = material.dynamic_friction
    tm_params.restitution = material.restitution
    gym.add_triangle_mesh(
        sim,
        terrain.vertices.flatten(order="C"),
        terrain.triangles.flatten(order="C"),
        tm_params,
    )
    return (
        torch.tensor(terrain.heightsamples)
        .view(terrain.tot_rows, terrain.tot_cols)
        .to(device)
    )


def create_heightfield(gym, sim, material, terrain, device):
    """Adds a heightfield terrain to the simulation, sets parameters based on the cfg."""
    hf_params = gymapi.HeightFieldParams()
    hf_params.column_scale = terrain.cfg.horizontal_scale
    hf_params.row_scale = terrain.cfg.horizontal_scale
    hf_params.vertical_scale = terrain.cfg.vertical_scale
    hf_params.nbRows = terrain.tot_cols
    hf_params.nbColumns = terrain.tot_rows
    hf_params.transform.p.x = -terrain.cfg.border_size
    hf_params.transform.p.y = -terrain.cfg.border_size
    hf_params.transform.p.z = 0.0
    hf_params.static_friction = material.static_friction
    hf_params.dynamic_friction = material.dynamic_friction
    hf_params.restitution = material.restitution

    gym.add_heightfield(sim, terrain.heightsamples, hf_params)
    return (
        torch.tensor(terrain.heightsamples)
        .view(terrain.tot_rows, terrain.tot_cols)
        .to(device)
    )


def create_ground_plane(gym, sim, material):
    """Adds a ground plane to the simulation, sets friction and restitution based on the cfg."""
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
    plane_params.static_friction = material.static_friction
    plane_params.dynamic_friction = material.dynamic_friction
    plane_params.restitution = material.restitution
    gym.add_ground(sim, plane_params)


