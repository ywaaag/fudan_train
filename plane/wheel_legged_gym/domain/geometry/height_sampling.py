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

"""Terrain height sampling arithmetic, independent of simulator handles."""
import torch
from wheel_legged_gym.domain.geometry.rotations import quat_apply_yaw


def sample_heights(env_ids=None, *, mesh_type, num_envs, num_height_points, device,
                   base_quat, root_states, height_points, height_samples, terrain_config):
    """Samples heights of the terrain at required points around each robot.
        The points are offset by the base's position and rotated by the base's yaw

    Args:
        env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

    Raises:
        NameError: [description]

    Returns:
        [type]: [description]
    """
    if mesh_type == "plane":
        return torch.zeros(
            num_envs,
            num_height_points,
            device=device,
            requires_grad=False,
        )
    elif mesh_type == "none":
        raise NameError("Can't measure height with terrain mesh type 'none'")

    if env_ids:
        points = quat_apply_yaw(
            base_quat[env_ids].repeat(1, num_height_points),
            height_points[env_ids],
        ) + (root_states[env_ids, :3]).unsqueeze(1)
    else:
        points = quat_apply_yaw(
            base_quat.repeat(1, num_height_points), height_points
        ) + (root_states[:, :3]).unsqueeze(1)

    points += terrain_config.border_size
    points = (points / terrain_config.horizontal_scale).long()
    px = points[:, :, 0].view(-1)
    py = points[:, :, 1].view(-1)
    px = torch.clip(px, 0, height_samples.shape[0] - 2)
    py = torch.clip(py, 0, height_samples.shape[1] - 2)

    heights1 = height_samples[px, py]
    heights2 = height_samples[px + 1, py]
    heights3 = height_samples[px, py + 1]
    heights = torch.min(heights1, heights2)
    heights = torch.min(heights, heights3)

    return heights.view(num_envs, -1) * terrain_config.vertical_scale

