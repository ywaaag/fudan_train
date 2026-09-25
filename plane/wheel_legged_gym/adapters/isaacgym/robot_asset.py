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

"""Isaac asset loading and metadata; actor creation and randomization stay with caller."""
import os
from dataclasses import dataclass
from isaacgym import gymapi


@dataclass(frozen=True)
class LoadedRobotAsset:
    handle: object
    num_dof: int
    num_bodies: int
    num_dofs: int
    dof_properties: object
    rigid_shape_properties: object
    body_names: list
    dof_names: list


def load_robot_asset(gym, sim, asset_cfg, package_root):
    asset_path = asset_cfg.file.format(
        WHEEL_LEGGED_GYM_ROOT_DIR=package_root
    )
    asset_root = os.path.dirname(asset_path)
    asset_file = os.path.basename(asset_path)

    asset_options = gymapi.AssetOptions()
    asset_options.default_dof_drive_mode = asset_cfg.default_dof_drive_mode
    asset_options.collapse_fixed_joints = asset_cfg.collapse_fixed_joints
    asset_options.replace_cylinder_with_capsule = (
        asset_cfg.replace_cylinder_with_capsule
    )
    asset_options.flip_visual_attachments = asset_cfg.flip_visual_attachments
    asset_options.fix_base_link = asset_cfg.fix_base_link
    asset_options.density = asset_cfg.density
    asset_options.angular_damping = asset_cfg.angular_damping
    asset_options.linear_damping = asset_cfg.linear_damping
    asset_options.max_angular_velocity = asset_cfg.max_angular_velocity
    asset_options.max_linear_velocity = asset_cfg.max_linear_velocity
    asset_options.armature = asset_cfg.armature
    asset_options.thickness = asset_cfg.thickness
    asset_options.disable_gravity = asset_cfg.disable_gravity

    robot_asset = gym.load_asset(
        sim, asset_root, asset_file, asset_options
    )
    num_dof = gym.get_asset_dof_count(robot_asset)
    num_bodies = gym.get_asset_rigid_body_count(robot_asset)
    dof_props_asset = gym.get_asset_dof_properties(robot_asset)
    rigid_shape_props_asset = gym.get_asset_rigid_shape_properties(robot_asset)

    # save body names from the asset
    body_names = gym.get_asset_rigid_body_names(robot_asset)
    dof_names = gym.get_asset_dof_names(robot_asset)
    num_bodies = len(body_names)
    num_dofs = len(dof_names)
    return LoadedRobotAsset(
        robot_asset, num_dof, num_bodies, num_dofs, dof_props_asset,
        rigid_shape_props_asset, body_names, dof_names,
    )
