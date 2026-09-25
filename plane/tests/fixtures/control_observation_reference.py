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


"""Frozen pre-extraction methods; do not refresh from production code.

Source SHA256: cbbafa5b0cf75d9e7bbf6d58faea12fd4c5e4bacc24fa60fa9ff14ec2db6a526
"""
import torch


class OriginalControlObservations:
    def compute_proprioception_observations(self):
            # note that observation noise need to modified accordingly !!!
            obs_buf = torch.cat(
                (
                    # self.base_lin_vel * self.obs_scales.lin_vel,
                    self.base_ang_vel * self.obs_scales.ang_vel,
                    self.projected_gravity,
                    self.commands[:, :3] * self.commands_scale,
                    (
                        self.dof_pos[:, [0, 1, 3, 4]]
                        - self.default_dof_pos[:, [0, 1, 3, 4]]
                    )
                    * self.obs_scales.dof_pos,
                    self.dof_vel * self.obs_scales.dof_vel,
                    self.actions,
                ),
                dim=-1,
            )
            return obs_buf

    def compute_observations(self):
            """Computes observations"""
            self.obs_buf = self.compute_proprioception_observations()

            if self.cfg.env.num_privileged_obs is not None:
                heights = (
                    torch.clip(
                        self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights,
                        -1,
                        1.0,
                    )
                    * self.obs_scales.height_measurements
                )
                self.privileged_obs_buf = torch.cat(
                    (
                        self.base_lin_vel * self.obs_scales.lin_vel,
                        self.obs_buf,
                        self.last_actions[:, :, 0],
                        self.last_actions[:, :, 1],
                        self.dof_acc * self.obs_scales.dof_acc,
                        heights,
                        self.torques * self.obs_scales.torque,
                        (self.base_mass - self.base_mass.mean()).view(self.num_envs, 1),
                        self.base_com,
                        self.default_dof_pos - self.raw_default_dof_pos,
                        self.friction_coef.view(self.num_envs, 1),
                        self.restitution_coef.view(self.num_envs, 1),
                    ),
                    dim=-1,
                )

            # add noise if needed
            if self.add_noise:
                self.obs_buf += (
                    2 * torch.rand_like(self.obs_buf) - 1
                ) * self.noise_scale_vec

            update_idx = (
                (self.envs_steps_buf / self.cfg.control.decimation)
                % self.cfg.env.obs_history_dec
            ) == 0
            self.obs_history[update_idx, :] = torch.cat(
                (self.obs_history[update_idx, self.num_obs :], self.obs_buf[update_idx, :]),
                dim=-1,
            )

    def _compute_torques(self, actions):
            """Compute torques from actions.
                Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
                [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

            Args:
                actions (torch.Tensor): Actions

            Returns:
                [torch.Tensor]: Torques sent to the simulation
            """
            # pd controller
            pos_ref = actions * self.cfg.control.pos_action_scale
            pos_ref[:, 2] *= 0
            pos_ref[:, 5] *= 0
            vel_ref = actions * self.cfg.control.vel_action_scale
            vel_ref[:, :2] *= 0
            vel_ref[:, 3:5] *= 0
            torques = self.p_gains * (
                pos_ref + self.default_dof_pos - self.dof_pos
            ) + self.d_gains * (vel_ref - self.dof_vel)
            raw_torques = torques * self.torques_scale
            if self._method_v1:
                self.command_metric_preclip_torque_saturation_sum += (
                    torch.abs(raw_torques) >= self.torque_limits * 0.99
                ).float().mean(dim=1) / float(self.cfg.control.decimation)
            return torch.clip(raw_torques, -self.torque_limits, self.torque_limits)

    def _get_noise_scale_vec(self, cfg):
            """Sets a vector used to scale the noise added to the observations.
                [NOTE]: Must be adapted when changing the observations structure

            Args:
                cfg (Dict): Environment config file

            Returns:
                [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
            """
            noise_vec = torch.zeros_like(self.obs_buf[0])
            self.add_noise = self.cfg.noise.add_noise
            noise_scales = self.cfg.noise.noise_scales
            noise_level = self.cfg.noise.noise_level
            # noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
            # noise_vec[3 : 3 + 3] = (
            #     noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
            # )
            # noise_vec[3 + 3 : 6 + 3] = noise_scales.gravity * noise_level
            # noise_vec[6 + 3 : 8 + 3] = 0.0  # commands
            # noise_vec[8 + 3 : 14 + 3] = (
            #     noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
            # )
            # noise_vec[14 + 3 : 20 + 3] = (
            #     noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
            # )
            # noise_vec[20 + 3 : 26 + 3] = 0.0  # previous actions
            noise_vec[:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
            noise_vec[3:6] = noise_scales.gravity * noise_level
            noise_vec[6:9] = 0.0  # commands
            noise_vec[9:13] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
            noise_vec[13:19] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
            noise_vec[19:25] = 0.0  # previous actions
            if self.cfg.terrain.measure_heights:
                noise_vec[48:235] = (
                    noise_scales.height_measurements
                    * noise_level
                    * self.obs_scales.height_measurements
                )
            return noise_vec
