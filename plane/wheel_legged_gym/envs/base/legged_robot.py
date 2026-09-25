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

from wheel_legged_gym import WHEEL_LEGGED_GYM_ROOT_DIR
from time import time
from warnings import WarningMessage
import numpy as np
import os

from isaacgym.torch_utils import (
    get_axis_params, quat_apply, quat_rotate, quat_rotate_inverse, to_torch, torch_rand_float,
)
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict

from wheel_legged_gym import WHEEL_LEGGED_GYM_ROOT_DIR
from wheel_legged_gym.adapters.isaacgym.base_task import BaseTask
from wheel_legged_gym.adapters.isaacgym.terrain_generation import Terrain
from wheel_legged_gym.adapters.isaacgym.asset_indices import build_asset_indices
from wheel_legged_gym.adapters.isaacgym.robot_asset import load_robot_asset
from wheel_legged_gym.adapters.isaacgym.actor_creation import create_actor_instances
from wheel_legged_gym.adapters.isaacgym.state_tensors import acquire_state_tensors
from wheel_legged_gym.adapters.isaacgym.state_reset import reset_dofs, reset_root_states
from wheel_legged_gym.adapters.isaacgym.dof_properties import read_dof_limits, apply_armature
from wheel_legged_gym.adapters.isaacgym.shape_properties import (
    ShapeRandomizationState, randomize_shape_properties,
)
from wheel_legged_gym.adapters.isaacgym.body_properties import (
    BodyRandomizationState, randomize_body_properties,
)
from wheel_legged_gym.domain.geometry.rotations import (
    quat_apply_yaw,
    wrap_to_pi,
)
from wheel_legged_gym.contracts.config_serialization import class_to_dict
from wheel_legged_gym.domain.geometry.origins import build_origin_layout
from wheel_legged_gym.domain.control.initialization import initialize as initialize_control_state
from wheel_legged_gym.domain.commands.resampling import CommandBuffers, resample as resample_commands
from wheel_legged_gym.domain.rewards.terms import (
    CMD_HEIGHT,
    CMD_VX,
    CMD_YAW,
    capped_tracking_terms,
    normalized_huber,
    smooth_gate,
    wheel_rolling_terms,
    bilateral_geometry_cost,
)
from wheel_legged_gym.domain.commands.command_curriculum import (
    evaluate_curriculum_window,
    validate_curriculum_stages,
)
from wheel_legged_gym.contracts.legged_robot_config import LeggedRobotCfg


from wheel_legged_gym.domain.rewards.api import evaluate as evaluate_reward
from types import SimpleNamespace
from wheel_legged_gym.domain.termination import TerminationState, check_termination as check_termination_state
from wheel_legged_gym.domain.geometry.height_sampling import sample_heights
from wheel_legged_gym.domain.rewards.inputs import RewardInputs
from wheel_legged_gym.domain.commands.metrics_state import create_command_metrics
from wheel_legged_gym.domain.commands.curriculum_window import create_window, accumulate_window
from wheel_legged_gym.domain.commands.tracking_metrics import (
    accumulate as accumulate_command_metrics, summarize as summarize_command_metrics,
)
from wheel_legged_gym.domain.commands.staged_progress import (
    window_is_due, advance_stage, log_metrics as staged_log_metrics,
)
from wheel_legged_gym.domain.commands.checkpoint_state import (
    method_state, restore_method_counter, staged_state, restored_stage,
)
from wheel_legged_gym.domain.control.actuation import mixed_pd_torques
from wheel_legged_gym.domain.observations.critic import privileged_observation
from wheel_legged_gym.adapters.isaacgym.terrain_creation import (
    create_ground_plane, create_heightfield, create_trimesh,
)
from wheel_legged_gym.domain.observations.policy import (
    proprioception, noise_scale_vector, update_history,
)


class LeggedRobot(BaseTask):
    def __init__(
        self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless
    ):
        """Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)
        self.pi = torch.acos(torch.zeros(1, device=self.device)) * 2

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._method_v1 = getattr(self.cfg.rewards, "reward_pipeline", "legacy_v0") == "normalized_v1"
        self._prepare_reward_function()
        self.init_done = True

    def step(self, actions):
        """Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        clip_actions = self.cfg.normalization.clip_actions
        self.command_metrics.action_clip_sum[:] += (
            torch.abs(actions) > clip_actions
        ).float().mean(dim=1)
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # step physics and render each frame
        self.render()
        self.pre_physics_step()
        for _ in range(self.cfg.control.decimation):
            self.envs_steps_buf += 1
            self.action_fifo = torch.cat(
                (self.actions.unsqueeze(1), self.action_fifo[:, :-1, :]), dim=1
            )
            self.torques = self._compute_torques(
                self.action_fifo[torch.arange(self.num_envs), self.action_delay_idx, :]
            ).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(
                self.sim, gymtorch.unwrap_tensor(self.torques)
            )
            if self.cfg.domain_rand.push_robots:
                self._push_robots()
            self.gym.simulate(self.sim)
            if self.device == "cpu":
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.compute_dof_vel()
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(
                self.privileged_obs_buf, -clip_obs, clip_obs
            )
        return (
            self.obs_buf,
            self.privileged_obs_buf,
            self.rew_buf,
            self.reset_buf,
            self.extras,
            self.obs_history,
        )

    def compute_dof_vel(self):
        diff = (
            torch.remainder(self.dof_pos - self.last_dof_pos + self.pi, 2 * self.pi)
            - self.pi
        )
        self.dof_pos_dot = diff / self.sim_params.dt

        if self.cfg.env.dof_vel_use_pos_diff:
            self.dof_vel = self.dof_pos_dot

        self.last_dof_pos[:] = self.dof_pos[:]

    def post_physics_step(self):
        """check terminations, compute observations and rewards
        calls self._post_physics_step_callback() for common computations
        calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        if self._method_v1 and self.feet_indices.numel() == 2:
            contact_now = torch.norm(
                self.contact_forces[:, self.feet_indices, :], dim=-1
            ) > float(self.cfg.rewards.contact_force_threshold)
            # Clone the source because the slices overlap in storage.  CUDA
            # often tolerates this for large batches, while the exact
            # single-env playback path raises an aliasing error.
            self.wheel_contact_history[:, 1:] = self.wheel_contact_history[:, :-1].clone()
            self.wheel_contact_history[:, 0] = contact_now
            self.wheel_contact_seen |= contact_now.any(dim=1)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel = (self.base_position - self.last_base_position) / self.dt
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.base_lin_vel)
        self.base_ang_vel[:] = quat_rotate_inverse(
            self.base_quat, self.root_states[:, 10:13]
        )
        self.projected_gravity[:] = quat_rotate_inverse(
            self.base_quat, self.gravity_vec
        )
        self.dof_acc = (self.last_dof_vel - self.dof_vel) / self.dt

        theta1 = torch.cat(
            (self.dof_pos[:, 0].unsqueeze(1), -self.dof_pos[:, 3].unsqueeze(1)), dim=1
        )
        theta2 = torch.cat(
            (
                (self.dof_pos[:, 1] + self.pi / 2).unsqueeze(1),
                (-self.dof_pos[:, 4] + self.pi / 2).unsqueeze(1),
            ),
            dim=1,
        )
        end_x = (
            self.cfg.asset.offset
            + self.cfg.asset.l1 * torch.cos(theta1)
            + self.cfg.asset.l2 * torch.cos(theta1 + theta2)
        )
        end_y = self.cfg.asset.l1 * torch.sin(theta1) + self.cfg.asset.l2 * torch.sin(
            theta1 + theta2
        )
        self.L0 = torch.sqrt(end_x**2 + end_y**2)
        self.theta0 = torch.arctan2(end_y, end_x) - self.pi / 2

        if self._method_v1:
            # Attribute this physics step to the command that was active during
            # it.  The callback may prepare the next command for observations.
            self._accumulate_command_tracking_metrics()
            self._post_physics_step_callback()
        else:
            self._post_physics_step_callback()
            self._accumulate_command_tracking_metrics()
        # print(self.base_height)
        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations()  # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_actions[:, :, 1] = self.last_actions[:, :, 0]
        self.last_actions[:, :, 0] = self.actions[:]
        self.last_base_position[:] = self.base_position[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

    def check_termination(self):
        state = TerminationState(
            getattr(self, 'forbidden_contact_streak', None),
            getattr(self, 'upright_streak', None),
            getattr(self, 'wheel_loss_streak', None),
            getattr(self, 'fail_buf', None),
            getattr(self, 'time_out_buf', None),
            getattr(self, 'edge_reset_buf', None),
            getattr(self, 'reset_buf', None),
        )
        bounds = None
        if self.cfg.terrain.mesh_type in ['heightfield', 'trimesh']:
            bounds = SimpleNamespace(x_max=self.terrain_x_max, x_min=self.terrain_x_min,
                                     y_max=self.terrain_y_max, y_min=self.terrain_y_min)
        check_termination_state(
            state, method_v1=self._method_v1, contact_forces=self.contact_forces,
            contact_indices=self.termination_contact_indices, projected_gravity=self.projected_gravity,
            episode_lengths=self.episode_length_buf, max_episode_length=self.max_episode_length,
            wheel_contact_seen=getattr(self, 'wheel_contact_seen', None),
            wheel_contact_history=getattr(self, 'wheel_contact_history', None),
            base_position=self.base_position, terrain_bounds=bounds, cfg=self.cfg,
            dt=self.dt, num_envs=self.num_envs, device=self.device,
        )
        self.fail_buf = state.fail_buf
        self.time_out_buf = state.time_out_buf
        self.edge_reset_buf = state.edge_reset_buf
        self.reset_buf = state.reset_buf
        if self._method_v1:
            self.forbidden_contact_streak = state.forbidden_contact_streak
            self.upright_streak = state.upright_streak
            self.wheel_loss_streak = state.wheel_loss_streak

    def reset_idx(self, env_ids):
        """Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        command_metrics = summarize_command_metrics(self.command_metrics, env_ids)
        self._accumulate_staged_curriculum_window(env_ids)
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
            if self.cfg.commands.curriculum and not self._uses_staged_curriculum():
                time_out_env_ids = self.time_out_buf.nonzero(as_tuple=False).flatten()
                self.update_command_curriculum(time_out_env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and not self._uses_staged_curriculum() and (
            self.common_step_counter % self.max_episode_length == 0
        ):
            self.update_command_curriculum(env_ids)
        self._maybe_advance_staged_curriculum()

        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        self._resample_commands(env_ids)

        # reset buffers
        self.last_actions[env_ids] = 0.0
        self.last_dof_vel[env_ids] = 0.0
        self.feet_air_time[env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.fail_buf[env_ids] = 0
        if self._method_v1:
            self.forbidden_contact_streak[env_ids] = 0.0
            self.upright_streak[env_ids] = 0.0
            self.wheel_loss_streak[env_ids] = 0.0
            self.wheel_contact_history[env_ids] = False
            self.wheel_contact_seen[env_ids] = False
        self.envs_steps_buf[env_ids] = 0
        self.last_dof_pos[env_ids] = self.dof_pos[env_ids]
        self.last_base_position[env_ids] = self.base_position[env_ids]
        self.obs_history[env_ids] = 0
        obs_buf = self.compute_proprioception_observations()
        self.obs_history[env_ids] = obs_buf[env_ids].repeat(1, self.obs_history_length)
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            )
            self.episode_sums[key][env_ids] = 0.0
        self.extras["episode"].update(command_metrics)
        for mode_id, mode_name in enumerate(("zero", "small", "reverse", "forward")):
            self.extras["episode"][f"sample_{mode_name}_fraction"] = torch.mean(
                (self.command_sample_mode[env_ids] == mode_id).float()
            )
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(
                self.terrain_levels.float()
            )
        if self.cfg.commands.curriculum and not self._uses_staged_curriculum():
            self.extras["episode"]["a_flat_max_command_x"] = torch.mean(
                self.command_ranges["lin_vel_x"][self.flat_idx, 1].float()
            )
        if self._uses_staged_curriculum():
            self.extras["episode"].update(self._staged_curriculum_log_metrics())
        if self.cfg.terrain.curriculum and self.cfg.commands.curriculum:
            self.extras["episode"]["a_smooth_slope_max_command_x"] = torch.mean(
                self.command_ranges["lin_vel_x"][self.smooth_slope_idx, 1].float()
            )
            self.extras["episode"]["a_rough_slope_max_command_x"] = torch.mean(
                self.command_ranges["lin_vel_x"][self.rough_slope_idx, 1].float()
            )
            self.extras["episode"]["a_stair_up_max_command_x"] = torch.mean(
                self.command_ranges["lin_vel_x"][self.stair_up_idx, 1].float()
            )
            self.extras["episode"]["a_stair_down_max_command_x"] = torch.mean(
                self.command_ranges["lin_vel_x"][self.stair_down_idx, 1].float()
            )
            self.extras["episode"]["a_discrete_max_command_x"] = torch.mean(
                self.command_ranges["lin_vel_x"][self.discrete_idx, 1].float()
            )
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
        self.last_dof_pos_reward[env_ids] = self.dof_pos[env_ids]
        self.last_base_pos_reward[env_ids] = self.root_states[env_ids, :2]
        for buffer in self.command_metrics.buffers():
            buffer[env_ids] = 0.0

    def _accumulate_command_tracking_metrics(self):
        accumulate_command_metrics(
            self.command_metrics, commands=self.commands,
            linear_velocity=self.base_lin_vel, angular_velocity=self.base_ang_vel,
            dof_velocity=self.dof_vel, torques=self.torques, torque_limits=self.torque_limits,
            zero_threshold=self.cfg.rewards.zero_command_threshold,
            wheel_radius=self.cfg.rewards.wheel_radius,
            zero_mask=self._zero_command_mask, method_wheel_terms=self._method_wheel_terms,
            wheel_contact_history=getattr(self, 'wheel_contact_history', None),
            method_v1=self._method_v1,
        )

    def compute_reward(self):
        """Compute rewards
        Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
        adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.0
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            if self.reward_pipeline != "normalized_v1" and name not in getattr(
                self.cfg.rewards, "unclipped_reward_names", ()
            ):
                rew = torch.clip(
                    rew,
                    -self.cfg.rewards.clip_single_reward * self.dt,
                    self.cfg.rewards.clip_single_reward * self.dt,
                )
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.0)
        termination_name = (
            "method_termination"
            if self.reward_pipeline == "normalized_v1"
            else "termination"
        )
        if self.raw_reward_scales.get(termination_name, 0.0) != 0:
            terminal_scale = (
                self.raw_reward_scales[termination_name]
                if self.reward_pipeline == "normalized_v1"
                else self.reward_scales.get(termination_name, 0.0)
            )
            rew = self._reward_termination() * terminal_scale
            self.rew_buf += rew
            self.episode_sums[termination_name] += rew

    def compute_proprioception_observations(self):
        return proprioception(
            angular_velocity=self.base_ang_vel,
            projected_gravity=self.projected_gravity,
            commands=self.commands, command_scale=self.commands_scale,
            position=self.dof_pos, default_position=self.default_dof_pos,
            velocity=self.dof_vel, actions=self.actions, scales=self.obs_scales,
        )

    def compute_observations(self):
        """Computes observations"""
        self.obs_buf = self.compute_proprioception_observations()

        if self.cfg.env.num_privileged_obs is not None:
            self.privileged_obs_buf = privileged_observation(
                root_height=self.root_states[:, 2], measured_heights=self.measured_heights,
                linear_velocity=self.base_lin_vel, clean_policy_observation=self.obs_buf,
                action_history=self.last_actions, acceleration=self.dof_acc,
                torques=self.torques, mass=self.base_mass, center_of_mass=self.base_com,
                default_position=self.default_dof_pos, raw_default_position=self.raw_default_dof_pos,
                friction=self.friction_coef, restitution=self.restitution_coef,
                scales=self.obs_scales, num_environments=self.num_envs,
            )

        # add noise if needed
        if self.add_noise:
            self.obs_buf += (
                2 * torch.rand_like(self.obs_buf) - 1
            ) * self.noise_scale_vec

        update_history(
            self.obs_history, self.obs_buf,
            environment_steps=self.envs_steps_buf,
            policy_decimation=self.cfg.control.decimation,
            history_decimation=self.cfg.env.obs_history_dec,
            observation_width=self.num_obs,
        )

    def create_sim(self):
        """Creates simulation, terrain and evironments"""
        self.up_axis_idx = 2  # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(
            self.sim_device_id,
            self.graphics_device_id,
            self.physics_engine,
            self.sim_params,
        )
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ["heightfield", "trimesh"]:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type == "plane":
            self._create_ground_plane()
        elif mesh_type == "heightfield":
            self._create_heightfield()
        elif mesh_type == "trimesh":
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError(
                "Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]"
            )
        self._create_envs()

    def set_camera(self, position, lookat):
        """Set camera position and direction"""
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    # ------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        state = ShapeRandomizationState(self.friction_coef, self.restitution_coef)
        result = randomize_shape_properties(
            props, env_id, config=self.cfg.domain_rand,
            num_envs=self.num_envs, device=self.device, state=state,
        )
        self.friction_coef = state.friction
        self.restitution_coef = state.restitution
        return result

    def _process_dof_props(self, props, env_id):
        if env_id == 0:
            (self.dof_pos_limits, self.dof_vel_limits, self.torque_limits) = read_dof_limits(
                props, num_dof=self.num_dof, device=self.device,
                soft_position_limit=self.cfg.rewards.soft_dof_pos_limit,
            )
        return apply_armature(props, self.dof_names, self.cfg.asset.dof_armature)

    def _process_rigid_body_props(self, props, env_id):
        state = BodyRandomizationState(
            self.base_mass, self.base_com, getattr(self, 'base_add_mass', None),
        )
        result = randomize_body_properties(
            props, env_id, config=self.cfg.domain_rand,
            num_envs=self.num_envs, device=self.device, state=state,
        )
        self.base_mass = state.mass
        self.base_com = state.com
        if state.added_mass is not None:
            self.base_add_mass = state.added_mass
        return result

    def _post_physics_step_callback(self):
        """Callback called before computing terminations, rewards, and observations
        Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        #
        env_ids = (
            (
                self.episode_length_buf
                % int(self.cfg.commands.resampling_time / self.dt)
                == 0
            )
            .nonzero(as_tuple=False)
            .flatten()
        )
        if not getattr(self.cfg.commands, "hold_command_until_reset", False):
            self._resample_commands(env_ids)
        if getattr(self.cfg.commands,'start_stop_ramp_seconds',0.)>0:
            self._get_start_stop_scheduler().advance(self.commands)
        if getattr(self.cfg.commands,'height_switch_interval',0.)>0:
            from wheel_legged_gym.domain.commands.height_commands import alternate_height
            alternate_height(self.commands,self.episode_length_buf,self.dt,
                self.cfg.commands.height_switch_interval,*self.cfg.commands.ranges.height)
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 1] = torch.clip(
                1.5 * wrap_to_pi(self.commands[:, 3] - heading), -5, 5
            )

        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        self.base_height = torch.mean(
            self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1
        )
            # 5) 更新 jump 的“腾空/落地”状态（非常关键：供多个 reward 共享，避免 reward 顺序依赖）
        # self._update_jump_state()
    # def _update_jump_state(self):
    #     # contact 判定阈值：跳跃建议用高一点（比 no_fly 更鲁棒）
    #     contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
    #     self.num_wheel_contacts = torch.sum(contact.float(), dim=1)

    #     self.fly = self.num_wheel_contacts == 0
    #     self.first_contact = (~self.fly) & (self.base_air_time > 0.0)

    #     # 缓存“落地时”的累计值，供 reward 用（然后本步会 reset）
    #     self.air_time_reward = self.base_air_time.clone()
    #     self.height_int_reward = self.jump_height_int.clone()

    #     # 额外高度（相对 base_height command），并对 jump_height 做截断
    #     base_h_cmd = self.commands[:, 2]
    #     jump_h_cmd = torch.clamp(self.commands[:, self.jump_cmd_idx], min=0.0)

    #     extra_h = torch.clamp(self.root_states[:, 2] - base_h_cmd, min=0.0)
    #     extra_h = torch.minimum(extra_h, jump_h_cmd)

    #     # 只在 fly 时积分；一旦落地，乘 fly -> 自动清零
    #     self.jump_height_int = (self.jump_height_int + self.dt * extra_h) * self.fly.float()
    #     self.base_air_time = (self.base_air_time + self.dt) * self.fly.float()

    def _get_start_stop_scheduler(self):
        if not hasattr(self,'start_stop_scheduler'):
            from wheel_legged_gym.domain.commands.start_stop_commands import RandomStartStop
            self.start_stop_scheduler=RandomStartStop(self.num_envs,self.device,self.dt,
                self.cfg.commands.start_stop_speed,self.cfg.commands.start_stop_ramp_seconds,
                getattr(self.cfg.commands,'start_stop_stride',2))
        return self.start_stop_scheduler

    def _resample_commands(self, env_ids):
        resample_commands(
            env_ids,
            state=CommandBuffers(self.commands, self.command_sample_mode,
                                 self.method_v1_segment_counter),
            ranges=self.command_ranges, config=self.cfg.commands, device=self.device,
            reset_start_stop=lambda commands, ids: self._get_start_stop_scheduler().reset(commands, ids),
            sample_heading=torch_rand_float,
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
        raw_torques = mixed_pd_torques(
            actions,
            position_scale=self.cfg.control.pos_action_scale,
            velocity_scale=self.cfg.control.vel_action_scale,
            p_gains=self.p_gains, d_gains=self.d_gains,
            default_position=self.default_dof_pos,
            position=self.dof_pos, velocity=self.dof_vel,
            torque_scale=self.torques_scale,
        )
        if self._method_v1:
            self.command_metrics.preclip_torque_saturation_sum[:] += (
                torch.abs(raw_torques) >= self.torque_limits * 0.99
            ).float().mean(dim=1) / float(self.cfg.control.decimation)
        return torch.clip(raw_torques, -self.torque_limits, self.torque_limits)

    def _reset_dofs(self, env_ids):
        reset_dofs(
            env_ids, gym=self.gym, sim=self.sim, dof_position=self.dof_pos,
            dof_velocity=self.dof_vel, default_position=self.default_dof_pos,
            dof_state=self.dof_state,
        )

    def _reset_root_states(self, env_ids):
        reset_root_states(
            env_ids, gym=self.gym, sim=self.sim, root_states=self.root_states,
            initial_state=self.base_init_state, origins=self.env_origins,
            custom_origins=self.custom_origins, device=self.device,
        )

    def _push_robots(self):
        """Random pushes the robots."""
        env_ids = (
            (
                self.envs_steps_buf
                % int(self.cfg.domain_rand.push_interval_s / self.sim_params.dt)
                == 0
            )
            .nonzero(as_tuple=False)
            .flatten()
        )
        if len(env_ids) == 0:
            return

        max_push_force = (
            self.base_mass.mean().item()
            * self.cfg.domain_rand.max_push_vel_xy
            / self.sim_params.dt
        )
        self.rigid_body_external_forces[:] = 0
        rigid_body_external_forces = torch_rand_float(
            -max_push_force, max_push_force, (self.num_envs, 3), device=self.device
        )
        self.rigid_body_external_forces[env_ids, 0, 0:3] = quat_rotate(
            self.base_quat[env_ids], rigid_body_external_forces[env_ids]
        )
        self.rigid_body_external_forces[env_ids, 0, 2] *= 0.5

        self.gym.apply_rigid_body_force_tensors(
            self.sim,
            gymtorch.unwrap_tensor(self.rigid_body_external_forces),
            gymtorch.unwrap_tensor(self.rigid_body_external_torques),
            gymapi.ENV_SPACE,
        )

    def _update_terrain_curriculum(self, env_ids):
        """Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        distance = torch.norm(
            self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1
        )
        # robots that walked far enough progress to harder terains
        move_up = distance > self.terrain.env_length / 4  #原本是/2
        # robots that walked less than half of their required distance go to simpler terrains
        move_down = (
            self.episode_sums["tracking_lin_vel"][env_ids] / self.max_episode_length_s
            < (self.reward_scales["tracking_lin_vel"] / self.dt) * 0.4
        ) * ~move_up
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        mask = self.terrain_levels[env_ids] >= self.max_terrain_level
        self.success_ids = env_ids[mask]
        mask = self.terrain_levels[env_ids] < 0
        self.fail_ids = env_ids[mask]
        # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(
            self.terrain_levels[env_ids] >= self.max_terrain_level,
            torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
            torch.clip(self.terrain_levels[env_ids], 0),
        )  # (the minumum level is zero)
        self.env_origins[env_ids] = self.terrain_origins[
            self.terrain_levels[env_ids], self.terrain_types[env_ids]
        ]
        if self.cfg.commands.curriculum:
            self.command_ranges["lin_vel_x"][self.fail_ids, 0] = torch.clip(
                self.command_ranges["lin_vel_x"][self.fail_ids, 0] + 0.25,
                -self.cfg.commands.basic_max_curriculum,
                -1,
            )
            self.command_ranges["lin_vel_x"][self.fail_ids, 1] = torch.clip(
                self.command_ranges["lin_vel_x"][self.fail_ids, 1] - 0.25,
                1,
                self.cfg.commands.basic_max_curriculum,
            )

    def update_command_curriculum(self, env_ids):
        """Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if self.cfg.terrain.curriculum and len(self.success_ids) != 0:
            # self.basic_terrain_idx = torch.cat((self.stair_up_idx, self.discrete_idx))
            # self.advanced_terrain_idx
            mask = (
                self.episode_sums["tracking_lin_vel"][self.success_ids]
                / self.max_episode_length
                > self.cfg.commands.curriculum_threshold
                * self.reward_scales["tracking_lin_vel"]
            )
            success_ids = self.success_ids[mask]
            basic_ids = torch.any(
                success_ids.unsqueeze(1) == self.basic_terrain_idx.unsqueeze(0), dim=1
            )
            basic_ids = success_ids[basic_ids]
            self.command_ranges["lin_vel_x"][success_ids, 0] -= 0.05
            self.command_ranges["lin_vel_x"][success_ids, 1] += 0.05
            self.command_ranges["lin_vel_x"][basic_ids, 0] -= 0.45
            self.command_ranges["lin_vel_x"][basic_ids, 1] += 0.45

            self.command_ranges["lin_vel_x"][self.basic_terrain_idx, :] = torch.clip(
                self.command_ranges["lin_vel_x"][self.basic_terrain_idx, :],
                -self.cfg.commands.basic_max_curriculum,
                self.cfg.commands.basic_max_curriculum,
            )
            self.command_ranges["lin_vel_x"][self.advanced_terrain_idx, :] = torch.clip(
                self.command_ranges["lin_vel_x"][self.advanced_terrain_idx, :],
                -self.cfg.commands.advanced_max_curriculum,
                self.cfg.commands.advanced_max_curriculum,
            )
        if self.cfg.terrain.curriculum == False:
            if (
                torch.mean(self.episode_sums["tracking_lin_vel"][env_ids])
                / self.max_episode_length
                > self.cfg.commands.curriculum_threshold
                * self.reward_scales["tracking_lin_vel"]
                and torch.mean(self.episode_sums["tracking_ang_vel"][env_ids])
                / self.max_episode_length
                > self.cfg.commands.curriculum_threshold
                * self.reward_scales["tracking_ang_vel"]
                * 0.8
            ):
                self.command_ranges["lin_vel_x"][:, 0] = torch.clip(
                    self.command_ranges["lin_vel_x"][:, 0] - 0.1,
                    -self.cfg.commands.basic_max_curriculum,
                    0.0,
                )
                self.command_ranges["lin_vel_x"][:, 1] = torch.clip(
                    self.command_ranges["lin_vel_x"][:, 1] + 0.1,
                    0.0,
                    self.cfg.commands.basic_max_curriculum,
                )

    def _uses_staged_curriculum(self):
        return bool(self.cfg.commands.curriculum) and getattr(
            self.cfg.commands, "curriculum_mode", "legacy"
        ) == "staged_performance"

    def _init_staged_command_curriculum(self):
        if not self._uses_staged_curriculum():
            return
        self.command_curriculum_stages = validate_curriculum_stages(
            self.cfg.commands.curriculum_stages
        )
        configured_stage = int(getattr(self.cfg.commands, "curriculum_initial_stage", 0))
        final_stage = len(self.command_curriculum_stages) - 1
        if not 0 <= configured_stage <= final_stage:
            raise ValueError(f"invalid initial command curriculum stage: {configured_stage}")
        self.command_curriculum_stage = configured_stage
        self.command_curriculum_pass_streak = 0
        self.command_curriculum_last_check_step = 0
        self.command_curriculum_last_metrics = {
            "passed": False,
            "enough_samples": False,
            "episodes": 0,
            "survival_fraction": 0.0,
            "zero_abs_vx": 0.0,
            "zero_abs_yaw": 0.0,
            "reverse_relative_error": 0.0,
            "forward_relative_error": 0.0,
            "yaw_relative_error": 0.0,
        }
        self._reset_staged_curriculum_window()
        self._apply_staged_curriculum_ranges()

    def _reset_staged_curriculum_window(self):
        self.command_curriculum_window_episodes = 0
        self.command_curriculum_window = create_window(self.device)

    def _apply_staged_curriculum_ranges(self):
        linear_limit, yaw_limit = self.command_curriculum_stages[
            self.command_curriculum_stage
        ]
        self.command_ranges["lin_vel_x"][:, 0] = -linear_limit
        self.command_ranges["lin_vel_x"][:, 1] = linear_limit
        self.command_ranges["ang_vel_yaw"][:, 0] = -yaw_limit
        self.command_ranges["ang_vel_yaw"][:, 1] = yaw_limit

    def _accumulate_staged_curriculum_window(self, env_ids):
        if not self._uses_staged_curriculum():
            return
        valid_ids = env_ids[self.episode_length_buf[env_ids] > 0]
        if len(valid_ids) == 0:
            return
        self.command_curriculum_window_episodes += len(valid_ids)
        accumulate_window(
            self.command_curriculum_window, valid_ids,
            timeouts=self.time_out_buf, metrics=self.command_metrics,
        )

    def _maybe_advance_staged_curriculum(self):
        if not self._uses_staged_curriculum():
            return
        if not window_is_due(
            self.common_step_counter, self.command_curriculum_last_check_step,
            self.command_curriculum_window_episodes, self.cfg.commands,
        ):
            return
        stats = {
            "episodes": self.command_curriculum_window_episodes,
            **{
                name: float(value.item())
                for name, value in self.command_curriculum_window.items()
            },
        }
        metrics = evaluate_curriculum_window(stats, self.cfg.commands)
        self.command_curriculum_last_metrics = metrics
        self.command_curriculum_last_check_step = self.common_step_counter
        (self.command_curriculum_stage, self.command_curriculum_pass_streak,
         advanced) = advance_stage(
            self.command_curriculum_stage, self.command_curriculum_pass_streak,
            metrics['passed'], self.command_curriculum_stages,
            self.cfg.commands.curriculum_required_passes,
        )
        if advanced:
            self._apply_staged_curriculum_ranges()
        self._reset_staged_curriculum_window()

    def _staged_curriculum_log_metrics(self):
        return staged_log_metrics(
            self.command_curriculum_stage, self.command_curriculum_pass_streak,
            self.command_curriculum_stages, self.command_curriculum_last_metrics,
        )

    def get_checkpoint_state(self):
        if self._method_v1:
            return method_state(
                self.cfg.commands, getattr(self.cfg, 'domain_rand_level', 0),
                self.method_v1_segment_counter,
            )
        if not self._uses_staged_curriculum():
            return {}
        return staged_state(self.command_curriculum_stage,
                            self.command_curriculum_pass_streak,
                            self.command_curriculum_last_metrics)

    def load_checkpoint_state(self, state):
        if self._method_v1:
            restore_method_counter(state, self.method_v1_segment_counter, self.device)
            return
        if not self._uses_staged_curriculum() or not state:
            return
        restored = restored_stage(state, self.command_curriculum_stages)
        if restored is None:
            return
        self.command_curriculum_stage, curriculum = restored
        self.command_curriculum_pass_streak = int(curriculum.get('pass_streak', 0))
        self.command_curriculum_last_metrics = curriculum.get(
            'last_metrics', self.command_curriculum_last_metrics,
        )
        self._reset_staged_curriculum_window()
        self._apply_staged_curriculum_ranges()

    def _get_noise_scale_vec(self, cfg):
        self.add_noise = self.cfg.noise.add_noise
        return noise_scale_vector(
            self.obs_buf[0], noise_scales=self.cfg.noise.noise_scales,
            noise_level=self.cfg.noise.noise_level,
            observation_scales=self.obs_scales,
            measure_heights=self.cfg.terrain.measure_heights,
        )

    # ----------------------------------------
    def _init_buffers(self):
        """Initialize torch tensors which will contain simulation states and processed quantities"""
        tensors = acquire_state_tensors(
            self.gym, self.sim, num_envs=self.num_envs,
            num_dof=self.num_dof, num_bodies=self.num_bodies,
        )
        self.root_states = tensors.root_states
        self.dof_state = tensors.dof_state
        self.dof_pos = tensors.dof_pos
        self.dof_vel = tensors.dof_vel
        self.dof_acc = tensors.dof_acc
        self.base_quat = tensors.base_quat
        self.contact_forces = tensors.contact_forces
        self.rigid_body_states = tensors.rigid_body_states

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(
            get_axis_params(-1.0, self.up_axis_idx), device=self.device
        ).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1.0, 0.0, 0.0], device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.torques = torch.zeros(
            self.num_envs,
            self.num_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.torques_scale = torch.ones(
            self.num_envs,
            self.num_dof,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.p_gains = torch.zeros(
            self.num_envs,
            self.num_dof,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.d_gains = torch.zeros(
            self.num_envs,
            self.num_dof,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.actions = torch.zeros(
            self.num_envs,
            self.num_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.last_actions = torch.zeros(
            self.num_envs,
            self.num_actions,
            2,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.base_position = self.root_states[:, :3]
        self.last_base_position = self.base_position.clone()
        self.last_dof_pos = torch.zeros_like(self.dof_pos)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(
            self.num_envs,
            self.cfg.commands.num_commands + 1,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )  # x vel, y vel, yaw vel, heading
        self.command_sample_mode = torch.full(
            (self.num_envs,),
            -1,
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )
        self.method_v1_segment_counter = torch.arange(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.command_metrics = create_command_metrics(self.num_envs, self.device)
        self.commands_scale = torch.tensor(
            [
                self.obs_scales.lin_vel,
                self.obs_scales.ang_vel,
                self.obs_scales.height_measurements,
            ],
            device=self.device,
            requires_grad=False,
        )  # TODO change this
        self.command_ranges["lin_vel_x"] = torch.zeros(
            self.num_envs,
            2,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.command_ranges["lin_vel_x"][:] = torch.tensor(
            self.cfg.commands.ranges.lin_vel_x
        )
        self.command_ranges["ang_vel_yaw"] = torch.zeros(
            self.num_envs,
            2,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.command_ranges["ang_vel_yaw"][:] = torch.tensor(
            self.cfg.commands.ranges.ang_vel_yaw
        )
        self.command_ranges["height"] = torch.zeros(
            self.num_envs,
            2,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.command_ranges["height"][:] = torch.tensor(self.cfg.commands.ranges.height)
        self._init_staged_command_curriculum()
        self.feet_air_time = torch.zeros(
            self.num_envs,
            self.feet_indices.shape[0],
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.last_contacts = torch.zeros(
            self.num_envs,
            len(self.feet_indices),
            dtype=torch.bool,
            device=self.device,
            requires_grad=False,
        )
        history_length = int(getattr(self.cfg.rewards, "contact_history_length", 4))
        self.wheel_contact_history = torch.zeros(
            self.num_envs,
            history_length,
            len(self.feet_indices),
            dtype=torch.bool,
            device=self.device,
            requires_grad=False,
        )
        self.wheel_contact_seen = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False
        )
        self.forbidden_contact_streak = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.upright_streak = torch.zeros_like(self.forbidden_contact_streak)
        self.wheel_loss_streak = torch.zeros_like(self.forbidden_contact_streak)
        self.base_lin_vel = quat_rotate_inverse(
            self.base_quat, self.root_states[:, 7:10]
        )
        self.base_ang_vel = quat_rotate_inverse(
            self.base_quat, self.root_states[:, 10:13]
        )
        self.rigid_body_external_forces = torch.zeros(
            (self.num_envs, self.num_bodies, 3), device=self.device, requires_grad=False
        )
        self.rigid_body_external_torques = torch.zeros(
            (self.num_envs, self.num_bodies, 3), device=self.device, requires_grad=False
        )
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.action_delay_idx = torch.zeros(
            self.num_envs,
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )
        delay_max = np.int64(
            np.ceil(self.cfg.domain_rand.delay_ms_range[1] / 1000 / self.sim_params.dt)
        )
        self.action_fifo = torch.zeros(
            (self.num_envs, delay_max, self.cfg.env.num_actions),
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0
        self.base_height = torch.mean(
            self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1
        )

        self.L0 = torch.zeros(
            self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.theta0 = torch.zeros(
            self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False
        )

        control_state = initialize_control_state(
            num_envs=self.num_envs, dof_count=self.num_dof,
            named_dof_count=self.num_dofs, dof_names=self.dof_names,
            default_angles=self.cfg.init_state.default_joint_angles,
            control=self.cfg.control, randomization=self.cfg.domain_rand,
            physics_dt=self.sim_params.dt, device=self.device,
            p_gains=self.p_gains, d_gains=self.d_gains, torque_scales=self.torques_scale,
            delay_indices=self.action_delay_idx, sample_uniform=torch_rand_float, report=print,
        )
        self.raw_default_dof_pos = control_state.raw_default_position
        self.default_dof_pos = control_state.default_position
        self.action_delay_idx = control_state.delay_indices
        # === reward 用的 DOF 历史（防止站立时腿/轮乱动）===
        self.last_dof_pos_reward = self.dof_pos.clone()
        # === reward 用的 base 位置历史（世界坐标）===
        self.last_base_pos_reward = self.root_states[:, :2].clone()




    def _prepare_reward_function(self):
        """Prepares a list of reward functions, whcih will be called to compute the total reward.
        Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        self.raw_reward_scales = dict(self.reward_scales)
        self.reward_pipeline = getattr(self.cfg.rewards, "reward_pipeline", "legacy_v0")
        # Remove zero scales and multiply each normalized term by dt exactly once.
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)
            else:
                if not (
                    self.reward_pipeline == "normalized_v1"
                    and key == "method_termination"
                ):
                    self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name in {"termination", "method_termination"}:
                continue
            self.reward_names.append(name)
            name = "_reward_" + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {
            name: torch.zeros(
                self.num_envs,
                dtype=torch.float,
                device=self.device,
                requires_grad=False,
            )
            for name in self.reward_scales.keys()
        }

    def _create_ground_plane(self):
        create_ground_plane(self.gym, self.sim, self.cfg.terrain)

    def _create_heightfield(self):
        self.height_samples = create_heightfield(
            self.gym, self.sim, self.cfg.terrain, self.terrain, self.device,
        )

    def _create_trimesh(self):
        self.height_samples = create_trimesh(
            self.gym, self.sim, self.cfg.terrain, self.terrain, self.device,
        )

    def _create_envs(self):
        """Creates environments:
        1. loads the robot URDF/MJCF asset,
        2. For each environment
           2.1 creates the environment,
           2.2 calls DOF and Rigid shape properties callbacks,
           2.3 create actor with these properties and add them to the env
        3. Store indices of different bodies of the robot
        """
        asset = load_robot_asset(self.gym, self.sim, self.cfg.asset, WHEEL_LEGGED_GYM_ROOT_DIR)
        robot_asset = asset.handle
        self.num_dof = asset.num_dof
        self.num_bodies = asset.num_bodies
        self.num_dofs = asset.num_dofs
        self.dof_names = asset.dof_names
        body_names = asset.body_names
        dof_props_asset = asset.dof_properties
        rigid_shape_props_asset = asset.rigid_shape_properties

        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = (
            self.cfg.init_state.pos
            + self.cfg.init_state.rot
            + self.cfg.init_state.lin_vel
            + self.cfg.init_state.ang_vel
        )
        self.base_init_state = to_torch(
            base_init_state_list, device=self.device, requires_grad=False
        )
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0.0, 0.0, 0.0)
        env_upper = gymapi.Vec3(0.0, 0.0, 0.0)
        self.actor_handles = []
        self.envs = []
        self.friction_coef = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.restitution_coef = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.base_mass = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.base_com = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False
        )
        create_actor_instances(
            gym=self.gym, sim=self.sim, robot_asset=robot_asset,
            num_envs=self.num_envs, origins=self.env_origins, device=self.device,
            start_pose=start_pose, env_lower=env_lower, env_upper=env_upper,
            asset_name=self.cfg.asset.name, self_collisions=self.cfg.asset.self_collisions,
            rigid_shape_props_asset=rigid_shape_props_asset, dof_props_asset=dof_props_asset,
            process_shapes=self._process_rigid_shape_props,
            process_dofs=self._process_dof_props,
            process_bodies=self._process_rigid_body_props,
            environments=self.envs, actors=self.actor_handles,
        )

        indices = build_asset_indices(
            gym=self.gym, environment=self.envs[0], actor=self.actor_handles[0],
            body_names=body_names, dof_names=self.dof_names, feet_names=feet_names,
            penalized_contact_names=penalized_contact_names,
            termination_contact_names=termination_contact_names,
            rewards=self.cfg.rewards, commands=self.cfg.commands, device=self.device,
            bilateral_indices_present=hasattr(self, 'bilateral_body_indices'),
            bilateral_body_indices=getattr(self, 'bilateral_body_indices', None),
        )
        self.feet_indices = indices.feet_indices
        self.penalised_contact_indices = indices.penalised_contact_indices
        self.termination_contact_indices = indices.termination_contact_indices
        if indices.wheel_dof_indices is not None:
            self.wheel_dof_indices = indices.wheel_dof_indices
        if indices.wheel_body_indices is not None:
            self.wheel_body_indices = indices.wheel_body_indices
        if indices.bilateral_body_indices is not None:
            self.bilateral_body_indices = indices.bilateral_body_indices
        if indices.fudan_leg_landmarks is not None:
            self.fudan_leg_landmarks = indices.fudan_leg_landmarks

    def _get_env_origins(self):
        layout = build_origin_layout(
            num_envs=self.num_envs, device=self.device, terrain_config=self.cfg.terrain,
            spacing=self.cfg.env.env_spacing,
            terrain_origin_array=(self.terrain.env_origins if self.cfg.terrain.mesh_type
                                  in ["heightfield", "trimesh"] else None),
        )
        self.custom_origins = layout.custom_origins
        self.env_origins = layout.env_origins
        self.flat_idx = layout.flat_idx
        if layout.custom_origins:
            self.terrain_levels = layout.terrain_levels
            self.terrain_types = layout.terrain_types
            self.smooth_slope_idx = layout.smooth_slope_idx
            self.rough_slope_idx = layout.rough_slope_idx
            self.stair_up_idx = layout.stair_up_idx
            self.stair_down_idx = layout.stair_down_idx
            self.discrete_idx = layout.discrete_idx
            self.basic_terrain_idx = layout.basic_terrain_idx
            self.advanced_terrain_idx = layout.advanced_terrain_idx
            self.max_terrain_level = layout.max_terrain_level
            self.terrain_origins = layout.terrain_origins
            self.terrain_x_max = layout.terrain_x_max
            self.terrain_x_min = layout.terrain_x_min
            self.terrain_y_max = layout.terrain_y_max
            self.terrain_y_min = layout.terrain_y_min

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        if self.cfg.terrain.mesh_type not in ["heightfield", "trimesh"]:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval = np.ceil(
            self.cfg.domain_rand.push_interval_s / self.dt
        )

    def _draw_debug_vis(self):
        """Draws visualizations for dubugging (slows down simulation a lot).
        Default behaviour: draws height measurement points
        """
        # draw height lines
        if not self.terrain.cfg.measure_heights:
            return
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
        for i in range(self.num_envs):
            base_pos = (self.root_states[i, :3]).cpu().numpy()
            heights = self.measured_heights[i].cpu().numpy()
            height_points = (
                quat_apply_yaw(
                    self.base_quat[i].repeat(heights.shape[0]), self.height_points[i]
                )
                .cpu()
                .numpy()
            )
            for j in range(heights.shape[0]):
                x = height_points[j, 0] + base_pos[0]
                y = height_points[j, 1] + base_pos[1]
                z = heights[j]
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(
                    sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose
                )

    def _init_height_points(self):
        """Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(
            self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False
        )
        x = torch.tensor(
            self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False
        )
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(
            self.num_envs,
            self.num_height_points,
            3,
            device=self.device,
            requires_grad=False,
        )
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _get_heights(self, env_ids=None):
        return sample_heights(
            env_ids, mesh_type=self.cfg.terrain.mesh_type, num_envs=self.num_envs,
            num_height_points=self.num_height_points, device=self.device,
            base_quat=self.base_quat, root_states=self.root_states,
            height_points=getattr(self, 'height_points', None),
            height_samples=getattr(self, 'height_samples', None),
            terrain_config=self.terrain.cfg if self.cfg.terrain.mesh_type not in ('plane', 'none') else None,
        )

    def pre_physics_step(self):
        self.rwd_linVelTrackPrev = self._reward_tracking_lin_vel()
        self.rwd_angVelTrackPrev = self._reward_tracking_ang_vel()

    # ------------ reward functions----------------
    def _reward_inputs(self):
        """Snapshot field references at the existing reward evaluation boundary."""
        return RewardInputs(
            _method_v1=getattr(self, "_method_v1", None),
            actions=getattr(self, "actions", None),
            base_ang_vel=getattr(self, "base_ang_vel", None),
            base_height=getattr(self, "base_height", None),
            base_lin_vel=getattr(self, "base_lin_vel", None),
            base_quat=getattr(self, "base_quat", None),
            bilateral_body_indices=getattr(self, "bilateral_body_indices", None),
            cfg=getattr(self, "cfg", None),
            commands=getattr(self, "commands", None),
            contact_forces=getattr(self, "contact_forces", None),
            default_dof_pos=getattr(self, "default_dof_pos", None),
            device=getattr(self, "device", None),
            dof_acc=getattr(self, "dof_acc", None),
            dof_pos=getattr(self, "dof_pos", None),
            dof_pos_limits=getattr(self, "dof_pos_limits", None),
            dof_vel=getattr(self, "dof_vel", None),
            dof_vel_limits=getattr(self, "dof_vel_limits", None),
            feet_indices=getattr(self, "feet_indices", None),
            fudan_leg_landmarks=getattr(self, "fudan_leg_landmarks", None),
            last_actions=getattr(self, "last_actions", None),
            num_envs=getattr(self, "num_envs", None),
            penalised_contact_indices=getattr(self, "penalised_contact_indices", None),
            projected_gravity=getattr(self, "projected_gravity", None),
            reset_buf=getattr(self, "reset_buf", None),
            reward_scales=getattr(self, "reward_scales", None),
            rigid_body_states=getattr(self, "rigid_body_states", None),
            root_states=getattr(self, "root_states", None),
            rwd_angVelTrackPrev=getattr(self, "rwd_angVelTrackPrev", None),
            rwd_linVelTrackPrev=getattr(self, "rwd_linVelTrackPrev", None),
            termination_contact_indices=getattr(self, "termination_contact_indices", None),
            theta0=getattr(self, "theta0", None),
            time_out_buf=getattr(self, "time_out_buf", None),
            torque_limits=getattr(self, "torque_limits", None),
            torques=getattr(self, "torques", None),
            wheel_body_indices=getattr(self, "wheel_body_indices", None),
            wheel_contact_history=getattr(self, "wheel_contact_history", None),
            wheel_dof_indices=getattr(self, "wheel_dof_indices", None),
        )

    def _method_task_gate(self):
        return evaluate_reward("_method_task_gate", self._reward_inputs())

    def _method_wheel_terms(self):
        return evaluate_reward("_method_wheel_terms", self._reward_inputs())

    def _reward_track_vx_coarse(self):
        return evaluate_reward("_reward_track_vx_coarse", self._reward_inputs())

    def _reward_track_vx_fine(self):
        return evaluate_reward("_reward_track_vx_fine", self._reward_inputs())

    def _reward_track_vx_gap(self):
        return evaluate_reward("_reward_track_vx_gap", self._reward_inputs())

    def _reward_track_yaw_coarse(self):
        return evaluate_reward("_reward_track_yaw_coarse", self._reward_inputs())

    def _reward_track_yaw_fine(self):
        return evaluate_reward("_reward_track_yaw_fine", self._reward_inputs())

    def _reward_track_yaw_gap(self):
        return evaluate_reward("_reward_track_yaw_gap", self._reward_inputs())

    def _reward_height_cost(self):
        return evaluate_reward("_reward_height_cost", self._reward_inputs())

    def _reward_lateral_velocity(self):
        return evaluate_reward("_reward_lateral_velocity", self._reward_inputs())

    def _reward_wheel_slip(self):
        return evaluate_reward("_reward_wheel_slip", self._reward_inputs())

    def _reward_airborne_wheel_spin(self):
        return evaluate_reward("_reward_airborne_wheel_spin", self._reward_inputs())

    def _reward_wheel_contact_loss(self):
        return evaluate_reward("_reward_wheel_contact_loss", self._reward_inputs())

    def _reward_forbidden_contact(self):
        return evaluate_reward("_reward_forbidden_contact", self._reward_inputs())

    def _reward_torque_cost(self):
        return evaluate_reward("_reward_torque_cost", self._reward_inputs())

    def _reward_power_cost(self):
        return evaluate_reward("_reward_power_cost", self._reward_inputs())

    def _reward_action_second_diff(self):
        return evaluate_reward("_reward_action_second_diff", self._reward_inputs())

    def _reward_lin_vel_z(self):
        return evaluate_reward("_reward_lin_vel_z", self._reward_inputs())

    def _reward_ang_vel_xy(self):
        return evaluate_reward("_reward_ang_vel_xy", self._reward_inputs())

    def _reward_orientation(self):
        return evaluate_reward("_reward_orientation", self._reward_inputs())

    def _reward_base_height(self):
        return evaluate_reward("_reward_base_height", self._reward_inputs())

    def _reward_base_height_enhance(self):
        return evaluate_reward("_reward_base_height_enhance", self._reward_inputs())

    def _reward_torques(self):
        return evaluate_reward("_reward_torques", self._reward_inputs())

    def _reward_power(self):
        return evaluate_reward("_reward_power", self._reward_inputs())

    def _reward_dof_vel(self):
        return evaluate_reward("_reward_dof_vel", self._reward_inputs())

    def _reward_dof_acc(self):
        return evaluate_reward("_reward_dof_acc", self._reward_inputs())

    def _reward_action_rate(self):
        return evaluate_reward("_reward_action_rate", self._reward_inputs())

    def _reward_action_smooth(self):
        return evaluate_reward("_reward_action_smooth", self._reward_inputs())

    def _reward_collision(self):
        return evaluate_reward("_reward_collision", self._reward_inputs())

    def _reward_termination(self):
        return evaluate_reward("_reward_termination", self._reward_inputs())

    def _reward_dof_pos_limits(self):
        return evaluate_reward("_reward_dof_pos_limits", self._reward_inputs())

    def _reward_dof_vel_limits(self):
        return evaluate_reward("_reward_dof_vel_limits", self._reward_inputs())

    def _reward_torque_limits(self):
        return evaluate_reward("_reward_torque_limits", self._reward_inputs())

    def _reward_tracking_lin_vel(self):
        return evaluate_reward("_reward_tracking_lin_vel", self._reward_inputs())

    def _reward_tracking_lin_vel_enhance(self):
        return evaluate_reward("_reward_tracking_lin_vel_enhance", self._reward_inputs())

    def _reward_high_speed_tracking(self):
        return evaluate_reward("_reward_high_speed_tracking", self._reward_inputs())

    def _reward_high_speed_yaw_tracking(self):
        return evaluate_reward("_reward_high_speed_yaw_tracking", self._reward_inputs())

    def _reward_high_speed_yaw_penalty(self):
        return evaluate_reward("_reward_high_speed_yaw_penalty", self._reward_inputs())

    def _reward_high_speed_slip(self):
        return evaluate_reward("_reward_high_speed_slip", self._reward_inputs())

    def _reward_tracking_ang_vel(self):
        return evaluate_reward("_reward_tracking_ang_vel", self._reward_inputs())

    def _reward_tracking_ang_vel_enhance(self):
        return evaluate_reward("_reward_tracking_ang_vel_enhance", self._reward_inputs())

    def _zero_command_mask(self):
        return evaluate_reward("_zero_command_mask", self._reward_inputs())

    def _reward_zero_base_velocity(self):
        return evaluate_reward("_reward_zero_base_velocity", self._reward_inputs())

    def _reward_zero_wheel_velocity(self):
        return evaluate_reward("_reward_zero_wheel_velocity", self._reward_inputs())

    def _reward_low_speed_tracking(self):
        return evaluate_reward("_reward_low_speed_tracking", self._reward_inputs())

    def _reward_zero_yaw_wheel_symmetry(self):
        return evaluate_reward("_reward_zero_yaw_wheel_symmetry", self._reward_inputs())

    def _reward_theta0_equ_0(self):
        return evaluate_reward("_reward_theta0_equ_0", self._reward_inputs())
    
    def _reward_tracking_lin_vel_pbrs(self):
        return evaluate_reward("_reward_tracking_lin_vel_pbrs", self._reward_inputs())

    def _reward_tracking_ang_vel_pbrs(self):
        return evaluate_reward("_reward_tracking_ang_vel_pbrs", self._reward_inputs())

    def _reward_stumble(self):
        return evaluate_reward("_reward_stumble", self._reward_inputs())

    def _reward_stand_still(self):
        return evaluate_reward("_reward_stand_still", self._reward_inputs())

    def _reward_stand_bilateral_geometry(self):
        return evaluate_reward("_reward_stand_bilateral_geometry", self._reward_inputs())

    def _reward_nominal_state(self):
        return evaluate_reward("_reward_nominal_state", self._reward_inputs())

    def _reward_feet_contact_forces(self):
        return evaluate_reward("_reward_feet_contact_forces", self._reward_inputs())
