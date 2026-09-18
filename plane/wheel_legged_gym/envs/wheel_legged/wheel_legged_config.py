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

from wheel_legged_gym.envs.base.legged_robot_config import (
    LeggedRobotCfg,
    LeggedRobotCfgPPO,
)

# 课程和奖励默认值
class WheelLeggedCfg(LeggedRobotCfg):

    class commands(LeggedRobotCfg.commands):
        curriculum = False  # 关闭命令课程学习
        curriculum_mode = "legacy"
        # Each stage is (abs linear limit [m/s], abs yaw limit [rad/s]).
        # 这里改课程阶段
        curriculum_min_episodes = 2048
        curriculum_check_interval_steps = 200
        curriculum_required_passes = 3
        curriculum_initial_stage = 0
        curriculum_stages = (
            (0.5, 0.8),
            (1.0, 1.0),
            (2.0, 2.0),
            (3.5, 3.5),
            (3.75, 3.75),
            (4.0, 4.0),
            (5.0, 5.0),
        )
        curriculum_min_survival = 0.95
        curriculum_max_zero_vx = 0.1
        curriculum_max_zero_yaw = 0.08
        curriculum_max_linear_relative_error = 0.40
        curriculum_max_yaw_relative_error = 0.45
        # Baseline remains uniform. Named experiments override this in-process.
        sampling_strategy = "uniform"
        mixture_zero_fraction = 0.20
        mixture_small_fraction = 0.20
        mixture_reverse_fraction = 0.30
        mixture_forward_fraction = 0.30
        mixture_small_linear_limit = 0.10
        mixture_small_yaw_limit = 0.10
        mixture_small_yaw_only_fraction = 0.50
        mixture_linear_yaw_zero_fraction = 0.50
        mixture_small_linear_anchors = None
        mixture_small_yaw_anchors = None
        mixture_endpoint_anchor_fraction = 0.0
        mixture_endpoint_linear_anchors = None
        training_profile = "legacy"
        training_phase = "legacy"
        hold_command_until_reset = False
        method_v1_level = 0
        method_v1_phase_config = {}
        
        class ranges:
            # Motion phase: start with small commands while preserving the
            # already-validated upright equilibrium.
            lin_vel_x = [-0.5, 0.5]    # m/s
            ang_vel_yaw = [-0.8, 0.8]  # rad/s
            # The generated tree URDF places the wheel center at z ~= 0.06 m
            # for a base root height of 0.40 m and a 0.06 m wheel radius.
            # At the nominal zero-joint pose the wheel center is about
            # -0.33993 m relative to the base and the radius is 0.06 m.
            # Keep the target at 0.40 m to avoid ground penetration/locking.
            height = [0.40, 0.40]

    class terrain(LeggedRobotCfg.terrain):
        curriculum = False  # 关闭地形课程学习
        mesh_type = "plane"  # 纯平面，无地形随机
        max_init_terrain_level = 0  # 从最简单地形开始（平面）

    class init_state(LeggedRobotCfg.init_state):
        # pos = [0.0, 0.0, 0.1]  # x,y,z [m]
        # default_joint_angles = { "lf0_Joint": -0.23, 
        #                         "lf1_Joint": -0.65, 
        #                         "l_wheel_Joint": 0.0, 
        #                         "rf0_Joint": 0.23, 
        #                         "rf1_Joint": 0.65, 
        #                         "r_wheel_Joint": 0.0, 
        #                         }
        pos = [0.0, 0.0, 0.4]  # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w
        default_joint_angles = { "left_leg_0": 0.0,
                                "left_leg_1": 0.0,
                                "left_wheel": 0.0,
                                "right_leg_0": 0.0,
                                "right_leg_1": 0.0,
                                "right_wheel": 0.0,
                                }


    class control(LeggedRobotCfg.control):
        pos_action_scale = 0.5
        vel_action_scale = 10.0
        # PD Drive parameters:
        stiffness = {"leg_0": 20.0, "leg_1": 20.0, "wheel": 0.0}
        # Wheel velocity target needs enough torque to overcome static
        # wheel-ground friction; 0.2 produced visible sliding with the
        # ~20 kg tree model.  Keep leg gains unchanged for this ablation.
        damping = {"leg_0": 1.0, "leg_1": 1.0, "wheel": 1.0}


    class rewards(LeggedRobotCfg.rewards):
        class scales(LeggedRobotCfg.rewards.scales):
            #这里改 reward 权重
            orientation = -500.0  # 强化机身水平约束（只站立）
            base_height = 2.0     # 强化站立高度
            tracking_lin_vel = 1.0       # enable forward/backward tracking
            high_speed_tracking = 0.0
            high_speed_yaw_tracking = 0.0
            high_speed_yaw_penalty = 0.0
            high_speed_slip = 0.0
            tracking_lin_vel_enhance = 0.0
            tracking_ang_vel = 1.0       # enable yaw tracking
            tracking_ang_vel_enhance = 0.0
            dof_vel = -0.01  # 鼓励关节速度低（接近0度时会停止运动）
            zero_base_velocity = -1.0
            zero_wheel_velocity = -1.0
            low_speed_tracking = 0.0
            zero_yaw_wheel_symmetry = 0.0
            # method_v1 normalized reward groups.  They stay disabled for the
            # legacy baseline and are enabled atomically by the profile.
            track_vx_coarse = 0.0
            track_vx_fine = 0.0
            track_vx_gap = 0.0
            track_yaw_coarse = 0.0
            track_yaw_fine = 0.0
            track_yaw_gap = 0.0
            height_cost = 0.0
            lateral_velocity = 0.0
            wheel_slip = 0.0
            airborne_wheel_spin = 0.0
            wheel_contact_loss = 0.0
            forbidden_contact = 0.0
            torque_cost = 0.0
            power_cost = 0.0
            action_second_diff = 0.0
            method_termination = 0.0
            stand_bilateral_geometry = 0.0
            
        dof_pos_target = 0.0  # 目标关节角度（0=-直腿）
        bilateral_geometry_tolerance_m = 0.005
        bilateral_geometry_scale_m = 0.05
        zero_command_threshold = 1.0e-2
        zero_yaw_rate_weight = 0.25
        low_speed_command_threshold = 0.1001  #这里改对称奖励的生效范围
        low_speed_yaw_weight = 1.0            #这里改对称奖励的生效范围
        zero_yaw_command_threshold = 0.05
        max_straight_line_speed = 5.0
        high_speed_command_threshold = 1.5
        high_speed_tracking_sigma = 0.5
        high_speed_yaw_tracking_sigma = 0.5
        high_speed_yaw_penalty_command_threshold = 0.5
        high_speed_slip_command_threshold = 2.5
        wheel_radius = 0.06
        wheel_slip_sigma = 0.25
        yaw_penalty_sigma = 0.5
        unclipped_reward_names = ()
        reward_pipeline = "legacy_v0"
        tracking_linear_cap = 1.0
        tracking_yaw_cap = 0.8
        tracking_coarse_sigma = 1.0
        tracking_fine_sigma = 0.25
        tracking_gap_delta = 1.0
        tracking_gap_clip = 4.0
        upright_gate_tolerance = 0.35
        height_gate_tolerance = 0.05
        wheel_slip_scale = 0.50
        wheel_air_spin_scale = 1.0
        contact_force_threshold = 1.0
        forbidden_contact_force_threshold = 5.0
        forbidden_contact_grace_s = 0.15
        wheel_loss_grace_s = 0.15
        contact_warmup_s = 0.50
        contact_history_length = 4

    class domain_rand(LeggedRobotCfg.domain_rand):
        # Robustness phase: keep perturbations moderate near the validated
        # nominal equilibrium.
        randomize_friction = True
        friction_range = [0.85, 1.15]
        randomize_restitution = False
        randomize_base_mass = True
        added_mass_range = [-0.5, 0.5]
        randomize_inertia = True
        randomize_inertia_range = [0.95, 1.05]
        randomize_base_com = True
        rand_com_vec = [0.01, 0.01, 0.01]
        randomize_Kp = True
        randomize_Kp_range = [0.97, 1.03]
        randomize_Kd = True
        randomize_Kd_range = [0.97, 1.03]
        randomize_motor_torque = True
        randomize_motor_torque_range = [0.95, 1.05]
        randomize_default_dof_pos = False
        push_robots = True
        push_interval_s = 5
        max_push_vel_xy = 0.4

    class asset(LeggedRobotCfg.asset):


        # file = "{WHEEL_LEGGED_GYM_ROOT_DIR}/resources/robots/infantry_V1/urdf/infantry_V1.urdf"
        file = "{WHEEL_LEGGED_GYM_ROOT_DIR}/../assets/wheel_leg_train.urdf"



        name = "WheelLegged"
        offset = 0.0
        # l1 = 0.215
        # l2 = 0.258 旧车的
        l1 = 0.2107245577
        l2 = 0.2512637371
        foot_name = "wheel"
        penalize_contacts_on = []
        terminate_after_contacts_on = []
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        dof_armature = {"leg_0": 0.01, "leg_1": 0.01, "wheel": 0.01}


class WheelLeggedCfgPPO(LeggedRobotCfgPPO):
    class runner(LeggedRobotCfgPPO.runner):
        # logging
        experiment_name = "wheel_legged"
        max_iterations = 50000
