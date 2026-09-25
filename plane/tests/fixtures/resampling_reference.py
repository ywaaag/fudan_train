"""Frozen LeggedRobot resampling before extraction, 2026-09-24."""
import torch
from isaacgym.torch_utils import torch_rand_float
from wheel_legged_gym.domain.commands.command_sampling import sample_method_v1, sample_zero_reverse_mixture

def _resample_commands(self, env_ids):
    """Randommly select commands of some environments

    Args:
        env_ids (List[int]): Environments ids for which new commands are needed
    """
    strategy = getattr(self.cfg.commands, "sampling_strategy", "uniform")
    if strategy == "height_bank":
        from wheel_legged_gym.domain.commands.command_sampling import sample_height_bank
        height_selected = sample_height_bank(self.command_ranges['lin_vel_x'][env_ids],
            self.command_ranges['ang_vel_yaw'][env_ids], self.command_ranges['height'][env_ids],
            self.method_v1_segment_counter[env_ids], self.cfg.commands.height_bank)
        self.commands[env_ids, :2] = height_selected[:, :2]
        self.command_sample_mode[env_ids] = -1
        self.method_v1_segment_counter[env_ids] += 1
    elif strategy == "fixed_bank":
        from wheel_legged_gym.domain.commands.command_sampling import sample_fixed_bank
        selected = sample_fixed_bank(self.command_ranges['lin_vel_x'][env_ids],
            self.command_ranges['ang_vel_yaw'][env_ids],
            self.method_v1_segment_counter[env_ids], self.cfg.commands.fixed_bank)
        self.commands[env_ids, :2] = selected
        self.command_sample_mode[env_ids] = -1
        self.method_v1_segment_counter[env_ids] += 1
    elif strategy == "method_v1":
        linear, yaw, mode = sample_method_v1(
            self.command_ranges["lin_vel_x"][env_ids],
            self.command_ranges["ang_vel_yaw"][env_ids],
            phase=getattr(self.cfg.commands, "training_phase", "stand"),
            small_linear_limit=float(
                getattr(self.cfg.commands, "mixture_small_linear_limit", 0.10)
            ),
            small_yaw_limit=float(
                getattr(self.cfg.commands, "mixture_small_yaw_limit", 0.10)
            ),
            slot_ids=self.method_v1_segment_counter[env_ids],
            translation_anchors=getattr(self.cfg.commands, 'translation_retention_anchors', None),
            zero_retention=getattr(self.cfg.commands, 'zero_retention', False),
        )
        self.commands[env_ids, 0] = linear
        self.commands[env_ids, 1] = yaw
        self.command_sample_mode[env_ids] = mode
        self.method_v1_segment_counter[env_ids] += 1
    elif strategy == "zero_reverse_mixture":
        linear, yaw, mode = sample_zero_reverse_mixture(
            self.command_ranges["lin_vel_x"][env_ids],
            self.command_ranges["ang_vel_yaw"][env_ids],
            zero_fraction=self.cfg.commands.mixture_zero_fraction,
            small_fraction=self.cfg.commands.mixture_small_fraction,
            reverse_fraction=self.cfg.commands.mixture_reverse_fraction,
            forward_fraction=self.cfg.commands.mixture_forward_fraction,
            small_linear_limit=self.cfg.commands.mixture_small_linear_limit,
            small_yaw_limit=self.cfg.commands.mixture_small_yaw_limit,
            small_yaw_only_fraction=self.cfg.commands.mixture_small_yaw_only_fraction,
            linear_yaw_zero_fraction=self.cfg.commands.mixture_linear_yaw_zero_fraction,
            small_linear_anchors=self.cfg.commands.mixture_small_linear_anchors,
            small_yaw_anchors=self.cfg.commands.mixture_small_yaw_anchors,
            endpoint_anchor_fraction=self.cfg.commands.mixture_endpoint_anchor_fraction,
            endpoint_linear_anchors=self.cfg.commands.mixture_endpoint_linear_anchors,
        )
        self.commands[env_ids, 0] = linear
        self.commands[env_ids, 1] = yaw
        self.command_sample_mode[env_ids] = mode
    elif strategy == "uniform":
        self.commands[env_ids, 0] = (
            self.command_ranges["lin_vel_x"][env_ids, 1]
            - self.command_ranges["lin_vel_x"][env_ids, 0]
        ) * torch.rand(len(env_ids), device=self.device) + self.command_ranges[
            "lin_vel_x"
        ][env_ids, 0]
        self.commands[env_ids, 1] = (
            self.command_ranges["ang_vel_yaw"][env_ids, 1]
            - self.command_ranges["ang_vel_yaw"][env_ids, 0]
        ) * torch.rand(len(env_ids), device=self.device) + self.command_ranges[
            "ang_vel_yaw"
        ][env_ids, 0]
        self.command_sample_mode[env_ids] = -1
    else:
        raise ValueError(f"unknown command sampling strategy: {strategy}")
    self.commands[env_ids, 2] = (
        self.command_ranges["height"][env_ids, 1]
        - self.command_ranges["height"][env_ids, 0]
    ) * torch.rand(len(env_ids), device=self.device) + self.command_ranges[
        "height"
    ][
        env_ids, 0
    ]

    # # 清空 jump_height
    if getattr(self.cfg.commands,'start_stop_ramp_seconds',0.)>0:
        # Reset cohort starts at exact zero; retention cohort keeps sampled anchors.
        self._get_start_stop_scheduler().reset(self.commands,env_ids)
    if strategy == 'height_bank':
        self.commands[env_ids, 2] = height_selected[:, 2]
    # self.commands[env_ids, self.jump_cmd_idx] = 0.0

    # # 采样 jump_height（像 diablo 一样：先采样，再用 mask 置 0）
    # jl, jr = self.cfg.commands.ranges.jump_height
    # jump_heights = torch.rand(len(env_ids), device=self.device) * (jr - jl) + jl

    # # threshold 越大 => 越多置 0 => 越不跳
    # mask = torch.rand(len(env_ids), device=self.device) < self.cfg.commands.threshold
    # jump_heights[mask] = 0.0

    # self.commands[env_ids, self.jump_cmd_idx] = jump_heights

    # # 你想要的 jump 状态机（全局更新）
    # self.jump = self.commands[:, self.jump_cmd_idx] != 0


    if self.cfg.commands.heading_command:
        self.commands[env_ids, 3] = torch_rand_float(
            self.command_ranges["heading"][0],
            self.command_ranges["heading"][1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)
