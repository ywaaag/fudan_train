"""Frozen control initialization before extraction, 2026-09-24."""
import torch
from isaacgym.torch_utils import torch_rand_float

def initialize(self):
    # joint positions offsets and PD gains
    self.raw_default_dof_pos = torch.zeros(
        self.num_dof,
        dtype=torch.float,
        device=self.device,
        requires_grad=False,
    )
    self.default_dof_pos = torch.zeros(
        self.num_envs,
        self.num_dof,
        dtype=torch.float,
        device=self.device,
        requires_grad=False,
    )
    for i in range(self.num_dofs):
        name = self.dof_names[i]
        angle = self.cfg.init_state.default_joint_angles[name]
        self.raw_default_dof_pos[i] = angle
        self.default_dof_pos[:, i] = angle
        found = False
        for dof_name in self.cfg.control.stiffness.keys():
            if dof_name in name:
                self.p_gains[:, i] = self.cfg.control.stiffness[dof_name]
                self.d_gains[:, i] = self.cfg.control.damping[dof_name]
                found = True
        if not found:
            self.p_gains[:, i] = 0.0
            self.d_gains[:, i] = 0.0
            if self.cfg.control.control_type in ["P", "V"]:
                print(
                    f"PD gain of joint {name} were not defined, setting them to zero"
                )
    if self.cfg.domain_rand.randomize_Kp:
        (
            p_gains_scale_min,
            p_gains_scale_max,
        ) = self.cfg.domain_rand.randomize_Kp_range
        self.p_gains *= torch_rand_float(
            p_gains_scale_min,
            p_gains_scale_max,
            self.p_gains.shape,
            device=self.device,
        )
    if self.cfg.domain_rand.randomize_Kd:
        (
            d_gains_scale_min,
            d_gains_scale_max,
        ) = self.cfg.domain_rand.randomize_Kd_range
        self.d_gains *= torch_rand_float(
            d_gains_scale_min,
            d_gains_scale_max,
            self.d_gains.shape,
            device=self.device,
        )
    if self.cfg.domain_rand.randomize_motor_torque:
        (
            torque_scale_min,
            torque_scale_max,
        ) = self.cfg.domain_rand.randomize_motor_torque_range
        self.torques_scale *= torch_rand_float(
            torque_scale_min,
            torque_scale_max,
            self.torques_scale.shape,
            device=self.device,
        )
    if self.cfg.domain_rand.randomize_default_dof_pos:
        self.default_dof_pos += torch_rand_float(
            self.cfg.domain_rand.randomize_default_dof_pos_range[0],
            self.cfg.domain_rand.randomize_default_dof_pos_range[1],
            (self.num_envs, self.num_dof),
            device=self.device,
        )
    if self.cfg.domain_rand.randomize_action_delay:
        action_delay_idx = torch.round(
            torch_rand_float(
                self.cfg.domain_rand.delay_ms_range[0] / 1000 / self.sim_params.dt,
                self.cfg.domain_rand.delay_ms_range[1] / 1000 / self.sim_params.dt,
                (self.num_envs, 1),
                device=self.device,
            )
        ).squeeze(-1)
        self.action_delay_idx = action_delay_idx.long()
