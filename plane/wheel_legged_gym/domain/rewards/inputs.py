"""Explicit reward state contract; tensor references are borrowed and must not be mutated."""
from dataclasses import dataclass
from typing import Optional
import torch


@dataclass(frozen=True)
class RewardInputs:
    _method_v1: bool
    actions: Optional[torch.Tensor]
    base_ang_vel: Optional[torch.Tensor]
    base_height: Optional[torch.Tensor]
    base_lin_vel: Optional[torch.Tensor]
    base_quat: Optional[torch.Tensor]
    bilateral_body_indices: Optional[torch.Tensor]
    cfg: object
    commands: Optional[torch.Tensor]
    contact_forces: Optional[torch.Tensor]
    default_dof_pos: Optional[torch.Tensor]
    device: str
    dof_acc: Optional[torch.Tensor]
    dof_pos: Optional[torch.Tensor]
    dof_pos_limits: Optional[torch.Tensor]
    dof_vel: Optional[torch.Tensor]
    dof_vel_limits: Optional[torch.Tensor]
    feet_indices: Optional[torch.Tensor]
    fudan_leg_landmarks: Optional[torch.Tensor]
    last_actions: Optional[torch.Tensor]
    num_envs: int
    penalised_contact_indices: Optional[torch.Tensor]
    projected_gravity: Optional[torch.Tensor]
    reset_buf: Optional[torch.Tensor]
    reward_scales: object
    rigid_body_states: Optional[torch.Tensor]
    root_states: Optional[torch.Tensor]
    rwd_angVelTrackPrev: Optional[torch.Tensor]
    rwd_linVelTrackPrev: Optional[torch.Tensor]
    termination_contact_indices: Optional[torch.Tensor]
    theta0: Optional[torch.Tensor]
    time_out_buf: Optional[torch.Tensor]
    torque_limits: Optional[torch.Tensor]
    torques: Optional[torch.Tensor]
    wheel_body_indices: Optional[torch.Tensor]
    wheel_contact_history: Optional[torch.Tensor]
    wheel_dof_indices: Optional[torch.Tensor]
