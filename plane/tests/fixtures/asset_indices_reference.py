"""Frozen asset index construction, 2026-09-24."""
import torch

def build(self, body_names, feet_names, penalized_contact_names, termination_contact_names):
    self.feet_indices = torch.zeros(
        len(feet_names), dtype=torch.long, device=self.device, requires_grad=False
    )
    for i in range(len(feet_names)):
        self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(
            self.envs[0], self.actor_handles[0], feet_names[i]
        )

    if getattr(self.cfg.rewards, "reward_pipeline", "legacy_v0") == "normalized_v1":
        expected_dofs = (
            "left_leg_0",
            "left_leg_1",
            "left_wheel",
            "right_leg_0",
            "right_leg_1",
            "right_wheel",
        )
        expected_wheel_dofs = ("left_wheel", "right_wheel")
        expected_wheels = ("left_wheel_link", "right_wheel_link")
        missing_dofs = [name for name in expected_dofs if name not in self.dof_names]
        missing_wheels = [name for name in expected_wheels if name not in body_names]
        if missing_dofs or missing_wheels:
            raise ValueError(
                "method_v1 asset contract mismatch: "
                f"missing_dofs={missing_dofs}, missing_wheels={missing_wheels}"
            )
        if len(expected_wheels) != 2:
            raise ValueError("method_v1 requires exactly two wheel bodies")
        self.wheel_dof_indices = torch.tensor(
            [self.dof_names.index(name) for name in expected_wheel_dofs],
            dtype=torch.long,
            device=self.device,
        )
        self.wheel_body_indices = torch.tensor(
            [body_names.index(name) for name in expected_wheels],
            dtype=torch.long,
            device=self.device,
        )
        self.feet_indices = self.wheel_body_indices.clone()
        landmarks = ('left_leg_1_link', 'right_leg_1_link',
                     'left_wheel_link', 'right_wheel_link')
        self.bilateral_body_indices = torch.tensor(
            [body_names.index(name) for name in landmarks],
            dtype=torch.long, device=self.device)

    if getattr(self.cfg.rewards,'straight_bilateral_geometry',False) and not hasattr(self,'bilateral_body_indices'):
        self.bilateral_body_indices=torch.tensor([body_names.index(n) for n in
            ('left_leg_1_link','right_leg_1_link','left_wheel_link','right_wheel_link')],
            dtype=torch.long,device=self.device)
    if getattr(self.cfg.commands, 'training_profile', '') == 'fudan_stand_v1':
        self.fudan_leg_landmarks = torch.tensor([body_names.index(n) for n in
            ('left_leg_0_link', 'right_leg_0_link', 'left_wheel_link', 'right_wheel_link')],
            dtype=torch.long, device=self.device)

    self.penalised_contact_indices = torch.zeros(
        len(penalized_contact_names),
        dtype=torch.long,
        device=self.device,
        requires_grad=False,
    )
    for i in range(len(penalized_contact_names)):
        self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(
            self.envs[0], self.actor_handles[0], penalized_contact_names[i]
        )

    self.termination_contact_indices = torch.zeros(
        len(termination_contact_names),
        dtype=torch.long,
        device=self.device,
        requires_grad=False,
    )
    for i in range(len(termination_contact_names)):
        self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(
            self.envs[0], self.actor_handles[0], termination_contact_names[i]
        )
