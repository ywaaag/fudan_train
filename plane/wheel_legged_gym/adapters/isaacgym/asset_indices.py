"""Resolve contact and policy landmark indices after actor creation."""
from dataclasses import dataclass
from typing import Optional
import torch


@dataclass(frozen=True)
class AssetIndices:
    feet_indices: Optional[torch.Tensor]
    wheel_dof_indices: Optional[torch.Tensor]
    wheel_body_indices: Optional[torch.Tensor]
    bilateral_body_indices: Optional[torch.Tensor]
    fudan_leg_landmarks: Optional[torch.Tensor]
    penalised_contact_indices: Optional[torch.Tensor]
    termination_contact_indices: Optional[torch.Tensor]


def build_asset_indices(*, gym, environment, actor, body_names, dof_names,
                        feet_names, penalized_contact_names, termination_contact_names,
                        rewards, commands, device, bilateral_indices_present=False,
                        bilateral_body_indices=None):
    """Keep body-handle lookup order and optional policy-contract checks intact."""
    wheel_dof_indices = wheel_body_indices = fudan_leg_landmarks = None
    feet_indices = torch.zeros(
        len(feet_names), dtype=torch.long, device=device, requires_grad=False
    )
    for i in range(len(feet_names)):
        feet_indices[i] = gym.find_actor_rigid_body_handle(
            environment, actor, feet_names[i]
        )

    if getattr(rewards, "reward_pipeline", "legacy_v0") == "normalized_v1":
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
        missing_dofs = [name for name in expected_dofs if name not in dof_names]
        missing_wheels = [name for name in expected_wheels if name not in body_names]
        if missing_dofs or missing_wheels:
            raise ValueError(
                "method_v1 asset contract mismatch: "
                f"missing_dofs={missing_dofs}, missing_wheels={missing_wheels}"
            )
        if len(expected_wheels) != 2:
            raise ValueError("method_v1 requires exactly two wheel bodies")
        wheel_dof_indices = torch.tensor(
            [dof_names.index(name) for name in expected_wheel_dofs],
            dtype=torch.long,
            device=device,
        )
        wheel_body_indices = torch.tensor(
            [body_names.index(name) for name in expected_wheels],
            dtype=torch.long,
            device=device,
        )
        feet_indices = wheel_body_indices.clone()
        bilateral_indices_present = True
        landmarks = ('left_leg_1_link', 'right_leg_1_link',
                     'left_wheel_link', 'right_wheel_link')
        bilateral_body_indices = torch.tensor(
            [body_names.index(name) for name in landmarks],
            dtype=torch.long, device=device)

    if getattr(rewards,'straight_bilateral_geometry',False) and not bilateral_indices_present:
        bilateral_body_indices=torch.tensor([body_names.index(n) for n in
            ('left_leg_1_link','right_leg_1_link','left_wheel_link','right_wheel_link')],
            dtype=torch.long,device=device)
    if getattr(commands, 'training_profile', '') == 'fudan_stand_v1':
        fudan_leg_landmarks = torch.tensor([body_names.index(n) for n in
            ('left_leg_0_link', 'right_leg_0_link', 'left_wheel_link', 'right_wheel_link')],
            dtype=torch.long, device=device)

    penalised_contact_indices = torch.zeros(
        len(penalized_contact_names),
        dtype=torch.long,
        device=device,
        requires_grad=False,
    )
    for i in range(len(penalized_contact_names)):
        penalised_contact_indices[i] = gym.find_actor_rigid_body_handle(
            environment, actor, penalized_contact_names[i]
        )

    termination_contact_indices = torch.zeros(
        len(termination_contact_names),
        dtype=torch.long,
        device=device,
        requires_grad=False,
    )
    for i in range(len(termination_contact_names)):
        termination_contact_indices[i] = gym.find_actor_rigid_body_handle(
            environment, actor, termination_contact_names[i]
        )
    return AssetIndices(
        feet_indices=feet_indices,
        wheel_dof_indices=wheel_dof_indices,
        wheel_body_indices=wheel_body_indices,
        bilateral_body_indices=bilateral_body_indices,
        fudan_leg_landmarks=fudan_leg_landmarks,
        penalised_contact_indices=penalised_contact_indices,
        termination_contact_indices=termination_contact_indices,
    )
