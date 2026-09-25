"""Simulator state views must alias source storage rather than silently copy it."""
import torch

from wheel_legged_gym.adapters.isaacgym import state_tensors


def test_acquisition_order_and_shared_view_writes(monkeypatch):
    root = torch.zeros(2, 13)
    dofs = torch.zeros(12, 2)
    contacts = torch.zeros(6, 3)
    bodies = torch.zeros(6, 13)
    events = []
    class Gym:
        def acquire_actor_root_state_tensor(self, sim):
            events.append('acquire_root'); return root
        def acquire_dof_state_tensor(self, sim):
            events.append('acquire_dof'); return dofs
        def acquire_net_contact_force_tensor(self, sim):
            events.append('acquire_contact'); return contacts
        def acquire_rigid_body_state_tensor(self, sim):
            events.append('acquire_body'); return bodies
        def refresh_dof_state_tensor(self, sim): events.append('refresh_dof')
        def refresh_actor_root_state_tensor(self, sim): events.append('refresh_root')
        def refresh_net_contact_force_tensor(self, sim): events.append('refresh_contact')
        def refresh_rigid_body_state_tensor(self, sim): events.append('refresh_body')
    monkeypatch.setattr(state_tensors.gymtorch, 'wrap_tensor', lambda tensor: tensor)
    views = state_tensors.acquire_state_tensors(Gym(), 'sim', num_envs=2, num_dof=6, num_bodies=3)
    assert events == ['acquire_root', 'acquire_dof', 'acquire_contact', 'acquire_body',
                      'refresh_dof', 'refresh_root', 'refresh_contact', 'refresh_body']
    views.dof_pos[1, 2] = 7
    views.dof_vel[0, 3] = 9
    views.base_quat[0, 0] = 1
    views.rigid_body_states[1, 2, 0] = 5
    views.contact_forces[0, 1, 2] = 4
    assert dofs[8, 0] == 7 and dofs[3, 1] == 9
    assert root[0, 3] == 1 and bodies[5, 0] == 5 and contacts[1, 2] == 4
    assert not views.dof_acc.any()
    views.dof_acc.fill_(42)
    assert dofs[0, 0] == 0
