"""Command-derived roll reference with the original quadratic orientation scale."""
import torch


def roll_reference(vx, yaw, max_lean_rad):
    if not 0. < max_lean_rad <= .17453292519943295:
        raise ValueError('Lean reference exceeds reviewed 10 degree ceiling')
    equilibrium_hint = torch.atan(vx * yaw / 9.81)
    return max_lean_rad * torch.tanh(equilibrium_hint / max_lean_rad)


def orientation_cost(projected_gravity, vx, yaw, max_lean_rad):
    roll = roll_reference(vx, yaw, max_lean_rad)
    return projected_gravity[:, 0].square() + (
        projected_gravity[:, 1] + torch.sin(roll)).square()
