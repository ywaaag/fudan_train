"""Open-loop, randomized timing for the reviewed turn cohort."""
import torch


class TurnEnvelope:
    def __init__(self, num_envs, device, dt, stride=4, *, turn_height=.4,
                 height_schedule=None, slots=None, cycle=20,
                 entry_range=(.5, 1.5), hold_range=(2., 5.),
                 speed_ramp_seconds=2., yaw_ramp_seconds=2.):
        self.dt = dt
        all_ids = torch.arange(num_envs, device=device)
        self.ids = (all_ids[torch.isin(all_ids.remainder(cycle),
                    torch.as_tensor(slots, device=device))] if slots is not None
                    else all_ids[::stride])
        self.turn_height = turn_height
        self.height_schedule = tuple(height_schedule or ())
        self.entry_range = entry_range
        self.hold_range = hold_range
        self.speed_ramp_seconds = speed_ramp_seconds
        self.yaw_ramp_seconds = yaw_ramp_seconds
        self.target = torch.zeros(num_envs, 2, device=device)
        self.age = torch.zeros(num_envs, device=device)
        self.entry = torch.zeros(num_envs, device=device)
        self.hold = torch.zeros(num_envs, device=device)
        self.target_height = torch.full((num_envs,), float(turn_height), device=device)

    def reset(self, commands, ids, targets):
        if not len(ids):
            return
        self.target[ids] = targets
        self.target_height[ids] = float(self.turn_height)
        for threshold, height in self.height_schedule:
            mask = torch.abs(targets[:, 0] * targets[:, 1]) >= float(threshold)
            self.target_height[ids[mask]] = float(height)
        self.age[ids] = 0.
        self.entry[ids] = self.entry_range[0] + (
            self.entry_range[1] - self.entry_range[0]) * torch.rand(len(ids), device=ids.device)
        self.hold[ids] = self.hold_range[0] + (
            self.hold_range[1] - self.hold_range[0]) * torch.rand(len(ids), device=ids.device)
        commands[ids, :2] = 0.

    def advance(self, commands):
        ids = self.ids
        self.age[ids] += self.dt
        t = self.age[ids] - self.entry[ids]
        speed_fraction = torch.clamp(t / self.speed_ramp_seconds, 0., 1.)
        yaw_fraction = torch.clamp((t - self.speed_ramp_seconds) / self.yaw_ramp_seconds, 0., 1.)
        exit_fraction = torch.clamp((t - self.speed_ramp_seconds -
            self.yaw_ramp_seconds - self.hold[ids]) / self.yaw_ramp_seconds, 0., 1.)
        commands[ids, 0] = self.target[ids, 0] * speed_fraction
        commands[ids, 1] = self.target[ids, 1] * yaw_fraction * (1. - exit_fraction)
        commands[ids, 2] = .4 - (.4 - self.target_height[ids]) * yaw_fraction * (1. - exit_fraction)
