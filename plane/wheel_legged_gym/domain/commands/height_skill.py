"""Open-loop public height commands for the independent squat cohort."""
import torch


class HeightSkill:
    def __init__(self, num_envs, device, dt, slots, *, cycle=20,
                 entry_range=(.8, 1.8), hold_range=(3., 6.), ramp_seconds=2.):
        all_ids = torch.arange(num_envs, device=device)
        self.ids = all_ids[torch.isin(all_ids.remainder(cycle),
                       torch.as_tensor(slots, device=device))]
        self.dt = dt
        self.entry_range = entry_range
        self.hold_range = hold_range
        self.ramp_seconds = ramp_seconds
        self.target = torch.full((num_envs,), .4, device=device)
        self.age = torch.zeros(num_envs, device=device)
        self.entry = torch.zeros(num_envs, device=device)
        self.hold = torch.zeros(num_envs, device=device)

    def reset(self, commands, ids, targets):
        if not len(ids):
            return
        self.target[ids] = targets[:, 1]
        self.age[ids] = 0.
        self.entry[ids] = self.entry_range[0] + (
            self.entry_range[1] - self.entry_range[0]) * torch.rand(len(ids), device=ids.device)
        self.hold[ids] = self.hold_range[0] + (
            self.hold_range[1] - self.hold_range[0]) * torch.rand(len(ids), device=ids.device)
        commands[ids, 0] = targets[:, 0]
        commands[ids, 1] = 0.
        commands[ids, 2] = .4

    def advance(self, commands):
        ids = self.ids
        self.age[ids] += self.dt
        t = self.age[ids] - self.entry[ids]
        enter = torch.clamp(t / self.ramp_seconds, 0., 1.)
        exit_fraction = torch.clamp((t - self.ramp_seconds - self.hold[ids]) /
                                    self.ramp_seconds, 0., 1.)
        commands[ids, 2] = .4 - (.4 - self.target[ids]) * enter * (1. - exit_fraction)
