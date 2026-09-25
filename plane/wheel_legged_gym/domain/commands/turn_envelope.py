"""Open-loop, randomized timing for the reviewed turn cohort."""
import torch


class TurnEnvelope:
    def __init__(self, num_envs, device, dt, stride=4):
        self.dt = dt
        self.ids = torch.arange(0, num_envs, stride, device=device)
        self.target = torch.zeros(num_envs, 2, device=device)
        self.age = torch.zeros(num_envs, device=device)
        self.entry = torch.zeros(num_envs, device=device)
        self.hold = torch.zeros(num_envs, device=device)

    def reset(self, commands, ids, targets):
        if not len(ids):
            return
        self.target[ids] = targets
        self.age[ids] = 0.
        self.entry[ids] = .5 + torch.rand(len(ids), device=ids.device)
        self.hold[ids] = 2. + 3. * torch.rand(len(ids), device=ids.device)
        commands[ids, :2] = 0.

    def advance(self, commands):
        ids = self.ids
        self.age[ids] += self.dt
        t = self.age[ids] - self.entry[ids]
        speed_fraction = torch.clamp(t / 2., 0., 1.)
        yaw_fraction = torch.clamp((t - 2.) / 2., 0., 1.)
        exit_fraction = torch.clamp((t - 4. - self.hold[ids]) / 2., 0., 1.)
        commands[ids, 0] = self.target[ids, 0] * speed_fraction
        commands[ids, 1] = self.target[ids, 1] * yaw_fraction * (1. - exit_fraction)
