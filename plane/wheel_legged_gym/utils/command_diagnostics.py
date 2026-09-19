"""Opt-in time-step weighted rollout metrics; never edits policy or rewards."""
import json
from pathlib import Path
import torch


class CommandDiagnostics:
    commands = (-1., -.5, -.1, 0., .1, .5, 1.)
    names = ('count', 'vx_sum', 'tracking_abs_sum', 'encoder_bias_sum', 'encoder_abs_sum',
             'yaw_abs_sum', 'height_abs_sum', 'failure_count', 'timeout_count')

    def __init__(self, env, log_dir):
        self.env = env
        self.path = Path(log_dir)/'command_diagnostics.jsonl'
        self.values = torch.tensor(self.commands, device=env.device)
        self.sums = torch.zeros(7, len(self.names), device=env.device)
        self.reward_names = sorted(env.episode_sums)
        self.reward_sums = torch.zeros(7, len(self.reward_names), device=env.device)
        self.bucket = None
        self.original_reward = env.compute_reward
        env.compute_reward = self.compute_reward

    def before_step(self, latent):
        command = self.env.commands[:, 0]
        distance = (command[:, None]-self.values).abs()
        self.bucket = distance.argmin(dim=1)
        # Accumulate mismatch as data rather than silently assigning an off-grid command.
        if (distance.min(dim=1).values > 1e-5).any():
            raise ValueError('Diagnostic command is outside the declared grid')
        error = latent[:, 0]/self.env.obs_scales.lin_vel - self.env.base_lin_vel[:, 0]
        self.sums[:, 3].scatter_add_(0, self.bucket, error)
        self.sums[:, 4].scatter_add_(0, self.bucket, error.abs())

    def compute_reward(self):
        # Called after physics and termination checks, before reset/resampling.
        before = torch.stack([self.env.episode_sums[k] for k in self.reward_names], dim=1)
        self.original_reward()
        if self.bucket is None:
            return
        env = self.env
        after = torch.stack([env.episode_sums[k] for k in self.reward_names], dim=1)
        self.reward_sums.index_add_(0, self.bucket, after-before)
        count = torch.ones_like(env.commands[:, 0])
        zero = torch.zeros_like(count)
        done = env.reset_buf.bool()
        timeout = env.time_out_buf.bool()
        batch = torch.stack([count, env.base_lin_vel[:, 0],
            (env.base_lin_vel[:, 0]-env.commands[:, 0]).abs(), zero, zero,
            env.base_ang_vel[:, 2].abs(), (env.base_height-env.commands[:, 2]).abs(),
            (done & ~timeout).float(), (done & timeout).float()], dim=1)
        self.sums.index_add_(0, self.bucket, batch)

    def flush(self, iteration, writer=None, encoder_action_shift=None):
        sums, rewards = self.sums.cpu(), self.reward_sums.cpu()
        rows = []
        total = float(sums[:, 0].sum())
        for i, command in enumerate(self.commands):
            n = float(sums[i, 0])
            row = {'command_vx':command, 'samples':int(n), 'env_seconds':n*self.env.dt,
                   'fraction':n/total if total else None}
            for name,j in [('vx_mean',1),('vx_mae',2),('encoder_vx_bias',3),
                           ('encoder_vx_mae',4),('yaw_mae',5),('height_mae',6)]:
                row[name] = float(sums[i,j])/n if n else None
            row.update(failures=int(sums[i,7]),timeouts=int(sums[i,8]),
                reward_per_env_second={k:float(rewards[i,j])/(n*self.env.dt) if n else None
                                       for j,k in enumerate(self.reward_names)})
            rows.append(row)
            if writer is not None:
                for key in ['samples','env_seconds','fraction','vx_mean','vx_mae','encoder_vx_bias','encoder_vx_mae','yaw_mae','height_mae','failures','timeouts']:
                    if row[key] is not None:writer.add_scalar(f'Command/{command:+.1f}/{key}',row[key],iteration)
        payload={'iteration':iteration,'rows':rows,'encoder_update_action_shift':encoder_action_shift,
                 'semantics':'tracking/reward after physics before resets; encoder error paired with pre-action state; all steps count, no episode-mean averaging'}
        with self.path.open('a') as stream:stream.write(json.dumps(payload)+'\n')
        self.sums.zero_();self.reward_sums.zero_()
        return payload
