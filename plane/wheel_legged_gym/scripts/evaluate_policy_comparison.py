"""Matched deterministic command grid for historical and current checkpoints.

Every policy uses the same current tree physics, reset distribution, contact
termination and parameter randomization. This measures deployable behavior in
that common environment, not historical training reward or a causal ablation.
"""
import argparse
import hashlib
import json
from pathlib import Path

import isaacgym  # must precede torch
import numpy as np
import torch
from isaacgym.torch_utils import quat_rotate_inverse
from wheel_legged_gym.envs import *  # noqa
from wheel_legged_gym.utils import task_registry
from wheel_legged_gym.utils.helpers import class_to_dict
from wheel_legged_gym.envs.wheel_legged.policy_experiments import (
    apply_training_profile, _apply_method_randomization,
)
from wheel_legged_gym.envs.wheel_legged.policy_experiments import apply_policy_experiment
from wheel_legged_gym.scripts.isaac_parity_trace import _gym_args, _disable_randomization
from wheel_legged_gym.scripts.isaac_command_grid import load_policy, _set_fixed_ranges


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--commands', type=float, nargs='+', default=[0, -.5, .5, -1, 1])
    p.add_argument('--envs-per-command', type=int, default=16)
    p.add_argument('--seed', type=int, default=19)
    p.add_argument('--randomization-level', type=int, choices=[0, 1], default=1)
    p.add_argument('--seconds', type=float, default=25)
    p.add_argument('--warmup', type=float, default=5)
    p.add_argument('--profile', choices=['method_v1','LEGACY_URDF'], default='method_v1')
    a = p.parse_args()
    if not 0 <= a.warmup < a.seconds or a.envs_per_command < 1:
        p.error('require seconds > warmup >= 0 and positive environment count')
    if a.out.exists():
        raise FileExistsError(a.out)
    cfg, train = task_registry.get_cfgs(name='wheel_legged')
    if a.profile == 'LEGACY_URDF':
        apply_policy_experiment(cfg, 'LEGACY_URDF', train)
    else:
        apply_training_profile(cfg, train, phase='translate', level=0)
    _disable_randomization(cfg)
    _apply_method_randomization(cfg, a.randomization_level)
    cfg.domain_rand_level = a.randomization_level
    cfg.domain_rand.push_robots = False
    cfg.commands.sampling_strategy = 'uniform'
    cfg.commands.curriculum = False
    cfg.commands.heading_command = False
    cfg.env.num_envs = len(a.commands) * a.envs_per_command
    cfg.env.episode_length_s = a.seconds + 10
    cfg.terrain.mesh_type = 'plane'
    cfg.terrain.curriculum = False
    cfg.commands.ranges.height = [.4, .4]
    args = _gym_args()
    args.num_envs, args.seed = cfg.env.num_envs, a.seed
    env, _ = task_registry.make_env(name='wheel_legged', args=args, env_cfg=cfg)
    try:
        commands = np.repeat(np.array([[v, 0., .4] for v in a.commands]),
                             a.envs_per_command, axis=0)
        _set_fixed_ranges(env, commands)
        env.reset()
        obs, history = env.get_observations()
        target = torch.tensor(commands, device=env.device, dtype=torch.float)
        policy = load_policy(a.checkpoint, env.device)
        expected_dofs = ['left_leg_0', 'left_leg_1', 'left_wheel',
                         'right_leg_0', 'right_leg_1', 'right_wheel']
        assert list(env.dof_names) == expected_dofs
        assert obs.shape[1] == 25 and history.shape[1] == 125
        assert abs(env.dt - .01) < 1e-7 and abs(env.sim_params.dt - .005) < 1e-7
        body_names = env.gym.get_actor_rigid_body_names(env.envs[0], env.actor_handles[0])
        nonwheel = [i for i, n in enumerate(body_names) if 'wheel' not in n]
        landmarks = [body_names.index(n) for n in ['left_leg_1_link', 'right_leg_1_link',
                                                  'left_wheel_link', 'right_wheel_link']]
        initial_root = env.root_states.detach().cpu().clone().tolist()
        initial_obs = obs.detach().cpu().clone().tolist()
        failures = torch.zeros(env.num_envs, device=env.device)
        timeouts = torch.zeros_like(failures)
        nonwheel_full = torch.zeros_like(failures)
        names = ['vx', 'vx_mae', 'abs_vx', 'yaw', 'abs_yaw', 'height', 'height_mae',
                 'abs_roll', 'abs_pitch', 'left_contact', 'right_contact',
                 'nonwheel_contact', 'slip_rms', 'torque_saturation',
                 'knee_mirror_m', 'wheel_mirror_m', 'encoder_vx_mae', 'action_clip']
        sums = torch.zeros(env.num_envs, len(names), device=env.device)
        maxima = torch.zeros_like(sums)
        count = 0
        for step in range(round(a.seconds / env.dt)):
            if not torch.allclose(env.commands[:, :3], target):
                raise RuntimeError('Actual command changed during fixed-command evaluation')
            if not torch.allclose(obs[:, 6:9], target * env.commands_scale):
                raise RuntimeError('Command is not present in policy observation')
            with torch.inference_mode():
                action, latent = policy.act_inference(obs, history)
                encoder_error = (env.base_lin_vel[:, 0] - latent[:, 0]/env.obs_scales.lin_vel).abs()
            if not torch.isfinite(action).all():
                raise RuntimeError('Nonfinite policy action')
            before_saturation = env.command_metric_preclip_torque_saturation_sum.clone()
            obs, _, _, dones, info, history = env.step(action)
            timeout = info.get('time_outs', torch.zeros_like(dones)).bool()
            failures += (dones.bool() & ~timeout).float()
            timeouts += (dones.bool() & timeout).float()
            body_contact = (env.contact_forces[:, nonwheel].norm(dim=-1) > 1).any(dim=-1).float()
            nonwheel_full += body_contact
            # Counters reset with episodes; post-reset samples are not valid
            # torque observations. Any reset independently fails acceptance.
            sat = (env.command_metric_preclip_torque_saturation_sum - before_saturation).clamp(min=0)
            sat = torch.where(dones.bool(), torch.zeros_like(sat), sat)
            if step < round(a.warmup / env.dt):
                continue
            gravity = env.projected_gravity
            roll = torch.atan2(-gravity[:, 1], -gravity[:, 2])
            pitch = torch.atan2(gravity[:, 0], gravity[:, 1:].norm(dim=-1))
            contact = (env.contact_forces[:, env.feet_indices, 2] > 1).float()
            positions = env.rigid_body_states[:, landmarks, :3] - env.root_states[:, None, :3]
            local = quat_rotate_inverse(env.base_quat[:, None, :].expand(-1, 4, -1).reshape(-1, 4),
                                        positions.reshape(-1, 3)).reshape(-1, 4, 3)
            mirror = local.new_tensor([1, -1, 1])
            vx, yaw = env.base_lin_vel[:, 0], env.base_ang_vel[:, 2]
            if getattr(env, '_method_v1', False):
                slip_metric = env._method_wheel_terms()['residual_rms']
            else:
                slip_metric = torch.abs(vx.unsqueeze(1) + .06 * env.dof_vel[:, [2, 5]]).mean(dim=1)
            values = torch.stack([vx, (vx-target[:, 0]).abs(), vx.abs(), yaw, yaw.abs(),
                env.base_height, (env.base_height-.4).abs(), roll.abs(), pitch.abs(),
                contact[:, 0], contact[:, 1], body_contact,
                slip_metric, sat,
                (local[:, 0]-local[:, 1]*mirror).norm(dim=-1),
                (local[:, 2]-local[:, 3]*mirror).norm(dim=-1), encoder_error,
                (action.abs() > env.cfg.normalization.clip_actions).float().mean(dim=-1)], dim=-1)
            if not torch.isfinite(values).all():
                raise RuntimeError('Nonfinite physics metrics')
            sums += values
            maxima = torch.maximum(maxima, values.abs())
            count += 1
        means = (sums / count).cpu()
        results = []
        for i, vx in enumerate(a.commands):
            sl = slice(i*a.envs_per_command, (i+1)*a.envs_per_command)
            metrics = {n: means[sl, j].mean().item() for j, n in enumerate(names)}
            results.append({'command': [vx, 0., .4], 'metrics': metrics,
                'failure_count': failures[sl].sum().item(), 'timeout_count': timeouts[sl].sum().item(),
                'survival_fraction': (failures[sl] == 0).float().mean().item(),
                'nonwheel_contact_full_fraction': (nonwheel_full[sl]/round(a.seconds/env.dt)).mean().item(),
                'per_env_metrics': {n: means[sl, j].tolist() for j, n in enumerate(names)},
                'max_abs_metrics': {n: maxima[sl, j].max().item() for j, n in enumerate(names)},
                'torque_metric_valid_without_resets': bool((failures[sl]+timeouts[sl]).sum().item() == 0)})
        configuration = {n: class_to_dict(getattr(cfg, n)) for n in
                         ['control', 'init_state', 'normalization', 'domain_rand', 'asset', 'sim', 'env']}
        payload = {'schema': 'matched_policy_grid_v1', 'checkpoint': str(a.checkpoint.resolve()),
            'checkpoint_sha256': digest(a.checkpoint), 'evaluator_sha256': digest(__file__),
            'seed': a.seed, 'randomization_level': a.randomization_level,
            'envs_per_command': a.envs_per_command, 'seconds': a.seconds, 'warmup': a.warmup,
            'measurement_samples': count, 'deterministic': True, 'noise': False,
            'policy_dt': env.dt, 'physics_dt': env.sim_params.dt, 'dof_names': env.dof_names,
            'torque_limits': env.torque_limits.cpu().tolist(), 'configuration': configuration,
            'initial_root_states': initial_root, 'initial_observations': initial_obs,
            'results': results, 'note': 'Reset counts cover warmup too; metrics after warmup include reset trajectories. Any reset fails the gate.'}
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(payload, indent=2)+'\n')
        print(json.dumps({'out': str(a.out), 'results': [{k:v for k,v in r.items()
            if k in ['command','metrics','failure_count','timeout_count']} for r in results]}))
    finally:
        env.gym.destroy_sim(env.sim)


if __name__ == '__main__':
    main()
