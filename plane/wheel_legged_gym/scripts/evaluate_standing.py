"""Deterministic standing audit, including tilt, bilateral pose and failures."""
import argparse
import json
import time
from pathlib import Path

import isaacgym  # must precede torch
import torch
from isaacgym import gymtorch
from isaacgym.torch_utils import quat_rotate_inverse
from wheel_legged_gym.envs import *
from wheel_legged_gym.utils import task_registry
from wheel_legged_gym.envs.wheel_legged.policy_experiments import (
    apply_training_profile, _apply_method_randomization,
)
from wheel_legged_gym.scripts.isaac_parity_trace import _gym_args, _disable_randomization
from wheel_legged_gym.scripts.isaac_command_grid import load_policy


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--num-envs', type=int, default=32)
    parser.add_argument('--seed', type=int, default=19)
    parser.add_argument('--randomization-level', type=int, choices=range(4), default=0)
    parser.add_argument('--seconds', type=float, default=25)
    parser.add_argument('--warmup', type=float, default=5)
    parser.add_argument('--height', type=float, default=0.40)
    parser.add_argument('--gui', action='store_true', help='Matched configuration visual audit; approximately real time')
    parser.add_argument('--push-delta-v', type=float, default=0., help='World horizontal velocity impulse magnitude (m/s), not force')
    parser.add_argument('--profile', choices=['method_v1', 'FUDAN_STAND', 'STAND_CONTROL', 'STAND_SYMMETRIC'], default='method_v1')
    opts = parser.parse_args()
    if not 0 <= opts.warmup < opts.seconds:
        parser.error('require 0 <= warmup < seconds')
    if not 0 <= opts.push_delta_v <= 0.5:
        parser.error('push-delta-v must be within [0,0.5]')
    cfg, train = task_registry.get_cfgs(name='wheel_legged')
    if opts.profile != 'method_v1':
        from wheel_legged_gym.envs.wheel_legged.policy_experiments import apply_policy_experiment
        apply_policy_experiment(cfg, opts.profile, train)
    else:
        apply_training_profile(cfg, train, phase='stand', level=0)
    _disable_randomization(cfg)
    _apply_method_randomization(cfg, opts.randomization_level)
    cfg.domain_rand.push_robots = False  # audit uses an explicit reproducible schedule
    cfg.env.num_envs = opts.num_envs
    cfg.env.episode_length_s = opts.seconds + 10
    cfg.terrain.mesh_type = 'plane'
    cfg.terrain.curriculum = False
    cfg.commands.ranges.height = [opts.height, opts.height]
    args = _gym_args()
    if opts.gui:
        args.headless = False
        args.graphics_device_id = 0
    args.num_envs, args.seed = opts.num_envs, opts.seed
    env, _ = task_registry.make_env(name='wheel_legged', args=args, env_cfg=cfg)
    if opts.gui:
        center = env.root_states[0, :3].detach().cpu().tolist()
        env.set_camera([center[0]+1.5, center[1]-1.5, 1.0], [center[0], center[1], .35])
    policy = load_policy(opts.checkpoint, env.device)
    expected_dofs = ['left_leg_0', 'left_leg_1', 'left_wheel',
                     'right_leg_0', 'right_leg_1', 'right_wheel']
    if list(env.dof_names) != expected_dofs:
        raise ValueError(f'Unexpected DOF order: {env.dof_names}')
    body_ids = [env.gym.find_actor_rigid_body_handle(env.envs[0], env.actor_handles[0], name)
                for name in ['left_leg_1_link', 'right_leg_1_link', 'left_wheel_link', 'right_wheel_link']]
    if min(body_ids) < 0:
        raise ValueError('Missing bilateral body landmarks')
    nonwheel_ids = [i for i, name in enumerate(env.gym.get_actor_rigid_body_names(env.envs[0], env.actor_handles[0]))
                    if 'wheel' not in name]
    env.reset()
    obs, history = env.get_observations()
    failures = torch.zeros(opts.num_envs, device=env.device)
    timeouts = failures.clone()
    samples = []
    geometry = []
    poses = []
    xy_samples = []
    push_results = []
    active_push = None
    directions = ((1.,0.),(-1.,0.),(0.,1.),(0.,-1.),(1.,0.))
    for step in range(round(opts.seconds / env.dt)):
        step_started = time.monotonic()
        if opts.push_delta_v and step in [round(t/env.dt) for t in (10,20,30,40,50)]:
            if active_push is not None:
                push_results.append(active_push)
            index = len(push_results)
            direction = directions[index]
            env.root_states[:, 7:9] += env.root_states.new_tensor(direction) * opts.push_delta_v
            env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(env.root_states))
            active_push = {'time_s': step*env.dt, 'direction_world_xy': direction,
                           'recovery_seconds': [None]*opts.num_envs}
            streak = torch.zeros(opts.num_envs, device=env.device)
            invalid = torch.zeros(opts.num_envs, dtype=torch.bool, device=env.device)
        with torch.inference_mode():
            action, _ = policy.act_inference(obs, history)
        obs, _, _, dones, info, history = env.step(action)
        timeout = info.get('time_outs', torch.zeros_like(dones)).bool()
        failures += (dones.bool() & ~timeout).float()
        timeouts += (dones.bool() & timeout).float()
        if active_push is not None:
            invalid |= dones.bool()
            upright = torch.acos(torch.clamp(-env.projected_gravity[:, 2], -1., 1.)) < .15
            contact_ok = (env.contact_forces[:, env.feet_indices, 2] > 1).all(dim=1)
            body_clear = ~(env.contact_forces[:, nonwheel_ids].norm(dim=-1) > 1).any(dim=1)
            good = upright & contact_ok & body_clear & (abs(env.base_height-.4)<.03) & (env.base_lin_vel[:,:2].norm(dim=1)<.2) & ~invalid
            streak = torch.where(good, streak+1, torch.zeros_like(streak))
            for i in (streak >= round(.5/env.dt)).nonzero().flatten().tolist():
                if active_push['recovery_seconds'][i] is None:
                    active_push['recovery_seconds'][i] = (step+1)*env.dt-active_push['time_s']
        if step * env.dt < opts.warmup:
            if opts.gui:
                time.sleep(max(0., env.dt-(time.monotonic()-step_started)))
            continue
        gravity = env.projected_gravity
        roll = torch.atan2(-gravity[:, 1], -gravity[:, 2])
        pitch = torch.atan2(gravity[:, 0], torch.norm(gravity[:, 1:], dim=-1))
        contact = (env.contact_forces[:, env.feet_indices, 2] > 1).float()
        values = torch.stack((env.base_lin_vel[:, 0], env.base_ang_vel[:, 2],
            env.base_height, roll, pitch, env.dof_vel[:, 2], env.dof_vel[:, 5],
            contact[:, 0], contact[:, 1],
            env.dof_pos[:, 0]-env.dof_pos[:, 3], env.dof_pos[:, 1]-env.dof_pos[:, 4],
            (env.contact_forces[:, nonwheel_ids].norm(dim=-1) > 1.).any(dim=-1).float()), dim=-1)
        samples.append(values.detach().cpu())
        xy_samples.append(env.root_states[:, :2].detach().cpu().clone())
        positions = env.rigid_body_states[:, body_ids, :3] - env.root_states[:, None, :3]
        root_positions = quat_rotate_inverse(env.base_quat[:, None, :].expand(-1, 4, -1).reshape(-1, 4),
                                             positions.reshape(-1, 3)).reshape(-1, 4, 3)
        reflection = root_positions.new_tensor([1., -1., 1.])
        knee_error = root_positions[:, 0] - root_positions[:, 1] * reflection
        wheel_error = root_positions[:, 2] - root_positions[:, 3] * reflection
        geometry.append(torch.cat((knee_error, wheel_error), dim=1).detach().cpu())
        poses.append(env.dof_pos[:, [0, 1, 3, 4]].detach().cpu().clone())
        if opts.gui:
            time.sleep(max(0., env.dt-(time.monotonic()-step_started)))
    data = torch.stack(samples)
    xy = torch.stack(xy_samples)
    displacement = (xy - xy[0]).norm(dim=-1)
    if active_push is not None:
        push_results.append(active_push)
    names = ['vx_m_s', 'yaw_rad_s', 'height_m', 'roll_rad', 'pitch_rad',
             'left_wheel_rad_s', 'right_wheel_rad_s', 'left_contact', 'right_contact',
             'leg0_difference_rad', 'leg1_difference_rad', 'nonwheel_contact_fraction']
    metrics = {}
    for i, name in enumerate(names):
        v = data[:, :, i]
        metrics[name] = {'mean': v.mean().item(), 'mean_abs': v.abs().mean().item(),
                         'max_abs': v.abs().max().item(),
                         'temporal_std': v.std(dim=0).mean().item()}
    payload = {'checkpoint': str(opts.checkpoint.resolve()), 'seed': opts.seed,
        'profile': opts.profile,
        'position_drift': {
            'measurement_seconds': (len(xy)-1)*env.dt,
            'final_displacement_mean_m': displacement[-1].mean().item(),
            'final_displacement_max_m': displacement[-1].max().item(),
            'max_excursion_m': displacement.max().item(),
            'path_length_mean_m': (xy[1:]-xy[:-1]).norm(dim=-1).sum(dim=0).mean().item(),
            'valid_without_resets': bool((failures.sum()+timeouts.sum()).item()==0),
        },
        'push_delta_v_m_s': opts.push_delta_v, 'push_results': push_results,
        'recovery_definition': '0.5s continuously: tilt<0.15rad, height error<0.03m, horizontal speed<0.2m/s, both wheels contact, no nonwheel contact or reset since push',
        'command_height': opts.height, 'dof_names': env.dof_names,
        'mean_leg_q': torch.stack(poses).mean(dim=(0, 1)).tolist(),
        'geometry_root_frame': {name: {'mean_abs_xyz_m': torch.stack(geometry)[:, :, sl].abs().mean(dim=(0, 1)).tolist(),
                                     'mean_distance_m': torch.stack(geometry)[:, :, sl].norm(dim=-1).mean().item()}
                                for name, sl in [('knee', slice(0, 3)), ('wheel', slice(3, 6))]},
        'randomization_level': opts.randomization_level, 'num_envs': opts.num_envs,
        'seconds': opts.seconds, 'warmup': opts.warmup, 'policy_dt': env.dt,
        'failure_count': failures.sum().item(), 'timeout_count': timeouts.sum().item(),
        'survival_fraction': (failures == 0).float().mean().item(), 'metrics': metrics}
    opts.out.parent.mkdir(parents=True, exist_ok=True)
    opts.out.write_text(json.dumps(payload, indent=2)+'\n')
    print(json.dumps({'out': str(opts.out), 'failures': payload['failure_count'],
        'survival': payload['survival_fraction'],
        'mean_abs_vx': metrics['vx_m_s']['mean_abs'],
        'mean_abs_yaw': metrics['yaw_rad_s']['mean_abs'],
        'height': metrics['height_m']['mean'],
        'roll_std': metrics['roll_rad']['temporal_std'],
        'pitch_std': metrics['pitch_rad']['temporal_std'],
        'contacts': [metrics[n]['mean'] for n in ['left_contact', 'right_contact']]}))
    env.gym.destroy_sim(env.sim)


if __name__ == '__main__':
    main()
