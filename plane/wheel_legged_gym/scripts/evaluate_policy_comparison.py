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
from wheel_legged_gym.app.bootstrap import create_task_registry
from wheel_legged_gym.contracts.config_serialization import class_to_dict
from wheel_legged_gym.experiments.primitives import (
    apply_method_randomization,
)
from wheel_legged_gym.app.experiment_inputs import apply_training_profile
from wheel_legged_gym.app.experiment_inputs import apply_policy_experiment
from wheel_legged_gym.adapters.isaacgym.evaluation_setup import build_evaluation_args, disable_evaluation_randomization
from wheel_legged_gym.adapters.isaacgym.policy_io import load_policy, set_fixed_command_ranges
from wheel_legged_gym.domain.rewards.turn_lean import roll_reference


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    task_registry = create_task_registry()
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--commands', type=float, nargs='+', default=[0, -.5, .5, -1, 1])
    p.add_argument('--yaw-commands', type=float, nargs='+', help='Yaw paired with each vx; defaults to zero')
    p.add_argument('--height-commands', type=float, nargs='+', help='Base root height in metres, paired with vx; defaults to .4')
    p.add_argument('--initial-height-commands', type=float, nargs='+')
    p.add_argument('--initial-commands', type=float, nargs='+', help='Optional vx before one timed switch; no reset at switch')
    p.add_argument('--initial-yaw-commands', type=float, nargs='+')
    p.add_argument('--switch-at', type=float, default=5.)
    p.add_argument('--transition-ramp-seconds',type=float,default=0.,help='Explicit linear command ramp; zero preserves step protocol')
    p.add_argument('--yaw-delay-seconds', type=float, default=0., help='Delay yaw ramp after the speed ramp begins')
    p.add_argument('--yaw-exit-at', type=float, help='Begin a slow return to straight motion at this time')
    p.add_argument('--yaw-exit-ramp-seconds', type=float, default=2.)
    p.add_argument('--yaw-exit-factor', type=float, choices=[0., -1.], default=0.,
                   help='0 returns to straight; -1 slowly reverses yaw')
    p.add_argument('--lean-reference-max-deg', type=float, default=0.)
    p.add_argument('--trace-stride', type=int, default=10)
    p.add_argument('--envs-per-command', type=int, default=16)
    p.add_argument('--seed', type=int, default=19)
    p.add_argument('--randomization-level', type=int, choices=[0, 1], default=1)
    p.add_argument('--seconds', type=float, default=25)
    p.add_argument('--warmup', type=float, default=5)
    p.add_argument('--profile', choices=['method_v1','LEGACY_URDF'], default='method_v1')
    p.add_argument('--diagnostic-mode',choices=['sampled','sampled_noisy'],help='Diagnostic action/noise mode; never an acceptance run')
    a = p.parse_args(argv)
    if a.trace_stride<1:p.error('trace-stride must be positive')
    if a.height_commands is None:
        a.height_commands = [.4] * len(a.commands)
    if (len(a.height_commands) != len(a.commands) or not np.isfinite(a.height_commands).all()
            or not all(.30 <= h <= .45 for h in a.height_commands)):
        p.error('height commands must match vx count and lie in [.30,.45] metres')
    if a.yaw_commands is None:
        a.yaw_commands = [0.] * len(a.commands)
    if len(a.yaw_commands) != len(a.commands) or not np.isfinite(a.commands + a.yaw_commands).all():
        p.error('vx/yaw lists must have equal length and finite values')
    transition = a.initial_commands is not None
    if a.transition_ramp_seconds<0 or (a.transition_ramp_seconds and not transition):
        p.error('Nonnegative transition ramp requires initial commands')
    if (a.yaw_delay_seconds < 0 or a.yaw_exit_ramp_seconds <= 0 or
            not 0 <= a.lean_reference_max_deg <= 10 or
            (a.yaw_delay_seconds and not transition) or
            (a.yaw_exit_at is not None and not transition) or
            (a.yaw_exit_factor and a.yaw_exit_at is None)):
        p.error('Invalid staged turn protocol')
    if (a.initial_yaw_commands is not None or a.initial_height_commands is not None) and not transition:
        p.error('initial yaw requires initial commands')
    if transition:
        if a.initial_yaw_commands is None:
            a.initial_yaw_commands = [0.] * len(a.commands)
        if a.initial_height_commands is None:
            a.initial_height_commands = a.height_commands
        if (len(a.initial_height_commands) != len(a.commands) or
                not all(.30 <= h <= .45 for h in a.initial_height_commands)):
            p.error('initial heights must match vx count and lie in [.30,.45]')
        if (len(a.initial_commands) != len(a.commands) or len(a.initial_yaw_commands) != len(a.commands)
                or not np.isfinite(a.initial_commands + a.initial_yaw_commands).all()
                or not 0 < a.switch_at < a.warmup):
            p.error('initial lists must match target lists; require 0 < switch-at < warmup')
    if not 0 <= a.warmup < a.seconds or a.envs_per_command < 1:
        p.error('require seconds > warmup >= 0 and positive environment count')
    if transition and a.switch_at+a.transition_ramp_seconds+a.yaw_delay_seconds>=a.warmup:
        p.error('Steady warmup must end after the command ramp')
    if a.yaw_exit_at is not None and not a.warmup < a.yaw_exit_at < a.seconds-a.yaw_exit_ramp_seconds:
        p.error('Yaw exit must follow steady warmup and finish before evaluation end')
    if a.out.exists():
        raise FileExistsError(a.out)
    cfg, train = task_registry.get_cfgs(name='wheel_legged')
    if a.profile == 'LEGACY_URDF':
        apply_policy_experiment(cfg, 'LEGACY_URDF', train)
    else:
        apply_training_profile(cfg, train, phase='translate', level=0)
    disable_evaluation_randomization(cfg)
    apply_method_randomization(cfg, a.randomization_level)
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
    if a.diagnostic_mode:cfg.noise.add_noise=a.diagnostic_mode=='sampled_noisy'
    args = build_evaluation_args()
    args.num_envs, args.seed = cfg.env.num_envs, a.seed
    env, _ = task_registry.make_env(name='wheel_legged', args=args, env_cfg=cfg)
    try:
        commands = np.repeat(np.array(list(zip(a.commands, a.yaw_commands, a.height_commands))),
                             a.envs_per_command, axis=0)
        initial_commands = (np.repeat(np.array(list(zip(a.initial_commands, a.initial_yaw_commands,
            a.initial_height_commands))), a.envs_per_command, axis=0)
            if transition else commands)
        set_fixed_command_ranges(env, initial_commands)
        env.reset()
        obs, history = env.get_observations()
        target = torch.tensor(initial_commands, device=env.device, dtype=torch.float)
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
                 'knee_mirror_m', 'wheel_mirror_m', 'encoder_vx_mae', 'action_clip', 'yaw_mae',
                 'roll', 'roll_target', 'pitch', 'lateral_velocity', 'abs_lateral_velocity',
                 'curvature', 'lateral_accel_command', 'lateral_accel_actual',
                 'left_vertical_force', 'right_vertical_force']
        sums = torch.zeros(env.num_envs, len(names), device=env.device)
        maxima = torch.zeros_like(sums)
        count = 0
        joint_sums = torch.zeros(env.num_envs,6,device=env.device)
        joint_min = torch.full_like(joint_sums,float('inf'))
        joint_max = torch.full_like(joint_sums,-float('inf'))
        torque_max = torch.zeros_like(joint_sums)
        response_trace = []
        travelled=torch.zeros(env.num_envs,device=env.device)
        for step in range(round(a.seconds / env.dt)):
            switch_step=round(a.switch_at/env.dt)
            ramp_steps=round(a.transition_ramp_seconds/env.dt)
            yaw_end_step = switch_step + ramp_steps + round(a.yaw_delay_seconds/env.dt)
            exit_step = round(a.yaw_exit_at/env.dt) if a.yaw_exit_at is not None else None
            exit_ramp_steps = round(a.yaw_exit_ramp_seconds/env.dt)
            if transition and (switch_step<=step<=yaw_end_step or
                               (exit_step is not None and step>=exit_step)):
                fraction=1. if ramp_steps==0 else min(1.,(step-switch_step)/ramp_steps)
                actual=initial_commands+(commands-initial_commands)*fraction
                yaw_fraction=1. if ramp_steps==0 else min(1.,max(0.,(step-switch_step-round(a.yaw_delay_seconds/env.dt))/ramp_steps))
                actual[:, 1]=initial_commands[:, 1]+(commands[:, 1]-initial_commands[:, 1])*yaw_fraction
                if exit_step is not None and step>=exit_step:
                    exit_fraction=min(1.,(step-exit_step)/exit_ramp_steps)
                    actual[:, 1]=commands[:, 1]*(1.+(a.yaw_exit_factor-1.)*exit_fraction)
                set_fixed_command_ranges(env, actual)
                target = torch.tensor(actual, device=env.device, dtype=torch.float)
                env.commands[:, :3] = target
                # Replace only current command channels, preserving the four past frames.
                # Recomputing observations here would append an extra history frame.
                obs[:, 6:9] = target * env.commands_scale
                history[:, -25+6:-25+9] = obs[:, 6:9]
            if not torch.allclose(env.commands[:, :3], target):
                raise RuntimeError('Actual command changed during fixed-command evaluation')
            if not torch.allclose(obs[:, 6:9], target * env.commands_scale):
                raise RuntimeError('Command is not present in policy observation')
            with torch.inference_mode():
                if a.diagnostic_mode:
                    action=policy.act(obs,history)
                    latent=policy.latent
                else:
                    action, latent = policy.act_inference(obs, history)
                encoder_error = (env.base_lin_vel[:, 0] - latent[:, 0]/env.obs_scales.lin_vel).abs()
            if not torch.isfinite(action).all():
                raise RuntimeError('Nonfinite policy action')
            before_saturation = env.command_metrics.preclip_torque_saturation_sum.clone()
            before_xy=env.root_states[:,:2].clone()
            obs, _, _, dones, info, history = env.step(action)
            timeout = info.get('time_outs', torch.zeros_like(dones)).bool()
            failures += (dones.bool() & ~timeout).float()
            timeouts += (dones.bool() & timeout).float()
            body_contact = (env.contact_forces[:, nonwheel].norm(dim=-1) > 1).any(dim=-1).float()
            nonwheel_full += body_contact
            if transition and step>=round(a.switch_at/env.dt):
                travelled+=(env.root_states[:,:2]-before_xy).norm(dim=-1)
            if transition and step % a.trace_stride == 0:
                response_trace.append({'time': (step+1)*env.dt,
                    'applied_command':target.detach().cpu().tolist(),
                    'policy_command_scaled':obs[:, 6:9].detach().cpu().tolist(),
                    'history_latest_command_scaled':history[:, -19:-16].detach().cpu().tolist(),
                    'wheel_vertical_force_n':env.contact_forces[:,env.feet_indices,2].cpu().tolist(),
                    'path_since_switch_m':travelled.cpu().tolist(),
                    'wheel_contacts':(env.contact_forces[:,env.feet_indices,2]>1).cpu().tolist(),
                    'roll':torch.atan2(-env.projected_gravity[:,1],-env.projected_gravity[:,2]).cpu().tolist(),
                    'roll_target':(roll_reference(target[:,0],target[:,1],
                        a.lean_reference_max_deg*np.pi/180).cpu().tolist()
                        if a.lean_reference_max_deg else [0.]*env.num_envs),
                    'resets':(failures+timeouts).cpu().tolist(),
                    'height': env.base_height.detach().cpu().tolist(),
                    'vx': env.base_lin_vel[:, 0].detach().cpu().tolist(),
                    'vy': env.base_lin_vel[:, 1].detach().cpu().tolist(),
                    'yaw': env.base_ang_vel[:, 2].detach().cpu().tolist(),
                    'base_xy':env.root_states[:, :2].detach().cpu().tolist()})
            # Counters reset with episodes; post-reset samples are not valid
            # torque observations. Any reset independently fails acceptance.
            sat = (env.command_metrics.preclip_torque_saturation_sum - before_saturation).clamp(min=0)
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
            signed_speed = torch.where(vx >= 0, vx.clamp(min=.1), vx.clamp(max=-.1))
            roll_target=(roll_reference(target[:,0],target[:,1],a.lean_reference_max_deg*np.pi/180)
                         if a.lean_reference_max_deg else torch.zeros_like(vx))
            if getattr(env, '_method_v1', False):
                slip_metric = env._method_wheel_terms()['residual_rms']
            else:
                slip_metric = torch.abs(vx.unsqueeze(1) + .06 * env.dof_vel[:, [2, 5]]).mean(dim=1)
            values = torch.stack([vx, (vx-target[:, 0]).abs(), vx.abs(), yaw, yaw.abs(),
                env.base_height, (env.base_height-target[:, 2]).abs(), roll.abs(), pitch.abs(),
                contact[:, 0], contact[:, 1], body_contact,
                slip_metric, sat,
                (local[:, 0]-local[:, 1]*mirror).norm(dim=-1),
                (local[:, 2]-local[:, 3]*mirror).norm(dim=-1), encoder_error,
                (action.abs() > env.cfg.normalization.clip_actions).float().mean(dim=-1),
                (yaw-target[:, 1]).abs(), roll, roll_target, pitch,
                env.base_lin_vel[:,1], env.base_lin_vel[:,1].abs(),
                torch.where(vx.abs()>.1,yaw/signed_speed,torch.zeros_like(vx)),
                target[:,0]*target[:,1],vx*yaw,
                env.contact_forces[:,env.feet_indices[0],2].clamp_min(0),
                env.contact_forces[:,env.feet_indices[1],2].clamp_min(0)], dim=-1)
            if not torch.isfinite(values).all():
                raise RuntimeError('Nonfinite physics metrics')
            sums += values
            joint_sums += env.dof_pos
            joint_min = torch.minimum(joint_min,env.dof_pos)
            joint_max = torch.maximum(joint_max,env.dof_pos)
            torque_max = torch.maximum(torque_max,env.torques.abs())
            maxima = torch.maximum(maxima, values.abs())
            count += 1
        means = (sums / count).cpu()
        results = []
        for i, vx in enumerate(a.commands):
            sl = slice(i*a.envs_per_command, (i+1)*a.envs_per_command)
            metrics = {n: means[sl, j].mean().item() for j, n in enumerate(names)}
            results.append({'command': [vx, a.yaw_commands[i], a.height_commands[i]], 'metrics': metrics,
                'joint_diagnostics':{'names':list(env.dof_names),
                    'mean_positions_rad':(joint_sums[sl]/count).mean(0).cpu().tolist(),
                    'min_positions_rad':joint_min[sl].min(0).values.cpu().tolist(),
                    'max_positions_rad':joint_max[sl].max(0).values.cpu().tolist(),
                    'max_applied_torque_nm':torque_max[sl].max(0).values.cpu().tolist(),
                    'scope':'Post-warmup positions; torque snapshots at policy steps, not all physics substeps. Wheel position is accumulated angle.'},
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
        if transition:
            payload.update(schema='matched_policy_transition_v1',
                trace_dt=env.dt*a.trace_stride,
                switch_at=round(a.switch_at/env.dt)*env.dt,
                transition_ramp_seconds=round(a.transition_ramp_seconds/env.dt)*env.dt,
                yaw_delay_seconds=round(a.yaw_delay_seconds/env.dt)*env.dt,
                yaw_exit_at=a.yaw_exit_at,
                yaw_exit_ramp_seconds=a.yaw_exit_ramp_seconds if a.yaw_exit_at is not None else None,
                yaw_exit_factor=a.yaw_exit_factor if a.yaw_exit_at is not None else None,
                lean_reference_max_deg=a.lean_reference_max_deg,
                initial_commands=list(map(list,zip(a.initial_commands,a.initial_yaw_commands,a.initial_height_commands))),
                response_trace=response_trace,
                transition_note='No reset at switch; 10 Hz per-environment response trace. Metrics and gate cover post-warmup steady state, not transient smoothness.')
        a.out.parent.mkdir(parents=True, exist_ok=True)
        if a.diagnostic_mode:
            payload.update(diagnostic_only=True,deterministic=False,noise=a.diagnostic_mode=='sampled_noisy',
                noise_mode=a.diagnostic_mode,limitation='Noise draws also affect RNG/reset trajectory; same seed does not guarantee identical physical initial states. Read recorded states.')
        a.out.write_text(json.dumps(payload, indent=2)+'\n')
        print(json.dumps({'out': str(a.out), 'results': [{k:v for k,v in r.items()
            if k in ['command','metrics','failure_count','timeout_count']} for r in results]}))
    finally:
        env.gym.destroy_sim(env.sim)


if __name__ == '__main__':
    main()
