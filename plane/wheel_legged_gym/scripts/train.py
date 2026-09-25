# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np
import os
import json
import sys
from datetime import datetime

import isaacgym
from wheel_legged_gym.app.bootstrap import create_task_registry
from wheel_legged_gym.app.arguments import get_args
import torch
from pathlib import Path

from wheel_legged_gym.app.experiment_inputs import apply_training_profile
from wheel_legged_gym.app.experiment_inputs import apply_policy_experiment
from wheel_legged_gym.app.optimizer_overrides import enforce_optimizer_overrides
from wheel_legged_gym.adapters.artifacts.experiment_manifest import write_experiment_manifest


def train(args):
    task_registry = create_task_registry()
    if str(args.policy_experiment).upper() == 'FUDAN_STAND' and args.resume:
        raise ValueError('FUDAN_STAND reproduction starts from scratch; use a new run without --resume')
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    matching_resume = False
    if str(args.policy_experiment or "").lower() in {"method_v1", "method"}:
        manifest = apply_training_profile(
            env_cfg,
            train_cfg,
            profile="method_v1",
            phase=args.phase,
            level=args.command_level,
        )
        if args.resume and getattr(args, "resume_mode", "full") == "full":
            # Full resume is safe only within the same recorded experiment.
            # Validate before allocating the simulator or loading optimizer state.
            from wheel_legged_gym import WHEEL_LEGGED_GYM_ROOT_DIR
            from wheel_legged_gym.adapters.artifacts.checkpoints import get_load_path
            root = Path(WHEEL_LEGGED_GYM_ROOT_DIR) / 'logs' / (args.experiment_name or train_cfg.runner.experiment_name)
            checkpoint = Path(get_load_path(str(root), load_run=args.load_run, checkpoint=args.checkpoint))
            previous = json.loads((checkpoint.parent / 'policy_experiment.json').read_text())
            expected = {key: manifest[key] for key in ('profile', 'phase', 'level', 'reward_version', 'reward_scales', 'optimizer')}
            expected.update(randomization_level=env_cfg.domain_rand_level,
                            symmetry_loss_coef=train_cfg.algorithm.symmetry_loss_coef)
            mismatched = [key for key, value in expected.items() if previous.get(key) != value]
            if mismatched:
                raise ValueError(f'Full resume configuration mismatch: {mismatched}; use --resume_mode policy for migration')
            matching_resume = True
    else:
        manifest = apply_policy_experiment(env_cfg, args.policy_experiment, train_cfg)
    if str(args.policy_experiment).upper() in {'MOTION_GOAL','HEIGHT_COURSE','TURN_ENVELOPE'}:
        from wheel_legged_gym import WHEEL_LEGGED_GYM_ROOT_DIR
        from wheel_legged_gym.adapters.artifacts.checkpoints import get_load_path
        from wheel_legged_gym.app.experiment_inputs import validate_motion_source as validate_source
        if str(args.policy_experiment).upper() == 'HEIGHT_COURSE':
            from wheel_legged_gym.app.experiment_inputs import validate_height_source as validate_source
        if str(args.policy_experiment).upper() == 'TURN_ENVELOPE':
            from wheel_legged_gym.app.experiment_inputs import validate_turn_source as validate_source
        root=Path(WHEEL_LEGGED_GYM_ROOT_DIR)/'logs'/(args.experiment_name or train_cfg.runner.experiment_name)
        checkpoint=Path(get_load_path(str(root),load_run=args.load_run,checkpoint=args.checkpoint))
        manifest.update(validate_source(checkpoint,args.resume,args.resume_mode,env_cfg))
        matching_resume=True
    if str(args.policy_experiment).upper() in {'LEGACY_ANCHORS','LEGACY_SPEED2','LEGACY_SPEED2_STOP','LEGACY_YAW'}:
        from wheel_legged_gym import WHEEL_LEGGED_GYM_ROOT_DIR
        from wheel_legged_gym.adapters.artifacts.checkpoints import get_load_path
        from wheel_legged_gym.adapters.artifacts.legacy_sources import validate_legacy_anchors_source as validate_source
        if str(args.policy_experiment).upper() == 'LEGACY_SPEED2':
            from wheel_legged_gym.adapters.artifacts.legacy_sources import validate_legacy_speed2_source as validate_source
        if str(args.policy_experiment).upper() == 'LEGACY_SPEED2_STOP':
            from wheel_legged_gym.adapters.artifacts.legacy_sources import validate_legacy_speed2_stop_source as validate_source
        if str(args.policy_experiment).upper() == 'LEGACY_YAW':
            from wheel_legged_gym.adapters.artifacts.legacy_sources import validate_legacy_yaw_source as validate_source
        root=Path(WHEEL_LEGGED_GYM_ROOT_DIR)/'logs'/(args.experiment_name or train_cfg.runner.experiment_name)
        checkpoint=Path(get_load_path(str(root),load_run=args.load_run,checkpoint=args.checkpoint))
        manifest.update(validate_source(checkpoint,args.resume,args.resume_mode,env_cfg))
        matching_resume=True
    if str(args.policy_experiment).upper() in {'EXPLORE_STOP_RETENTION','EXPLORE_CLEAN_OBS'}:
        from wheel_legged_gym import WHEEL_LEGGED_GYM_ROOT_DIR
        from wheel_legged_gym.adapters.artifacts.checkpoints import get_load_path
        from wheel_legged_gym.adapters.artifacts.legacy_sources import validate_stop_retention_source as validate_source
        root = Path(WHEEL_LEGGED_GYM_ROOT_DIR) / 'logs' / (args.experiment_name or train_cfg.runner.experiment_name)
        checkpoint = Path(get_load_path(str(root), load_run=args.load_run, checkpoint=args.checkpoint))
        manifest.update(validate_source(checkpoint,args.resume,args.resume_mode,manifest))
    if str(args.policy_experiment).upper() in {'H3_SPEED1', 'ENCODER_FROZEN', 'ENCODER_UPDATING', 'ENCODER_ANCHORED', 'ANCHORED_WHEEL_EXPLORE'}:
        from wheel_legged_gym import WHEEL_LEGGED_GYM_ROOT_DIR
        from wheel_legged_gym.adapters.artifacts.checkpoints import get_load_path
        from wheel_legged_gym.adapters.artifacts.legacy_sources import validate_h3_speed1_source as validate_source
        root = Path(WHEEL_LEGGED_GYM_ROOT_DIR) / 'logs' / (args.experiment_name or train_cfg.runner.experiment_name)
        checkpoint = Path(get_load_path(str(root), load_run=args.load_run, checkpoint=args.checkpoint))
        manifest.update(validate_source(checkpoint, args.resume, args.resume_mode, manifest))
    if str(args.policy_experiment).upper() == 'H3_LOW_SPEED':
        from wheel_legged_gym import WHEEL_LEGGED_GYM_ROOT_DIR
        from wheel_legged_gym.adapters.artifacts.checkpoints import get_load_path
        from wheel_legged_gym.adapters.artifacts.legacy_sources import validate_h3_low_speed_source as validate_source
        root = Path(WHEEL_LEGGED_GYM_ROOT_DIR) / 'logs' / (args.experiment_name or train_cfg.runner.experiment_name)
        checkpoint = Path(get_load_path(str(root), load_run=args.load_run, checkpoint=args.checkpoint))
        manifest.update(validate_source(checkpoint, args.resume, args.resume_mode))
    if str(args.policy_experiment).upper() in {'LOW_SPEED', 'LOW_SPEED_TRACKING'}:
        if not args.resume or args.resume_mode != 'full':
            raise ValueError('LOW_SPEED requires explicit full-state warm start')
        from wheel_legged_gym import WHEEL_LEGGED_GYM_ROOT_DIR
        from wheel_legged_gym.adapters.artifacts.checkpoints import get_load_path
        root = Path(WHEEL_LEGGED_GYM_ROOT_DIR) / 'logs' / (args.experiment_name or train_cfg.runner.experiment_name)
        checkpoint = Path(get_load_path(str(root), load_run=args.load_run, checkpoint=args.checkpoint))
        previous = json.loads((checkpoint.parent/'policy_experiment.json').read_text())
        if (previous.get('reward_pipeline') != 'normalized_v1'
            or previous.get('name') not in {'STAND_SYMMETRIC', 'LOW_SPEED', 'LOW_SPEED_TRACKING'}
            or previous.get('randomization_level') != 1):
            raise ValueError('LOW_SPEED requires compatible standing or low-speed source')
        manifest['source_checkpoint'] = str(checkpoint)
        manifest['source_reward_scales'] = previous['reward_scales']
    if str(args.policy_experiment).upper() in {'STAND_CONTROL', 'STAND_SYMMETRIC'}:
        if not args.resume or args.resume_mode != 'full':
            raise ValueError('Standing controlled fine-tune requires --resume --resume_mode=full')
        from wheel_legged_gym import WHEEL_LEGGED_GYM_ROOT_DIR
        from wheel_legged_gym.adapters.artifacts.checkpoints import get_load_path
        root = Path(WHEEL_LEGGED_GYM_ROOT_DIR) / 'logs' / (args.experiment_name or train_cfg.runner.experiment_name)
        checkpoint = Path(get_load_path(str(root), load_run=args.load_run, checkpoint=args.checkpoint))
        previous = json.loads((checkpoint.parent / 'policy_experiment.json').read_text())
        if previous.get('reward_pipeline') != 'normalized_v1' or previous.get('phase') != 'stand':
            raise ValueError('Controlled stand fine-tune requires a normalized_v1 standing checkpoint')
        old_scales = previous['reward_scales']
        new_scales = manifest['reward_scales']
        mismatch = [k for k in set(old_scales) | set(new_scales)
                    if k != 'stand_bilateral_geometry' and old_scales.get(k, 0.) != new_scales.get(k, 0.)]
        if mismatch or previous.get('randomization_level') != 1:
            raise ValueError(f'Control rewards/randomization differ from source: {mismatch}')
        manifest['source_checkpoint'] = str(checkpoint)
    env, env_cfg = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg
    )
    if matching_resume:
        # Optimizer moments and its adaptive learning rate came from checkpoint.
        ppo_runner.alg.learning_rate = ppo_runner.alg.optimizer.param_groups[0]['lr']
    else:
        enforce_optimizer_overrides(ppo_runner, manifest)
    if str(args.policy_experiment).upper() == 'H3_LOW_SPEED':
        from wheel_legged_gym.adapters.artifacts.checkpoint_migration import verify_and_restore_std
        manifest['migration_verification'] = verify_and_restore_std(ppo_runner, checkpoint)
    task_registry.save_cfgs(name=args.task)
    if str(args.policy_experiment).upper() in {'MOTION_GOAL','HEIGHT_COURSE','TURN_ENVELOPE'}:
        from wheel_legged_gym.adapters.artifacts.checkpoint_migration import verify_full_resume
        spec = manifest.get('turn_spec',manifest.get('height_spec',manifest.get('motion_goal_spec')))
        manifest['resume_verification']=verify_full_resume(ppo_runner,checkpoint,
            expected_iteration=spec['source_iteration'])
        manifest['effective_start_learning_rate']=ppo_runner.alg.learning_rate
        if manifest.get('freeze_motion_encoder',False):
            ppo_runner.alg.freeze_encoder_updates=True
        if manifest.get('dynamic_fixed_lr'):
            manifest['restored_ppo_learning_rate']=ppo_runner.alg.learning_rate
            ppo_runner.alg.learning_rate=manifest['dynamic_fixed_lr']
            ppo_runner.alg.schedule='fixed'
            for group in ppo_runner.alg.optimizer.param_groups:group['lr']=manifest['dynamic_fixed_lr']
            manifest['effective_start_learning_rate']=ppo_runner.alg.learning_rate
            manifest['optimizer'].update(learning_rate=ppo_runner.alg.learning_rate,schedule='fixed')
        if manifest.get('dynamic_reference_coef'):
            import copy
            ppo_runner.alg.reference_policy=copy.deepcopy(ppo_runner.alg.actor_critic).eval()
            for parameter in ppo_runner.alg.reference_policy.parameters():parameter.requires_grad_(False)
            ppo_runner.alg.reference_coef=manifest['dynamic_reference_coef']
            ppo_runner.alg.reference_dynamic_stride=(env_cfg.commands.turn_stride
                if manifest.get('turn_spec') else env_cfg.commands.start_stop_stride)
            ppo_runner.alg.storage.track_minibatch_env_ids=True
            manifest['reference_checkpoint_sha256']=manifest['source_checkpoint_sha256']
    if manifest.get('turn_spec'):
        manifest['training_command'] = [sys.executable] + sys.argv
    if str(args.policy_experiment).upper() in {'LEGACY_ANCHORS','LEGACY_SPEED2','LEGACY_SPEED2_STOP','LEGACY_YAW'}:
        from wheel_legged_gym.adapters.artifacts.checkpoint_migration import verify_full_resume
        manifest['resume_verification']=verify_full_resume(ppo_runner,checkpoint,
            expected_iteration={'LEGACY_ANCHORS':500,'LEGACY_SPEED2':700,'LEGACY_SPEED2_STOP':900,'LEGACY_YAW':1400}[str(args.policy_experiment).upper()])
        manifest['effective_start_learning_rate']=ppo_runner.alg.learning_rate
    if str(args.policy_experiment).upper() in {'EXPLORE_STOP_RETENTION','EXPLORE_CLEAN_OBS'}:
        from wheel_legged_gym.adapters.artifacts.checkpoint_migration import verify_full_resume
        manifest['resume_verification']=verify_full_resume(ppo_runner,checkpoint,expected_iteration=500)
    if str(args.policy_experiment).upper() in {'H3_SPEED1', 'ENCODER_FROZEN', 'ENCODER_UPDATING', 'ENCODER_ANCHORED', 'ANCHORED_WHEEL_EXPLORE'}:
        from wheel_legged_gym.adapters.artifacts.checkpoint_migration import verify_full_resume
        manifest['resume_verification'] = verify_full_resume(ppo_runner, checkpoint)
    if str(args.policy_experiment).upper() == 'ANCHORED_WHEEL_EXPLORE':
        model = ppo_runner.alg.actor_critic
        manifest['std_before_override'] = model.std.detach().cpu().tolist()
        with torch.no_grad():
            model.std[2].copy_(model.std[5])
        manifest['std_after_override'] = model.std.detach().cpu().tolist()
    if manifest.get('command_diagnostics'):
        from wheel_legged_gym.utils.command_diagnostics import CommandDiagnostics
        ppo_runner.command_diagnostics = CommandDiagnostics(env, task_registry.log_dir)
        ppo_runner.alg.freeze_encoder_updates = manifest['freeze_encoder_updates']
        ppo_runner.alg.encoder_action_anchor_coef = manifest.get('encoder_action_anchor_coef', 0.0)
        ppo_runner.alg.encoder_ablation_diagnostics = True
    write_experiment_manifest(
        Path(task_registry.log_dir) / "policy_experiment.json",
        manifest,
        env_cfg,
        train_cfg,
        args,
    )
    ppo_runner.learn(
        num_learning_iterations=train_cfg.runner.max_iterations,
        init_at_random_ep_len=True,
    )


if __name__ == "__main__":
    args = get_args()
    train(args)
