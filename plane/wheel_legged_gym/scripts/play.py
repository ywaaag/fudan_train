# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA
# SPDX-License-Identifier: BSD-3-Clause

import os
import threading
import numpy as np

import isaacgym
import torch

from wheel_legged_gym import WHEEL_LEGGED_GYM_ROOT_DIR
from wheel_legged_gym.envs import *
from wheel_legged_gym.utils import get_args, export_policy_as_jit, task_registry
from wheel_legged_gym.utils.web_panel import TelemetryBuffer, WebPanelServer

try:
    from pynput import keyboard
except ImportError:
    print("Missing dependency: pynput. Please install it with: pip install pynput")
    raise


# --------------------
# Global command state
# --------------------
cmd_x = 0.0
ang_vel = 0.0
cmd_height = 0.40
running = True
turn_left_pressed = False
turn_right_pressed = False
command_source = "keyboard"
command_lock = threading.RLock()
runtime_limits = None


def _panel_range(env_min_name, env_max_name, fallback):
    """Read an optional runtime panel range without changing training config."""
    lo = float(os.environ.get(env_min_name, fallback[0]))
    hi = float(os.environ.get(env_max_name, fallback[1]))
    if lo > hi:
        raise ValueError(f"{env_min_name} must be <= {env_max_name}")
    return [lo, hi]


LIN_VEL_CMD = float(os.environ.get("WHEEL_LEG_PLAY_LIN_VEL", "2.0"))
YAW_STEP = float(os.environ.get("WHEEL_LEG_PLAY_YAW_RATE", "2.0"))
HEIGHT_STEP = 0.02

# Initial viewer camera. Applied once after the environments are created.
INITIAL_CAMERA_POSITION = [20.0, -20.0, 10.0]
INITIAL_CAMERA_LOOK_AT = [20.0, 40.0, 0.0]



def update_yaw_cmd():
    global ang_vel
    if turn_left_pressed and not turn_right_pressed:
        ang_vel = YAW_STEP
    elif turn_right_pressed and not turn_left_pressed:
        ang_vel = -YAW_STEP
    else:
        ang_vel = 0.0


def get_command_state():
    with command_lock:
        return {
            "cmd_x": float(cmd_x),
            "ang_vel": float(ang_vel),
            "cmd_height": float(cmd_height),
            "source": command_source,
        }


def get_running_state():
    with command_lock:
        return bool(running)


def set_panel_limits(new_limits):
    """Update runtime command ranges used by the panel and command clamp."""
    global runtime_limits
    with command_lock:
        runtime_limits = {
            "cmd_x": [float(new_limits["cmd_x"][0]), float(new_limits["cmd_x"][1])],
            "ang_vel": [float(new_limits["ang_vel"][0]), float(new_limits["ang_vel"][1])],
            "cmd_height": [
                float(new_limits["cmd_height"][0]),
                float(new_limits["cmd_height"][1]),
            ],
        }
        return {key: list(values) for key, values in runtime_limits.items()}


def set_panel_command(payload):
    """Apply a partial command update from the local web panel."""
    global cmd_x, ang_vel, cmd_height, command_source
    global turn_left_pressed, turn_right_pressed
    with command_lock:
        had_held_turn = turn_left_pressed or turn_right_pressed
        if "cmd_x" in payload:
            cmd_x = float(payload["cmd_x"])
        if "ang_vel" in payload:
            ang_vel = float(payload["ang_vel"])
        if "cmd_height" in payload:
            cmd_height = float(payload["cmd_height"])
        # A panel/gamepad command supersedes a held A/D key until the next
        # physical key event, preventing stale keyboard state from winning.
        turn_left_pressed = False
        turn_right_pressed = False
        if had_held_turn and "ang_vel" not in payload:
            ang_vel = 0.0
        command_source = str(payload.get("source", "panel"))[:32]
        return get_command_state()


def on_press(key):
    global cmd_x, ang_vel, cmd_height, running, command_source
    global turn_left_pressed, turn_right_pressed

    if key == keyboard.Key.esc:
        with command_lock:
            running = False
        print("[CMD] quit (ESC)")
        return False

    try:
        k = key.char.lower()
    except Exception:
        return

    if k == "q":
        with command_lock:
            running = False
        print("[CMD] quit (q)")
        return False

    with command_lock:
        command_source = "keyboard"
        if k == "w":
            cmd_x = LIN_VEL_CMD
            print(f"[CMD] forward: x={cmd_x:.2f}")
        elif k == "s":
            cmd_x = -LIN_VEL_CMD
            print(f"[CMD] backward: x={cmd_x:.2f}")
        elif k == "a":
            if not turn_left_pressed:
                print("[CMD] turn left (hold)")
            turn_left_pressed = True
            update_yaw_cmd()
        elif k == "d":
            if not turn_right_pressed:
                print("[CMD] turn right (hold)")
            turn_right_pressed = True
            update_yaw_cmd()
        elif k == "e":
            cmd_x = 0.0
            turn_left_pressed = False
            turn_right_pressed = False
            update_yaw_cmd()
            print("[CMD] stop")
        elif k == "x":
            cmd_height += HEIGHT_STEP
            print(f"[CMD] height up: h={cmd_height:.2f}")
        elif k == "c":
            cmd_height -= HEIGHT_STEP
            print(f"[CMD] height down: h={cmd_height:.2f}")


def on_release(key):
    global turn_left_pressed, turn_right_pressed, command_source
    try:
        k = key.char.lower()
    except Exception:
        return

    with command_lock:
        command_source = "keyboard"
        if k == "a":
            turn_left_pressed = False
            update_yaw_cmd()
        elif k == "d":
            turn_right_pressed = False
            update_yaw_cmd()
    return


def apply_manual_commands(env, env_cfg):
    global cmd_x, ang_vel, cmd_height

    with command_lock:
        limits = runtime_limits or {
            "cmd_x": env_cfg.commands.ranges.lin_vel_x,
            "ang_vel": env_cfg.commands.ranges.ang_vel_yaw,
            "cmd_height": env_cfg.commands.ranges.height,
        }
        cmd_x = float(
            np.clip(
                cmd_x,
                limits["cmd_x"][0],
                limits["cmd_x"][1],
            )
        )
        ang_vel = float(
            np.clip(
                ang_vel,
                limits["ang_vel"][0],
                limits["ang_vel"][1],
            )
        )
        cmd_height = float(
            np.clip(
                cmd_height,
                limits["cmd_height"][0],
                limits["cmd_height"][1],
            )
        )
        local_cmd_x, local_ang_vel, local_cmd_height = cmd_x, ang_vel, cmd_height

    env.commands[:, 2] = local_cmd_height

    jump_ids = getattr(env, "jump_ramp_idx", None)
    if jump_ids is None or len(jump_ids) == 0:
        env.commands[:, 0] = local_cmd_x
        env.commands[:, 1] = local_ang_vel
        return

    manual_mask = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    manual_mask[jump_ids] = False
    manual_ids = manual_mask.nonzero(as_tuple=False).flatten()
    if len(manual_ids) != 0:
        env.commands[manual_ids, 0] = local_cmd_x
        env.commands[manual_ids, 1] = local_ang_vel

    env.commands[jump_ids, 0] = env_cfg.commands.jump_ramp_lin_vel_x
    env.commands[jump_ids, 2] = env_cfg.commands.jump_ramp_height
    env.commands[jump_ids, 3] = env_cfg.commands.jump_ramp_heading


def play(args):
    global running, runtime_limits

    print("\n====== Keyboard Control Mode (NO Enter) ======")
    print("w      : forward")
    print("s      : backward")
    print("a      : hold to turn left")
    print("d      : hold to turn right")
    print("e      : stop")
    print("x      : height up")
    print("c      : height down")
    print("q/ESC  : quit")
    print("camera : fixed overview")
    print("=============================================\n")

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50)
    env_cfg.env.episode_length_s = 20
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 10
    env_cfg.terrain.max_init_terrain_level = env_cfg.terrain.num_rows - 1
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.lift_robots = False
    env_cfg.domain_rand.downward_impulse_robots = False
    env_cfg.domain_rand.downward_impulse_interval_s = 3
    env_cfg.domain_rand.downward_impulse_vel_range = [2.4, 2.8]
    # env_cfg.domain_rand.vmc_force_events = True
    env_cfg.terrain.curriculum = True
    # Playback commands come from the keyboard/panel, not the training
    # sampler.  Keep the sampler from overwriting a manually selected target
    # every few seconds (resets are still handled by the normal env logic).
    env_cfg.commands.resampling_time = 1.0e9

    # The checkpoint was trained with a narrow height range ([0.40, 0.40]),
    # which made the original panel height slider appear locked.  These are
    # play-time limits only; the training configuration is not modified.
    with command_lock:
        runtime_limits = {
            "cmd_x": _panel_range(
                "WHEEL_LEG_PANEL_X_MIN",
                "WHEEL_LEG_PANEL_X_MAX",
                [-2.0, 2.0],
            ),
            "ang_vel": _panel_range(
                "WHEEL_LEG_PANEL_YAW_MIN",
                "WHEEL_LEG_PANEL_YAW_MAX",
                [-2.0, 2.0],
            ),
            "cmd_height": _panel_range(
                "WHEEL_LEG_PANEL_HEIGHT_MIN",
                "WHEEL_LEG_PANEL_HEIGHT_MAX",
                [0.10, 1.00],
            ),
        }

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    if getattr(env, "viewer", None) is not None:
        env.set_camera(INITIAL_CAMERA_POSITION, INITIAL_CAMERA_LOOK_AT)

    apply_manual_commands(env, env_cfg)
    obs, obs_history = env.get_observations()

    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg
    )
    policy = ppo_runner.get_inference_policy(device=env.device)
    is_sequence_policy = bool(ppo_runner.alg.actor_critic.is_sequence)

    if EXPORT_POLICY:
        path = os.path.join(
            WHEEL_LEGGED_GYM_ROOT_DIR,
            "logs",
            train_cfg.runner.experiment_name,
            "exported",
            "policies",
        )
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print("Exported policy to:", path)

    telemetry = TelemetryBuffer(maxlen=2400)
    panel = None
    panel_enabled = os.environ.get("WHEEL_LEG_PANEL", "1").lower() not in {
        "0",
        "false",
        "no",
        "off",
    }
    if panel_enabled:
        try:
            panel_port = int(os.environ.get("WHEEL_LEG_PANEL_PORT", "8765"))
            panel = WebPanelServer(
                command_getter=get_command_state,
                command_setter=set_panel_command,
                telemetry=telemetry,
                limits={
                    "cmd_x": runtime_limits["cmd_x"],
                    "ang_vel": runtime_limits["ang_vel"],
                    "cmd_height": runtime_limits["cmd_height"],
                },
                running_getter=get_running_state,
                limits_setter=set_panel_limits,
                metadata={
                    "task": getattr(args, "task", "wheel_legged"),
                    "experiment_name": getattr(args, "experiment_name", ""),
                    "checkpoint": getattr(args, "checkpoint", ""),
                },
                port=panel_port,
            )
            print(f"[WEB] control panel: {panel.start()}")
        except (OSError, ValueError, FileNotFoundError) as exc:
            print(f"[WEB] panel disabled: {exc}")

    i = 0
    try:
        while running and i < 100000:
            apply_manual_commands(env, env_cfg)
            if is_sequence_policy:
                actions, _ = policy(obs, obs_history)
            else:
                actions = policy(obs)

            obs, _, _, _, _, obs_history = env.step(actions)
            apply_manual_commands(env, env_cfg)

            if i % 5 == 0:
                telemetry.append(
                    {
                        "step": i,
                        "time": i * float(env.dt),
                        "cmd_x": env.commands[0, 0].item(),
                        "cmd_yaw": env.commands[0, 1].item(),
                        "cmd_height": env.commands[0, 2].item(),
                        "vx": env.base_lin_vel[0, 0].item(),
                        "vy": env.base_lin_vel[0, 1].item(),
                        "real_yaw": env.base_ang_vel[0, 2].item(),
                        "vz": env.root_states[0, 9].item(),
                        "height": env.root_states[0, 2].item(),
                        "wheel_vel_l": env.dof_vel[0, 2].item(),
                        "wheel_vel_r": env.dof_vel[0, 5].item(),
                        "wheel_tau_l": env.torques[0, 2].item(),
                        "wheel_tau_r": env.torques[0, 5].item(),
                    }
                )

            if i % 50 == 0:
                if i == 0:
                    print(f"dof_names={env.dof_names}")
                    print(f"torque_limits={env.torque_limits.detach().cpu().tolist()}")
                vz = env.root_states[0, 9].item()
                yaw_rate = env.base_ang_vel[0, 2].item()
                wheel_action_l = actions[0, 2].item()
                wheel_action_r = actions[0, 5].item()
                wheel_vel_l = env.dof_vel[0, 2].item()
                wheel_vel_r = env.dof_vel[0, 5].item()
                wheel_torque_l = env.torques[0, 2].item()
                wheel_torque_r = env.torques[0, 5].item()
                # left_F = env.vmc_F[0, 0].item()
                # right_F = env.vmc_F[0, 1].item()
                print(
                    f"[{i}] vz={vz:.3f}, cmd_x={env.commands[0, 0].item():.2f}, "
                    f"cmd_yaw={env.commands[0, 1].item():.3f}, real_yaw={yaw_rate:.3f}, "
                    f"wheel_action=({wheel_action_l:.3f},{wheel_action_r:.3f}), "
                    f"wheel_vel=({wheel_vel_l:.3f},{wheel_vel_r:.3f}), "
                    f"wheel_tau=({wheel_torque_l:.3f},{wheel_torque_r:.3f}), "
                    # f"F_left={left_F:.2f}, F_right={right_F:.2f}"
                )
            i += 1
    finally:
        try:
            listener.stop()
        except Exception:
            pass
        if panel is not None:
            panel.stop()


if __name__ == "__main__":
    EXPORT_POLICY = False
    args = get_args()
    play(args)
