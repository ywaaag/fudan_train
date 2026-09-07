# Wheel-leg Web Control Panel

`wheel_legged_gym/scripts/play.py` now starts a local control panel automatically.
After launching the usual play command, open:

```text
http://127.0.0.1:8765/
```

The panel provides:

- sliders for `cmd_x`, `ang_vel` and `cmd_height`;
- browser WSAD control (hold a key, release to return that command to zero);
- an optional browser Gamepad API bridge: left-stick Y controls forward/backward,
  right-stick X controls yaw, and D-pad up/down controls height;
- live Canvas curves for commanded/real velocity, yaw, height, vertical speed,
  wheel velocity and wheel torque;
- a CSV download of the telemetry currently retained by the play process.

The control-range editor in the panel changes the runtime clamp immediately.
This is why the target-height control is now usable: the old task config had
`height = [0.40, 0.40]`, so every incoming height command was clipped back to
`0.40`.

The panel uses only Python's standard library on the simulator side. The
existing `pynput` keyboard controls remain available. Commands are clamped to
the active task ranges before they are written to `env.commands`.

Environment variables:

```bash
WHEEL_LEG_PANEL=0             # disable the panel
WHEEL_LEG_PANEL_PORT=8765     # change the local port
WHEEL_LEG_PANEL_X_MIN=-2.0
WHEEL_LEG_PANEL_X_MAX=2.0
WHEEL_LEG_PANEL_YAW_MIN=-2.0
WHEEL_LEG_PANEL_YAW_MAX=2.0
WHEEL_LEG_PANEL_HEIGHT_MIN=0.10
WHEEL_LEG_PANEL_HEIGHT_MAX=1.00
```

These six range variables are optional. The same values can be edited after
startup in the panel, without changing the training configuration files. The
checkpoint was trained around a 0.40 m target height, so values far outside
that range may make the robot unstable.

The panel binds to `127.0.0.1` only. Stop the Isaac Gym play process with the
existing `q`/`ESC` keyboard controls; the HTTP server shuts down with it.
