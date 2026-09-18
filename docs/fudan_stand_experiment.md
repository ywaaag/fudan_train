# FUDAN_STAND: adapted reference reproduction

Primary read-only reference:
`/home/kellen/wheel_leg_mjrl-lqr/fudan_train/plane/wheel_legged_gym`.
Original `/home/kellen/fudan_rl_wheel_leg/plane/wheel_legged_gym` was also compared.

This is an independent profile, not a replacement of method_v1.
Reward formulas use the legacy implementations, including one dt scaling
and per-term clipping to +/-dt. PPO: actor LR=1e-3 adaptive, encoder LR=1e-3,
entropy=0.01, 5 epochs, 4 mini-batches, 48 rollout steps; symmetry loss=0.
No method_v1 geometry/Huber/contact rewards are mixed into this baseline.

Adapted reference scales:

| Term | Scale |
|---|---:|
| tracking_lin_vel / tracking_ang_vel | 1 / 1 |
| base_height | 2 |
| nominal_state | -1 |
| orientation | -500 |
| lin_vel_z / ang_vel_xy | -1 / -0.2 |
| dof_vel / dof_acc | -0.01 / -2.5e-7 |
| torques | -0.0001 |
| action_rate / action_smooth | -0.01 / -0.01 |
| collision / dof_pos_limits | -1 / -1 |
| zero_base_velocity / zero_wheel_velocity | -1 / -1 |

Explicit adaptations and limits:

- Commands fixed to vx=0, yaw=0, height=0.40 m, flat ground, no curriculum.
- Current robot's URDF, nominal reset, PD, dt, 25/125/6 contract retained.
- Virtual leg angles measured from actual hip-to-wheel vectors in root frame.
  Reference's hard-coded right-joint sign inversion is NOT reused.
- First baseline disables domain randomization; observation noise remains
  enabled for training as in the reference. Robustness training comes later.
- Reference empty contact penalty/termination lists retained for reproduction.
  Evaluation independently records nonwheel contact; survival alone is not acceptance.
- Reference nominal_state aligns virtual leg axes; it does not guarantee
  equal knee bend or full bilateral geometry. Audit both knee and wheel points.
- From-scratch experiment. Resume requests are explicitly rejected for now
  to avoid accidentally loading incompatible method_v1 critic/optimizers.
- Legacy training wheel-contact diagnostic is not method_v1's contact metric;
  use evaluate_standing.py's measured contact fractions for acceptance.

Commands (after standard fudan_leg environment setup; cwd=plane):

```bash
python wheel_legged_gym/scripts/train.py \
  --task=wheel_legged --headless --num_envs=64 \
  --policy_experiment=FUDAN_STAND --max_iterations=2 --seed=31 \
  --run_name=fudan_stand_reference_smoke_v2

python wheel_legged_gym/scripts/train.py \
  --task=wheel_legged --headless --num_envs=4096 \
  --policy_experiment=FUDAN_STAND --max_iterations=200 --seed=31 \
  --run_name=fudan_stand_reference_probe_v2 > outputs/fudan_stand_probe_v2.log 2>&1

python wheel_legged_gym/scripts/evaluate_standing.py \
  --profile=FUDAN_STAND --checkpoint=logs/wheel_legged/<run>/model_200.pt \
  --num-envs=32 --randomization-level=0 --out=outputs/fudan_stand_audit.json
```

Reference SHA256, for provenance:
- wheel_legged_config.py: e1c8a3e91f22fd9047a44246243e44de4c47677f0b30bff84118791015ebe543
- legged_robot.py: 9cea66e8f142847f1867079104438acd157a438d31ba756d9ab2d2f516f35186

## Executed result

13 profile/kinematic tests passed; 2-iteration smoke passed. Fresh 200-iteration
run: `Sep17_21-53-07_fudan_stand_reference_probe_v1/model_200.pt`.
Independent deterministic audit: 32 environments, seed 19, no randomization,
25 seconds total / 5 seconds warmup.

- Mean height: 0.0800 m (target 0.40 m).
- Nonwheel contact fraction: 1.0.
- Both wheel contact fractions: 1.0; failure count 0.
- Mean absolute vx: 0.0000417 m/s; yaw: 0.002169 rad/s.
- Knee/wheel mirror errors: 0.0094 / 0.0445 mm.

This is a symmetric collapsed posture, NOT standing. The exact reference's
empty contact lists and narrow height exponential permit a low-motion ground
resting solution. Reward/episode-length/zero-speed alone would falsely accept
it. No long training or promotion was performed. Keep this reproduction as
a baseline; a safety/height-recovery ablation must be a separately named run.
Audit JSON: `plane/outputs/fudan_stand_reference_probe_v1_audit.json`.
