# Stand geometry reward

The original Fudan `_reward_nominal_state` compares virtual leg angles
`theta0`. This implementation uses the same idea of bilateral posture
regularization, but measures actual tree-URDF landmarks rather than
reusing the reference robot's kinematic equations.

Only method_v1/stand enables `stand_bilateral_geometry=-0.3`.
For left/right knee joint origins (`leg_1_link`) and wheel centers:

1. Subtract root position, then inverse-rotate by root quaternion.
2. Reflect the right point across body y=0: `[x,-y,z]`.
3. Compute each left/right distance; allow 0.005 m tolerance.
4. Apply smooth-L1 to excess distance normalized by 0.05 m, then average
   knee/wheel costs. No early clipping that hides large geometric errors.

Reward is negative cost, scaled by policy dt once. It applies only to zero
vx/yaw commands. It does not project actions or force identical torques.
The model's small COM bias and differential balancing remain possible.

The old stand_still position penalty is disabled in this phase: it included
wheel rotation angles and clipped the combined penalty, neither of which
directly measures bilateral posture. Zero wheel velocity, body velocity,
height, orientation, wheel contact, slip and failure terms remain active.
Randomization level 1 and policy equivariance coefficient 0.01 are retained.

Changing reward invalidates full optimizer/critic continuation assumptions:
the first probe uses policy-only resume from the earlier `model_200.pt`.
This resets critic/optimizers and exploration std, so compare deterministic
audits, not early stochastic rewards alone.

Validation: 13 geometry/profile tests and one-iteration Isaac Gym smoke.
Probe: `stand_geometry_probe_v1`, seed 23, 4096 environments, 200 iterations.
Source run: `Sep17_17-41-11_stand_rand1_entropyfix_probe_sep17` / 200.
Training stdout: `/tmp/stand_geometry_probe.log`; TensorBoard is stored in
the usual timestamped run directory. Evaluate before promoting to long training.

Run with the standard fudan_leg environment:

```bash
export FUDAN_STAND_RANDOMIZATION_LEVEL=1
python wheel_legged_gym/scripts/train.py \
  --task=wheel_legged --headless --num_envs=4096 \
  --resume --resume_mode=policy \
  --load_run=Sep17_17-41-11_stand_rand1_entropyfix_probe_sep17 --checkpoint=200 \
  --policy_experiment=method_v1 --phase=stand --command_level=0 \
  --max_iterations=200 --seed=23 --run_name=stand_geometry_probe_v2
```

All reference files are read-only; no URDF/PD/observation/action contract changes.

First probe outcome: 25-second deterministic level-1 audit (32 envs, seed 19)
gave knee/wheel mirror errors 8.49/11.58 mm, but 4 failure resets affecting
3 environments. Mean absolute vx was 0.0963 m/s. It is not promoted.

Second probe keeps the same rewards and changes only standing initialization
noise std from 0.5 to 0.15. The manifest now records init_noise_std.
Run name: stand_geometry_low_noise_probe_v2; same source checkpoint and seed,
200 iterations. One-iteration smoke passed before starting the probe.

Second probe failed (83 failure resets, 28.1% survival in the same audit).
Lower initialization noise alone did not resolve migration instability.
Third probe starts again from the original stable model_200, retaining the
second probe's noise setting but switching the stand optimizer to fixed
1e-5 actor learning rate. Encoder LR remains 1e-5. The optimizer manifest
now uses the actual train_cfg values. Other phases keep adaptive 1e-4.
Run name: stand_geometry_fixedlr_probe_v3, 200 iterations, seed 23.
This is a hypothesis-driven short probe, not a validated robustness result.

Final disposition: third probe also failed (542 resets, 0% survival).
Both initialization-noise and optimizer experiments were reverted to the
pre-existing 0.5 / adaptive 1e-4 settings. The geometry reward implementation
is retained as a candidate, NOT a validated long-training configuration.
No new model replaces the original stable checkpoint. Further work must
isolate policy/encoder/critic migration effects before additional long runs.
Reports: plane/outputs/stand_geometry_probe_rand1_seed19.json,
stand_geometry_v2_seed19.json and stand_geometry_v3_seed19.json.
