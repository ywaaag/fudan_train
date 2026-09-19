# Low-speed motion relay

Standing baseline committed before this experiment: 696f487. Source checkpoint:
`Sep18_21-21-37_stand_validated_20260918_212129/model_3100.pt`.
Final target is +/-4 m/s translation, +/-4 rad/s spin and smooth turning;
the first stage only attempts +/-0.5 m/s. No position-hold objective is added.

LOW_SPEED is a separate method_v1 translate level **0** profile. The previous
statement that level 1 meant 0.5 m/s was incorrect: levels are zero-indexed
(0.5, 1, 2, 3, 4). Sampling: 20% exact zero, 20% +/-0.1 anchors, 30% reverse
endpoint, 30% forward endpoint. Yaw remains zero; height is 0.40 m.

Level-1 parameter randomization, no pushes in training. Fixed actor LR 1e-5,
encoder LR 1e-5, entropy 0.001. Preserve checkpoint actor/encoder/critic/std
and Adam moments via explicit full-state warm start. Reward semantics change,
so critic transfer is an initialization, not an exact continuation. The source
manifest must be normalized_v1 STAND_SYMMETRIC or LOW_SPEED with level-1
randomization; record source reward scales for review. Stale critic adaptation
is a risk to inspect during the short probe.

No geometric stand penalty or stand_still penalty in locomotion. Existing
mirror-equivariance regularization remains 0.01, allowing differential actions.
Zero-speed rewards remain to retain stopping, but no world-position target.

Standard fudan_leg environment, cwd=plane:

```bash
python wheel_legged_gym/scripts/train.py \
 --task=wheel_legged --headless --num_envs=4096 \
 --resume --resume_mode=full \
 --load_run=Sep18_21-21-37_stand_validated_20260918_212129 --checkpoint=3100 \
 --policy_experiment=LOW_SPEED --max_iterations=500 --seed=23 \
 --run_name=low_speed_05_new
```

Automated one-stage run from repository root:

```bash
/home/kellen/anaconda3/envs/fudan_leg/bin/python tools/run_low_speed.py
```

The script trains to model_3600.pt, then runs 9 fixed-command deterministic
audits: seeds 19/37/53, commands -0.5/0/+0.5, 16 envs, 25 seconds, 5 warmup.
Audits assert that actual commands were not overwritten. Require no failures,
no nonwheel contacts, per-wheel contact >=99%, height within 3 cm, vx MAE <=0.1
(zero <=0.05), yaw MAE <=0.1. It stops for review; no automatic speed increase.
These tests are not yet smooth-transition or high-speed turning acceptance.
Logs and status are under plane/outputs/low_speed_05_<timestamp>.

Validation: config/sampler test, 64-env one-iteration resume smoke, fixed-command
evaluation smoke. Training and audit outcomes must be inspected separately.

## Response ablation: LOW_SPEED_TRACKING

First 500-iteration LOW_SPEED run failed all directional tracking gates.
Additional seed19 diagnostics (8 envs, 15 s / 5 warmup) measured encoder vx
MAE 0.0155 m/s in reverse and 0.0196 m/s forward. These errors are much
smaller than task tracking errors; they do not establish encoder error as
the dominant cause. Actual commands are asserted each evaluation step.

LOW_SPEED_TRACKING changes only tracking_linear_cap from 1.0 to 0.5 m/s,
matching the first stage's velocity endpoints. This changes coarse/fine/gap
normalization together, sharpening the reward distinction between stationary
and correctly moving behavior. It does not compensate or offset commands.
Sampling, optimizer settings and physical parameters remain the same.
It starts again from the same model_3100 with seed23 for a matched 500-iteration
comparison, preserving the original failed run for reference.

```bash
/home/kellen/anaconda3/envs/fudan_leg/bin/python tools/run_low_speed.py \
 --profile LOW_SPEED_TRACKING
```

Two config/reward tests and a one-iteration 64-env resume smoke passed.
The new reward normalization is recorded in manifest reward_parameters.
Nine independent audits run automatically after training, with unchanged gates.
No stage advancement or performance claim until these tests finish.
