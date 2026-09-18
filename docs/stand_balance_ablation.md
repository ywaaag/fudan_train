# Stable, approximately symmetric standing (balancing motion permitted)

The target now prioritizes survival/contact, upright posture and bilateral
geometry. Small balancing motion is permitted; no exact-zero-speed claim
is used to select a candidate. The earlier 3 m/s goal is not this experiment.

Two named profiles share the same stable model_200 source:
`Sep17_17-41-11_stand_rand1_entropyfix_probe_sep17`.

- STAND_CONTROL reconstructs its original stand reward including stand_still=-0.2.
- STAND_SYMMETRIC differs only by stand_bilateral_geometry=-0.1.
- Full model, critic, std, Adam states and iteration are loaded. A manifest
  check allows only the geometry term to differ from the source reward.
- Both use fixed actor/critic LR=1e-5; encoder LR=0 freezes the estimator.
- Both use level-1 domain randomization and identical seed 23.
- No geometry projection, raw action equality or asset/PD changes.
- Profile overrides are explicit; old method_v1/FUDAN_STAND are not replaced.

Smoke verified exact encoder equality, std mean 0.18403 -> 0.18393 and
Adam step 4000 -> 4020. Tests verify only the geometry reward differs.

Run the matched experiment from repo root with fudan_leg Python:

```bash
/home/kellen/anaconda3/envs/fudan_leg/bin/python tools/run_stand_ablation.py
```

The script runs 1-iteration smoke and 200-iteration training per branch,
then independent 32-env audits with seeds 19 and 37 (25 seconds, 5 warmup).
Outputs are redirected to `plane/outputs/stand_ablation_<timestamp>/`.
It does not auto-promote a model or launch a long run. Read status.json and
the audit JSON, not full training stdout. Each probe finishes at model_400.pt.

Direct treatment command (cwd=plane, standard fudan_leg environment):

```bash
python wheel_legged_gym/scripts/train.py \
  --task=wheel_legged --headless --num_envs=4096 \
  --resume --resume_mode=full \
  --load_run=Sep17_17-41-11_stand_rand1_entropyfix_probe_sep17 --checkpoint=200 \
  --policy_experiment=STAND_SYMMETRIC --max_iterations=200 --seed=23 \
  --run_name=stand_symmetric_probe_new
```

Compare actual root-frame knee/wheel mirror distances, roll/pitch bias and
fluctuation, nonwheel contact, per-wheel contact and resets. Joint differences
and mean reward are auxiliary. No experiment here establishes real-world
robustness or push recovery without additional tests.

## Completed matched experiment

Job: `plane/outputs/stand_ablation_20260917_221553`.
The supervisor was interrupted while the control trainer continued. Recovery
attached to that trainer, skipped its completed smoke, and did not restart it.
`run_stand_ablation.py --job <existing-job>` now supports this recovery and
uses a supervisor lock to reject duplicate schedulers.

Both branches completed 200 added iterations (200 -> 400), then audits with
32 randomization-level-1 environments each at seeds 19 and 37.

| Metric | Control | Geometry treatment |
|---|---:|---:|
| Failure resets across both audits | 0 | 0 |
| Wheel contact L/R | 1 / 1 | 1 / 1 |
| Nonwheel contact fraction | 0 | 0 |
| Mean knee mirror error | 33.21 mm | 19.98 mm |
| Mean wheel mirror error | 57.75 mm | 34.82 mm |
| Mean absolute roll | 0.102 deg | 0.091 deg |
| Mean absolute pitch | 1.573 deg | 1.250 deg |
| Mean absolute vx across seeds | 0.0309--0.0333 m/s | 0.0351--0.0380 m/s |

Treatment checkpoint:
`Sep17_22-23-26_stand_ablation_20260917_221553_stand_symmetric_probe/model_400.pt`.
This is about 40% lower geometry error than matched continuation without the
geometry reward. It is not perfect symmetry, nor evidence for real-world
robustness or stronger unseen disturbances. Compare with the source audit
separately; continuation-control improvements are not automatically gains
over the original source model.

Source checkpoint audited with identical seed 19/settings: knee 21.04 mm,
wheel 36.62 mm, height 0.4141 m, no failures or nonwheel contact. Treatment
seed19 is 19.76/34.50 mm (about 6% lower), height 0.4095 m. Its pitch bias
is larger than the source though still small. Thus the strongest finding is
prevention of continuation-induced asymmetry, not a dramatic source-model
improvement. Keep the source available for visual and subsequent comparisons.

18 profile/mirror/geometry regression tests passed. Exact implementation files:
`envs/wheel_legged/stand_balance.py`, `envs/wheel_legged/policy_experiments.py`,
`scripts/train.py`, `scripts/evaluate_standing.py` (all under plane/wheel_legged_gym),
`plane/tests/test_stand_balance.py`, and `tools/run_stand_ablation.py`.
