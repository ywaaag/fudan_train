# Standing push validation

Future short training segments default to 500 iterations (user preference,
2026-09-18). The completed 400 -> 600 experiment remains a 200-iteration
historical run. New invocations of continue_stand_validated.py still start
from the validated 400 checkpoint and now target 900; select a newer source
only after its full audit is reviewed.

Compare source model_200 and STAND_SYMMETRIC model_400 before further training.
`tools/audit_stand_push.py` runs three matched seeds (19, 37, 53), 32 envs,
60 seconds each, level-1 parameter randomization, no observation/action noise,
zero commanded vx/yaw and 0.40 m commanded height. Episode timeout is 70 s.

At t=10/20/30/40/50 s add a world-frame horizontal velocity increment of
0.10 m/s, directions +x/-x/+y/-y/+x. This is a velocity impulse test, not
a calibrated applied-force test. Built-in random pushes are disabled.

Recovery requires 0.5 s continuously with tilt <0.15 rad, height error <0.03 m,
horizontal speed <0.20 m/s, both wheel contacts, no nonwheel contact, and no
reset since that push. A reset invalidates recovery for that push. Remaining
unrecovered events are recorded as null. Geometry metrics include transients.
Small balancing motion is allowed, but runaway motion is not.

Reports: `plane/outputs/stand_push_comparison_v1/`. `status.json` stores only
compact progress/results. Training is not launched automatically by this audit.

Matching GUI (standard fudan_leg setup, cwd=plane; exit previous viewer first):

```bash
python wheel_legged_gym/scripts/evaluate_standing.py \
 --gui --profile=STAND_SYMMETRIC --num-envs=1 --seed=19 \
 --height=0.40 --seconds=60 --warmup=5 --randomization-level=1 \
 --push-delta-v=0.10 \
 --checkpoint=logs/wheel_legged/Sep17_22-23-26_stand_ablation_20260917_221553_stand_symmetric_probe/model_400.pt \
 --out=outputs/candidate_push_gui.json
```

For a matched source comparison, keep every option and replace checkpoint by
`logs/wheel_legged/Sep17_17-41-11_stand_rand1_entropyfix_probe_sep17/model_200.pt`
and output by `outputs/source_push_gui.json`. Camera setup is identical.
GUI code has been syntax-checked; headless impulse smoke was physically run.
Desktop rendering has not been verified in this task.

Passing this audit supports only these parameter ranges and small impulses;
it does not demonstrate hardware safety or robustness to stronger pushes.

## Completed comparison and next bounded segment

Each model: 96 randomized environments, 480 impulses, no failure or nonwheel
contact, all impulse trials met recovery conditions.

| Metric | Source 200 | Candidate 400 |
|---|---:|---:|
| Longest recovery including 0.5 s confirmation | 0.91 s | 0.76 s |
| Mean knee mirror error | 20.65 mm | 18.98 mm |
| Mean wheel mirror error | 35.72 mm | 33.00 mm |

Candidate passed the small-push gate. `tools/continue_stand_validated.py`
adds only 200 iterations (400 -> 600), with unchanged STAND_SYMMETRIC reward,
level-1 training randomization, full resume and frozen encoder. It then runs
the same 60-second three-seed impulse audits and stops for review. It does
not inject new training pushes or claim stronger-disturbance robustness.
Outputs go to a new `plane/outputs/stand_validated_<timestamp>` directory.
