---
name: codex-tool-discovery
description: "Use the Fudan training repository's existing summaries, checkpoint monitors, evaluation aggregators, ONNX tools, completion hook, and architecture audit before inventing scripts or opening large logs."
---

# Codex Tool Discovery

Apply this skill when a task asks to inspect training progress, choose a
checkpoint, review turn performance, wait for a job, export/verify ONNX,
render a trace, or check architecture.

## Required First Step

Read [`docs/CODEX_TOOL_INDEX.md`](../../../docs/CODEX_TOOL_INDEX.md). It is the
single task-to-tool map. Then read the target run's `status.json`, manifest and
latest summary. Do not scan all event files or stdout before the summary says
which field is missing.
For authorized training/evaluation, follow [`docs/CODEX_WORKFLOW.md`](../../../docs/CODEX_WORKFLOW.md)
for smoke, monitoring, checkpoint selection and completion order.

## Default Read-Only Paths

- Training scalars: `tools/summarize_training.py RUN --window 50`.
- Checkpoint-aligned scalars: add `--through-iteration CHECKPOINT_ITERATION`.
- Existing job state: `jq` the job `status.json`; use `tools/wait_for_completion.py JOB --timeout N` to wait.
- Turn evidence: `tools/summarize_turn_envelope.py --long --job REVIEW_DIR`.
- Architecture: `python3 tools/check_architecture.py`.

These commands do not start training. A summary is evidence navigation, not an
acceptance result; retain missing, failed and unmeasured points.

## Side-Effect Boundary

`--help`, `summarize_training`, `check_architecture` without `--write`, and
`wait_for_completion` are discovery-safe. ONNX export, candidate review,
trace rendering, turn aggregation, and completion `--report-only` write files;
use a new output path and confirm it does not exist. Candidate review and the
turn monitor start Isaac or poll a child process; run them only inside the
user-authorized experiment budget. Never run historical `tools/run_*`,
`continue_*`, or supervisor entrypoints during architecture discovery.

## Checkpoint Selection

Do not select by reward alone. For existing motion workflows, reuse
`wheel_legged_gym.workflows.candidate_screening.screen_candidates` semantics:
single-seed screen first, then at most a small set with seeds 19/37/53; require
safety and posture retention before comparing scores. For `TURN_LEAN_LONG`, use
`tools/review_turn_lean_long.py`, `tools/summarize_turn_envelope.py --long`
and `tools/screen_turn_lean_candidates.py` on complete same-protocol reviews;
compare original gate, lean-aware gate, all seeds, worst environment, COM,
height target, roll target error, contact, reset reason, slip and power.

Never interpolate unmeasured command points. Keep the source checkpoint SHA and
evaluation JSON next to every selected candidate.

## Completion Hook

After a supervisor reaches a terminal status, read `completion_report.md` and
`completion_hook.json`. Do not guess an HAPI session. Do not acknowledge a
review you did not perform. The hook reports and deduplicates; it never proves
training success or authorizes another run.

## Failure Routing

Read the smallest relevant artifact first: failed `--help` means inspect the
CLI wrapper; missing summary tags means use `summarize_training` with the
correct run; checkpoint mismatch means stop candidate comparison and inspect
the manifest/SHA; evaluator failure means inspect its operation log and raw
JSON; architecture violations mean inspect the reported source/target line.
Only then open the corresponding source module or full stdout.
