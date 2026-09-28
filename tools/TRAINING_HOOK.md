# Training completion hook

Historical stand workflow: `train_stand_long.py` started `training_completion_hook.py` when
`codex` is on PATH. The watcher reads only the small `status.json` every 30 s.
It called `codex exec` once on `finished_pending_review`,
`paused_on_regression`, or `error`. It does not restart training.

That historical mode created an independent Codex review, not a message injected into the
current interactive conversation. Saved CLI authentication is used; the
review consumes the configured Codex account/API quota. Desktop notification
is best-effort and requires a working desktop notification service.

Attach to an already running job:

```bash
python3 tools/training_completion_hook.py \
  --job /home/kellen/fudan_train/plane/outputs/<job>
```

The training supervisor launches the same command detached. A file lock
prevents simultaneous watchers; `completion_hook.json` prevents repeated
reviews. A failed/timeout review is recorded and not retried automatically.
If a supervisor is killed before writing a terminal status, the watcher
continues waiting; this hook is not a process-crash detector.

Artifacts in the job directory:

- `completion_hook.json`: trigger, timestamps, return code and review status.
- `completion_context.json`: compact audit data sent to Codex.
- `codex_review.md`: Chinese review report.
- `codex_hook.log`: diagnostic output for the Codex call.
- `completion_watcher.log`: watcher diagnostics.

The review uses read-only sandbox settings and a prompt restricted to supplied
metrics. No training edits or automatic model promotion are requested.
The call times out after 300 seconds.

Verified on 2026-09-17: a labeled smoke event produced a report; the actual
`stand_long_20260917_175225` regression pause also produced a report with
return code 0. The guard failed because leg0 difference was 0.3114 rad,
above the supervisor threshold 0.30 rad. All checkpoints were retained.

Official interface: https://developers.openai.com/codex/noninteractive
## Detached training contract

Use `tools/submit_turn_training.py` for an authorized turn run. It requires an
existing `OUTPUT_ROOT/budget_ledger.json`, reserves the authorized iterations
under a file lock, then writes `job_spec.json` and `submit_snapshot.json` with
one attempt/event ID. A detached orchestrator owns trainer, monitor and one
completion hook. The submit command returns `running` only after bounded
startup handshake confirms all three processes and records PID start ticks.
It redirects output to job logs. The submitting Codex must not poll TensorBoard, `summaries/`,
`probes/`, or checkpoints while the run is active.

Pass `--hapi-session` only when the current session is the intended recipient.
The orchestrator waits for the trainer exit code, writes `trainer_exit.json`,
settles the budget by attempt ID, and creates `terminal_summary.json`, a
nonempty `completion_report.md` and `report_ready.json`. Only then can the
hook use the persisted event ID and existing lock/wake protocol. The hook
never starts training or an independent `codex exec`. Without an explicit
`--hapi-session` on a new job it creates a local report and does not guess
a recipient. `completion_review.json` remains the independent human review record.
An unprovable update count retains its reservation as `unknown_consumption`.
Do not infer budget exhaustion solely from a target checkpoint; require
trainer return code zero and the actual update count.

`tools/wait_for_completion.py` is an optional human tool, not a worker loop.
`--once --json --quiet --no-report` reads only small marker/status files; timeout
means the wait ended and is not itself a training failure.
