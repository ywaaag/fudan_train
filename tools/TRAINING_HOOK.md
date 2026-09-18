# Training completion hook

`train_stand_long.py` starts `training_completion_hook.py` automatically when
`codex` is on PATH. The watcher reads only the small `status.json` every 30 s.
It calls `codex exec` once on `finished_pending_review`,
`paused_on_regression`, or `error`. It does not restart training.

This creates an independent Codex review, not a message injected into the
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
