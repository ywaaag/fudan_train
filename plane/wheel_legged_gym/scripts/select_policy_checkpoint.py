"""Rank checkpoints from one run using deterministic Isaac command grids."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


SCRIPT = Path(__file__).with_name("isaac_command_grid.py")


def checkpoint_iteration(path: Path) -> int:
    return int(path.stem.rsplit("_", 1)[-1])


def score_grid(payload: dict) -> dict:
    normalized_errors = []
    linear_errors = []
    yaw_errors = []
    terminations = 0
    for item in payload["results"]:
        forward, yaw, _ = item["command"]
        if forward == 0.0 and yaw == 0.0:
            error = abs(item["mean_vx_m_s"])
            tolerance = 0.03
            linear_errors.append(error)
        elif forward != 0.0:
            error = abs(item["mean_vx_m_s"] - forward)
            tolerance = 0.05
            linear_errors.append(error)
        else:
            error = abs(item["mean_yaw_rate_rad_s"] - yaw)
            tolerance = 0.05
            yaw_errors.append(error)
        normalized_errors.append(error / tolerance)
        terminations += int(item["termination_count"])
    return {
        "passed": bool(payload["passed"]),
        "passed_commands": sum(bool(item["passed"]) for item in payload["results"]),
        "max_normalized_error": max(normalized_errors),
        "mean_normalized_error": sum(normalized_errors) / len(normalized_errors),
        "mean_linear_error_m_s": sum(linear_errors) / len(linear_errors),
        "mean_yaw_error_rad_s": sum(yaw_errors) / max(len(yaw_errors), 1),
        "termination_count": terminations,
        "zero_mean_vx_m_s": payload["results"][0]["mean_vx_m_s"],
        "forward_005_mean_vx_m_s": payload["results"][1]["mean_vx_m_s"],
        "reverse_005_mean_vx_m_s": payload["results"][2]["mean_vx_m_s"],
    }


def evaluate(
    checkpoint: Path,
    report_dir: Path,
    *,
    seconds: float,
    warmup_seconds: float,
    seed: int,
    device_id: int,
) -> dict:
    output = report_dir / f"{checkpoint.stem}_isaac_grid.json"
    process = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--checkpoint",
            str(checkpoint),
            "--seconds",
            str(seconds),
            "--warmup-seconds",
            str(warmup_seconds),
            "--seed",
            str(seed),
            "--device-id",
            str(device_id),
            "--out",
            str(output),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if not output.is_file():
        raise RuntimeError(
            f"grid evaluator failed for {checkpoint}: returncode={process.returncode}\n"
            + process.stderr[-1500:]
        )
    payload = json.loads(output.read_text(encoding="utf-8"))
    return {
        "checkpoint": str(checkpoint.resolve()),
        "iteration": checkpoint_iteration(checkpoint),
        "grid_report": str(output.resolve()),
        **score_grid(payload),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int)
    parser.add_argument("--step", type=int, default=100)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--warmup-seconds", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.step <= 0:
        parser.error("--step must be positive")
    checkpoints = sorted(args.run_dir.glob("model_*.pt"), key=checkpoint_iteration)
    checkpoints = [
        path
        for path in checkpoints
        if checkpoint_iteration(path) >= args.start
        and (args.stop is None or checkpoint_iteration(path) <= args.stop)
        and checkpoint_iteration(path) % args.step == 0
    ]
    if not checkpoints:
        parser.error("no checkpoints matched the requested range")
    args.report_dir.mkdir(parents=True, exist_ok=True)
    ranking = [
        evaluate(
            checkpoint,
            args.report_dir,
            seconds=args.seconds,
            warmup_seconds=args.warmup_seconds,
            seed=args.seed,
            device_id=args.device_id,
        )
        for checkpoint in checkpoints
    ]
    ranking.sort(
        key=lambda item: (
            item["termination_count"] > 0,
            -item["passed_commands"],
            item["max_normalized_error"],
            item["mean_normalized_error"],
        )
    )
    payload = {
        "run_dir": str(args.run_dir.resolve()),
        "seconds": args.seconds,
        "warmup_seconds": args.warmup_seconds,
        "seed": args.seed,
        "best_checkpoint": ranking[0]["checkpoint"],
        "any_passed": any(item["passed"] for item in ranking),
        "ranking": ranking,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("RESULT " + json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
