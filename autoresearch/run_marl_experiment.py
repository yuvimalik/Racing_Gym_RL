"""
Run a single multi-agent autoresearch experiment against the maintained stack.

This wraps train.py and evaluate.py so the recursive search loop can treat each
candidate as a single subprocess that returns one JSON payload.
"""

from __future__ import annotations

import argparse
import json
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds >= 3600:
        return f"{seconds / 3600:.1f}h"
    if seconds >= 60:
        return f"{seconds / 60:.1f}m"
    return f"{seconds:.0f}s"


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_yaml(path: Path, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _stream_reader(stream, stream_name: str, output_queue: queue.Queue, sink) -> None:
    try:
        while True:
            try:
                line = stream.readline()
            except (ValueError, OSError):
                break
            if line == "":
                break
            output_queue.put((stream_name, line))
            sink.write(line)
            sink.flush()
    finally:
        try:
            stream.close()
        except Exception:
            pass


def run_command(cmd: list[str], *, cwd: Path, stdout_path: Path, stderr_path: Path, timeout: int) -> tuple[int, bool]:
    with open(stdout_path, "w", encoding="utf-8") as stdout_sink, open(
        stderr_path, "w", encoding="utf-8"
    ) as stderr_sink:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        output_queue: queue.Queue = queue.Queue()
        threads = [
            threading.Thread(
                target=_stream_reader,
                args=(proc.stdout, "stdout", output_queue, stdout_sink),
                daemon=True,
            ),
            threading.Thread(
                target=_stream_reader,
                args=(proc.stderr, "stderr", output_queue, stderr_sink),
                daemon=True,
            ),
        ]
        for thread in threads:
            thread.start()

        timed_out = False
        deadline = time.time() + timeout
        while True:
            if proc.poll() is not None and output_queue.empty():
                break
            remaining = max(0.0, deadline - time.time())
            if remaining == 0.0 and proc.poll() is None:
                timed_out = True
                proc.kill()
            try:
                stream_name, line = output_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            prefix = "[child-stdout]" if stream_name == "stdout" else "[child-stderr]"
            sys.stderr.write(f"  {prefix} {line}")
            sys.stderr.flush()

        for thread in threads:
            thread.join(timeout=1)
        return proc.wait(), timed_out


def ensure_stage_config(base_config: dict, checkpoint_dir: Path, timesteps: int) -> tuple[dict, Path]:
    cfg = json.loads(json.dumps(base_config))
    training = cfg.setdefault("training", {})
    paths = cfg.setdefault("paths", {})
    training["trainer_backend"] = "torch"
    training["total_timesteps"] = int(timesteps)
    if "eval_freq" in training:
        training["eval_freq"] = max(1, min(int(training["eval_freq"]), int(timesteps)))
    if "save_freq" in training:
        training["save_freq"] = max(1, min(int(training["save_freq"]), int(timesteps)))
    visual_eval = training.setdefault("visual_eval", {})
    visual_eval["enabled"] = False

    paths["model_dir"] = str((checkpoint_dir / "models").resolve())
    paths["log_dir"] = str((checkpoint_dir / "logs").resolve())
    paths["results_dir"] = str((checkpoint_dir / "results").resolve())

    effective_config_path = checkpoint_dir / "effective_config.yaml"
    write_yaml(effective_config_path, cfg)
    return cfg, effective_config_path


def locate_latest_run_summary(results_root: Path) -> Optional[Path]:
    summaries = sorted(
        results_root.glob("**/run_summary.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return summaries[0] if summaries else None


def locate_checkpoint(model_dir: Path) -> Optional[Path]:
    best_path = model_dir / "best_model_torch.pt"
    final_path = model_dir / "final_model_torch.pt"
    if best_path.exists():
        return best_path
    if final_path.exists():
        return final_path
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one multi-agent autoresearch experiment")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--timesteps", type=int, required=True)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--resume-mode", type=str, default="policy_only", choices=["full", "policy_only"])
    parser.add_argument("--experiment-id", type=str, default=None)
    parser.add_argument("--timeout", type=int, default=7200)
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_stdout = checkpoint_dir / "train_stdout.log"
    train_stderr = checkpoint_dir / "train_stderr.log"
    eval_stdout = checkpoint_dir / "eval_stdout.log"
    eval_stderr = checkpoint_dir / "eval_stderr.log"
    metrics_path = checkpoint_dir / "metrics.json"

    base_config_path = Path(args.config).resolve()
    base_config = load_yaml(base_config_path)
    _, effective_config_path = ensure_stage_config(base_config, checkpoint_dir, args.timesteps)
    model_dir = checkpoint_dir / "models"
    results_root = checkpoint_dir / "results"

    train_cmd = [
        sys.executable,
        "train.py",
        "--config",
        str(effective_config_path),
        "--seed",
        str(args.seed),
        "--trainer_backend",
        "torch",
        "--resume_mode",
        str(args.resume_mode),
    ]
    if args.resume:
        train_cmd.extend(["--resume", str(Path(args.resume).resolve()), "--timesteps_add", str(args.timesteps)])

    log(f"[marl_experiment] Base config: {base_config_path}")
    log(f"[marl_experiment] Effective config: {effective_config_path}")
    log(f"[marl_experiment] Timesteps: {args.timesteps:,} | eval_episodes={args.eval_episodes}")
    log(f"[marl_experiment] Resume: {args.resume if args.resume else 'none'}")
    log(f"[marl_experiment] Running train.py ...")

    wall_t0 = time.time()
    train_t0 = time.time()
    train_return_code, train_timed_out = run_command(
        train_cmd,
        cwd=PROJECT_ROOT,
        stdout_path=train_stdout,
        stderr_path=train_stderr,
        timeout=args.timeout,
    )
    train_seconds = time.time() - train_t0

    if train_timed_out:
        payload = {
            "mean_reward": -999.0,
            "error": f"Training timeout after {args.timeout}s",
            "train_return_code": train_return_code,
            "train_wall_clock_seconds": train_seconds,
            "effective_config_path": str(effective_config_path),
        }
        metrics_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(payload))
        return

    checkpoint_path = locate_checkpoint(model_dir)
    run_summary_path = locate_latest_run_summary(results_root)
    if train_return_code != 0 or checkpoint_path is None:
        payload = {
            "mean_reward": -999.0,
            "error": "train.py failed before producing a checkpoint",
            "train_return_code": train_return_code,
            "train_wall_clock_seconds": train_seconds,
            "effective_config_path": str(effective_config_path),
            "run_summary_path": str(run_summary_path) if run_summary_path else None,
        }
        metrics_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(payload))
        return

    eval_stats_path = checkpoint_dir / "evaluation_stats.json"
    eval_cmd = [
        sys.executable,
        "evaluate.py",
        "--model",
        str(checkpoint_path),
        "--config",
        str(effective_config_path),
        "--episodes",
        str(args.eval_episodes),
        "--no-video",
        "--output-json",
        str(eval_stats_path),
        "--seed",
        str(args.seed + 10000),
    ]

    log(f"[marl_experiment] Training done in {format_duration(train_seconds)}")
    log(f"[marl_experiment] Evaluating checkpoint: {checkpoint_path}")

    eval_t0 = time.time()
    eval_return_code, eval_timed_out = run_command(
        eval_cmd,
        cwd=PROJECT_ROOT,
        stdout_path=eval_stdout,
        stderr_path=eval_stderr,
        timeout=max(1800, args.timeout // 2),
    )
    eval_seconds = time.time() - eval_t0

    if eval_timed_out or eval_return_code != 0 or not eval_stats_path.exists():
        payload = {
            "mean_reward": -999.0,
            "error": "evaluate.py failed after training",
            "train_return_code": train_return_code,
            "eval_return_code": eval_return_code,
            "train_wall_clock_seconds": train_seconds,
            "eval_wall_clock_seconds": eval_seconds,
            "effective_config_path": str(effective_config_path),
            "checkpoint_path": str(checkpoint_path),
            "run_summary_path": str(run_summary_path) if run_summary_path else None,
        }
        metrics_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(payload))
        return

    stats = json.loads(eval_stats_path.read_text(encoding="utf-8"))
    payload = {
        **stats,
        "train_return_code": train_return_code,
        "eval_return_code": eval_return_code,
        "train_wall_clock_seconds": train_seconds,
        "eval_wall_clock_seconds": eval_seconds,
        "total_wall_clock_seconds": time.time() - wall_t0,
        "timesteps_requested": int(args.timesteps),
        "effective_config_path": str(effective_config_path),
        "base_config_path": str(base_config_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_dir": str(checkpoint_dir),
        "model_dir": str(model_dir),
        "results_dir": str(results_root),
        "run_summary_path": str(run_summary_path) if run_summary_path else None,
        "train_stdout_path": str(train_stdout),
        "train_stderr_path": str(train_stderr),
        "eval_stdout_path": str(eval_stdout),
        "eval_stderr_path": str(eval_stderr),
    }
    if args.experiment_id is not None:
        payload["experiment_id"] = args.experiment_id
    metrics_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
