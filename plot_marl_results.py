#!/usr/bin/env python3
"""Plot saved MARL training and evaluation artifacts for a single run."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_jsonl(path: Path):
    if not path.is_file():
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def find_latest_run(results_root: Path) -> Path:
    run_dirs = sorted(
        [p for p in results_root.iterdir() if p.is_dir() and (p / "training_metrics.jsonl").exists()],
        key=lambda p: p.stat().st_mtime,
    )
    if not run_dirs:
        raise FileNotFoundError(f"No run directories with training metrics found under {results_root}")
    return run_dirs[-1]


def sort_records(records):
    return sorted(records, key=lambda r: (r.get("stream_steps", 0), r.get("timestamp", "")))


def load_eval_records(run_dir: Path):
    history = load_jsonl(run_dir / "torch_eval_history.jsonl")
    if history:
        return sort_records(history)
    eval_dir = run_dir / "evaluations"
    records = []
    for json_path in sorted(eval_dir.glob("*.json")):
        with open(json_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        payload.setdefault("stream_steps", 0)
        records.append(payload)
    return sort_records(records)


def plot_training(training_records, output_dir: Path):
    if not training_records:
        print("No training metrics found; skipping training plots.")
        return

    steps = [r["stream_steps"] for r in training_records]
    env_steps = [r.get("env_steps", 0) for r in training_records]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes[0, 0].plot(steps, [r["policy_loss"] for r in training_records], label="policy_loss")
    axes[0, 0].plot(steps, [r["value_loss"] for r in training_records], label="value_loss")
    axes[0, 0].set_title("Losses vs Stream Steps")
    axes[0, 0].set_xlabel("Stream steps")
    axes[0, 0].legend()

    axes[0, 1].plot(steps, [r["approx_kl"] for r in training_records], label="approx_kl")
    axes[0, 1].plot(steps, [r["clip_fraction"] for r in training_records], label="clip_fraction")
    axes[0, 1].set_title("Policy Stability")
    axes[0, 1].set_xlabel("Stream steps")
    axes[0, 1].legend()

    axes[1, 0].plot(env_steps, [r["steps_per_second"] for r in training_records], label="stream steps/s")
    axes[1, 0].plot(env_steps, [r["env_steps_per_second"] for r in training_records], label="env steps/s")
    axes[1, 0].set_title("Throughput vs Env Steps")
    axes[1, 0].set_xlabel("Env steps")
    axes[1, 0].legend()

    axes[1, 1].plot(steps, [r["learning_rate"] for r in training_records], label="learning_rate")
    axes[1, 1].plot(steps, [r["grad_norm"] for r in training_records], label="grad_norm")
    axes[1, 1].set_title("Optimizer State")
    axes[1, 1].set_xlabel("Stream steps")
    axes[1, 1].legend()

    output_path = output_dir / "training_curves.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_eval(eval_records, output_dir: Path):
    if not eval_records:
        print("No evaluation records found; skipping eval plots.")
        return

    usable = [r for r in eval_records if "mean_reward" in r]
    if not usable:
        print("Evaluation history only contains errors; skipping eval plots.")
        return

    steps = [r.get("stream_steps", 0) for r in usable]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes[0, 0].plot(steps, [r["mean_reward"] for r in usable], marker="o")
    axes[0, 0].set_title("Eval Reward")
    axes[0, 0].set_xlabel("Stream steps")

    axes[0, 1].plot(steps, [r.get("mean_progress", 0.0) for r in usable], marker="o", label="progress")
    axes[0, 1].plot(steps, [r.get("offtrack_rate", 0.0) for r in usable], marker="o", label="offtrack")
    axes[0, 1].set_title("Progress and Off-track")
    axes[0, 1].set_xlabel("Stream steps")
    axes[0, 1].legend()

    axes[1, 0].plot(steps, [r.get("mean_rank", 0.0) for r in usable], marker="o", label="mean_rank")
    axes[1, 0].plot(steps, [r.get("collision_rate", 0.0) for r in usable], marker="o", label="collision_rate")
    axes[1, 0].set_title("Competitive Metrics")
    axes[1, 0].set_xlabel("Stream steps")
    axes[1, 0].legend()

    axes[1, 1].plot(steps, [r.get("mean_speed", 0.0) for r in usable], marker="o", label="mean_speed")
    axes[1, 1].plot(steps, [r.get("mean_overtakes", 0.0) for r in usable], marker="o", label="mean_overtakes")
    axes[1, 1].set_title("Speed and Overtakes")
    axes[1, 1].set_xlabel("Stream steps")
    axes[1, 1].legend()

    output_path = output_dir / "evaluation_curves.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_episodes(episode_records, output_dir: Path):
    if not episode_records:
        print("No episode summaries found; skipping episode plots.")
        return

    steps = [r.get("stream_steps", 0) for r in episode_records]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    axes[0].plot(steps, [r.get("reward", 0.0) for r in episode_records], alpha=0.7)
    axes[0].set_title("Episode Reward")
    axes[0].set_xlabel("Stream steps")
    axes[1].plot(steps, [r.get("length", 0) for r in episode_records], alpha=0.7)
    axes[1].set_title("Episode Length")
    axes[1].set_xlabel("Stream steps")

    output_path = output_dir / "episode_curves.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot saved MARL training/eval artifacts.")
    parser.add_argument("--run-dir", type=str, default=None, help="Run directory containing saved JSONL artifacts")
    parser.add_argument("--results-dir", type=str, default="results", help="Root results directory used to find the latest run")
    parser.add_argument("--show", action="store_true", help="Open generated figures after saving")
    args = parser.parse_args()

    if args.run_dir:
        run_dir = Path(args.run_dir).resolve()
    else:
        run_dir = find_latest_run(Path(args.results_dir).resolve())

    output_dir = run_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using run directory: {run_dir}")
    training_records = sort_records(load_jsonl(run_dir / "training_metrics.jsonl"))
    episode_records = sort_records(load_jsonl(run_dir / "episode_summaries.jsonl"))
    eval_records = load_eval_records(run_dir)

    plot_training(training_records, output_dir)
    plot_eval(eval_records, output_dir)
    plot_episodes(episode_records, output_dir)

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
