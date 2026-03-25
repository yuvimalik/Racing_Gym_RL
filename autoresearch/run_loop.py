"""
LOCKED - Autoresearch orchestration loop (Karpathy-style).

Iteratively:
  1. Calls Gemini API with experiment history + program.md + current train_ppo.py
  2. Writes new train_ppo.py
  3. Runs experiment as subprocess (with timeout)
  4. Compares to best and promotes result artifacts
  5. Appends to experiments.jsonl

Usage:
    python -m autoresearch.run_loop --config config/multi_car_config.yaml --max-experiments 20
"""

import argparse
import json
import queue
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

AUTORESEARCH_DIR = Path(__file__).resolve().parent
TRAIN_PPO_PATH = AUTORESEARCH_DIR / "train_ppo.py"
PROGRAM_PATH = AUTORESEARCH_DIR / "program.md"
RESULTS_DIR = AUTORESEARCH_DIR / "results"


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds >= 3600:
        return f"{seconds / 3600:.1f}h"
    if seconds >= 60:
        return f"{seconds / 60:.1f}m"
    return f"{seconds:.0f}s"


def safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def load_experiment_history(experiments_log: Path, max_recent: int = 10) -> list:
    if not experiments_log.exists():
        return []
    history = []
    with open(experiments_log, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                history.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return history[-max_recent:]


def get_best_entry(history: list) -> dict | None:
    best_entry = None
    best_reward = -float("inf")
    for entry in history:
        reward = safe_float(entry.get("mean_reward"), default=-float("inf"))
        if reward > best_reward:
            best_reward = reward
            best_entry = entry
    return best_entry


def call_llm_api(history: list, program: str, current_code: str, model: str = "gemini-2.5-flash") -> str:
    try:
        from google import genai
    except ImportError:
        print(
            "[run_loop] ERROR: Gemini SDK not installed. Run: pip install google-genai",
            file=sys.stderr,
        )
        sys.exit(1)

    client = genai.Client()

    history_text = ""
    if history:
        history_text = "## Recent Experiment Results (most recent last)\n\n"
        for i, exp in enumerate(history):
            history_text += (
                f"### Experiment {exp.get('experiment_id', i)}\n"
                f"- mean_reward: {safe_float(exp.get('mean_reward'), default=float('nan')):.2f}\n"
                f"- mean_progress: {safe_float(exp.get('mean_progress'), default=0.0):.4f}\n"
                f"- mean_speed: {safe_float(exp.get('mean_speed'), default=0.0):.2f}\n"
                f"- steps/s: {safe_float(exp.get('steps_per_second'), default=0.0):.1f}\n"
                f"- was_best: {bool(exp.get('was_best', False))}\n"
            )
            if "error" in exp:
                history_text += f"- ERROR: {exp['error']}\n"
            if "changes_description" in exp:
                history_text += f"- changes: {exp['changes_description']}\n"
            history_text += "\n"

    system_prompt = """You are an expert RL researcher optimizing a PPO agent for car racing.

You will be given:
1. A research program (goals and search priorities)
2. Recent experiment history (what was tried, what worked)
3. The current train_ppo.py code

Your job: produce an IMPROVED version of train_ppo.py that you believe will achieve higher mean_reward.

RULES:
- Output ONLY the complete Python code for train_ppo.py - no markdown, no explanations
- Keep the same class interfaces (CnnActorCritic with .act() and .raw_to_env_action(), PPOTrainer with .train())
- The HYPERPARAMS dict must remain at module level
- Do NOT import from train.py - only use autoresearch.prepare for env/eval
- Make ONE focused change per experiment (not five changes at once)
- If the last experiment crashed, fix the bug before trying something new
- If a change improved reward, consider pushing that direction further
- If a change hurt reward, revert it and try something different

Common improvements to try:
- Action distribution: Beta for bounded actions, squashed Gaussian
- Network: LayerNorm, skip connections, deeper value head
- LR schedule: cosine annealing, warmup
- Entropy: scheduled decay
- GAE lambda: tune between 0.9-0.97
- Value loss: Huber instead of MSE
- Reward/observation normalization
- Gradient accumulation for larger effective batch size"""

    user_message = f"""## Research Program
{program}

## Experiment History
{history_text if history_text else "No experiments run yet - this is the first iteration."}

## Current train_ppo.py
```python
{current_code}
```

Produce an improved train_ppo.py. Output ONLY the Python code, nothing else."""

    response = client.models.generate_content(
        model=model,
        contents=f"{system_prompt}\n\n{user_message}",
    )
    code = (response.text or "").strip()
    if code.startswith("```python"):
        code = code[len("```python"):].strip()
    if code.startswith("```"):
        code = code[3:].strip()
    if code.endswith("```"):
        code = code[:-3].strip()
    if not code:
        raise RuntimeError("Gemini returned empty content")
    return code


def estimate_runtime(history: list, timesteps: int) -> str:
    usable = [
        safe_float(entry.get("steps_per_second"), default=0.0)
        for entry in history
        if safe_float(entry.get("steps_per_second"), default=0.0) > 0
    ]
    if not usable:
        return "unknown"
    avg_sps = sum(usable) / len(usable)
    return format_duration(timesteps / max(avg_sps, 1e-6))


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


def run_experiment(
    config_path: str,
    timesteps: int,
    num_envs: int,
    eval_episodes: int,
    seed: int,
    experiment_id: int,
    run_dir: Path,
    timeout: int = 600,
    resume: str = None,
) -> dict:
    checkpoint_dir = run_dir
    stdout_log_path = run_dir / "stdout.log"
    stderr_log_path = run_dir / "stderr.log"
    metrics_path = run_dir / "metrics.json"

    cmd = [
        sys.executable,
        "-m",
        "autoresearch.run_experiment",
        "--config",
        config_path,
        "--timesteps",
        str(timesteps),
        "--eval-episodes",
        str(eval_episodes),
        "--seed",
        str(seed),
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--num-envs",
        str(num_envs),
        "--experiment-id",
        str(experiment_id),
    ]
    if resume:
        cmd.extend(["--resume", str(resume)])

    log(f"[run_loop] Command: {' '.join(cmd)}")

    with open(stdout_log_path, "w", encoding="utf-8") as stdout_sink, open(
        stderr_log_path, "w", encoding="utf-8"
    ) as stderr_sink:
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        output_queue: queue.Queue = queue.Queue()
        stdout_lines: list[str] = []
        stderr_lines: list[str] = []

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

            if stream_name == "stdout":
                stdout_lines.append(line.rstrip("\n"))
                sys.stderr.write(f"  [child-stdout] {line}")
            else:
                stderr_lines.append(line.rstrip("\n"))
                sys.stderr.write(f"  {line}")
            sys.stderr.flush()

        for thread in threads:
            thread.join(timeout=1)
        return_code = proc.wait()

    if timed_out:
        metrics = {
            "mean_reward": -999.0,
            "error": f"Timeout after {timeout}s",
            "return_code": return_code,
        }
    else:
        last_json_line = None
        for line in reversed(stdout_lines):
            candidate = line.strip()
            if not candidate:
                continue
            try:
                last_json_line = json.loads(candidate)
                break
            except json.JSONDecodeError:
                continue
        if last_json_line is None:
            metrics = {
                "mean_reward": -999.0,
                "error": "No JSON metrics found in stdout",
                "return_code": return_code,
            }
        else:
            metrics = dict(last_json_line)
            metrics["return_code"] = return_code

    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    return metrics


def promote_best(
    run_dir: Path,
    log_entry: dict,
    best_dir: Path,
    best_code_path: Path,
    best_metrics_path: Path,
) -> None:
    best_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = run_dir / "final.pt"
    if checkpoint_path.exists():
        shutil.copy2(checkpoint_path, best_dir / "final.pt")

    candidate_code_path = run_dir / "candidate_train_ppo.py"
    if candidate_code_path.exists():
        shutil.copy2(candidate_code_path, best_code_path)

    with open(best_metrics_path, "w", encoding="utf-8") as handle:
        json.dump(log_entry, handle, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Autoresearch orchestration loop")
    parser.add_argument("--config", type=str, default="config/multi_car_config.yaml")
    parser.add_argument("--max-experiments", type=int, default=20)
    parser.add_argument("--timesteps", type=int, default=500_000, help="Training budget per experiment")
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=600, help="Per-experiment timeout in seconds")
    parser.add_argument("--model", type=str, default="gemini-2.5-flash", help="Gemini model for code generation")
    parser.add_argument("--skip-first-edit", action="store_true",
                        help="Run first experiment with current train_ppo.py (baseline)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to warm-start all experiments from")
    parser.add_argument("--results-subdir", type=str, default=None,
                        help="Optional subdirectory under autoresearch/results for an isolated branch")
    parser.add_argument("--program", type=str, default=None,
                        help="Optional research program markdown path")
    parser.add_argument("--bootstrap-code", type=str, default=None,
                        help="Optional starting train_ppo.py snapshot for a fresh branch")
    args = parser.parse_args()

    branch_results_dir = RESULTS_DIR / args.results_subdir if args.results_subdir else RESULTS_DIR
    experiments_log = branch_results_dir / "experiments.jsonl"
    best_dir = branch_results_dir / "best"
    best_code_path = branch_results_dir / "best_train_ppo.py"
    best_metrics_path = branch_results_dir / "best_metrics.json"

    branch_results_dir.mkdir(parents=True, exist_ok=True)
    best_dir.mkdir(parents=True, exist_ok=True)

    program_path = Path(args.program) if args.program else PROGRAM_PATH
    if program_path.exists():
        program = program_path.read_text(encoding="utf-8")
    else:
        program = "Maximize mean_reward. Try one focused change per experiment."
        log(f"[run_loop] WARNING: No program file found at {program_path}. Using default.")

    history = load_experiment_history(experiments_log)
    best_code = TRAIN_PPO_PATH.read_text(encoding="utf-8")
    if best_code_path.exists():
        best_code = best_code_path.read_text(encoding="utf-8")
        TRAIN_PPO_PATH.write_text(best_code, encoding="utf-8")
    elif args.bootstrap_code:
        bootstrap_code_path = Path(args.bootstrap_code)
        best_code = bootstrap_code_path.read_text(encoding="utf-8")
        TRAIN_PPO_PATH.write_text(best_code, encoding="utf-8")
    best_entry = get_best_entry(history)
    best_reward = safe_float(best_entry["mean_reward"], default=-float("inf")) if best_entry else -float("inf")
    experiment_id = max((int(entry.get("experiment_id", 0)) for entry in history), default=0)

    log("#" * 72)
    log("[run_loop] AUTORESEARCH LOOP")
    log(f"[run_loop] max_experiments={args.max_experiments}")
    log(f"[run_loop] timesteps_per_experiment={args.timesteps:,}")
    log(f"[run_loop] num_envs={args.num_envs} | eval_episodes={args.eval_episodes}")
    log(f"[run_loop] results_dir={branch_results_dir}")
    log(f"[run_loop] program={program_path}")
    log(f"[run_loop] starting_best_reward={best_reward:.2f}")
    log(f"[run_loop] best_checkpoint={(best_dir / 'final.pt') if (best_dir / 'final.pt').exists() else 'none'}")
    log("#" * 72)

    for loop_index in range(args.max_experiments):
        experiment_id += 1
        run_dir = branch_results_dir / f"run_{experiment_id:03d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()

        current_code = best_code
        changes_description = "baseline (no changes)"
        is_baseline = loop_index == 0 and args.skip_first_edit

        log("-" * 72)
        log(
            f"[run_loop] Experiment {loop_index + 1}/{args.max_experiments} "
            f"(id={experiment_id}) | mode={'baseline' if is_baseline else 'edited'} | "
            f"best_reward={best_reward:.2f}"
        )
        log(f"[run_loop] Resume source: {args.resume if args.resume else 'none'}")
        log(f"[run_loop] Estimated runtime: {estimate_runtime(history, args.timesteps)}")

        if not is_baseline:
            log(f"[run_loop] Calling Gemini API for experiment {experiment_id}...")
            try:
                new_code = call_llm_api(history, program, current_code, model=args.model)
                compile(new_code, "train_ppo.py", "exec")
                TRAIN_PPO_PATH.write_text(new_code, encoding="utf-8")
                changes_description = "LLM-generated changes"
                current_code = new_code
                log(f"[run_loop] New train_ppo.py written ({len(new_code)} chars)")
            except SyntaxError as exc:
                log(f"[run_loop] LLM produced invalid code: {exc}")
                continue
            except Exception as exc:
                log(f"[run_loop] Gemini API error: {exc}")
                continue

        candidate_code_path = run_dir / "candidate_train_ppo.py"
        candidate_code_path.write_text(current_code, encoding="utf-8")

        metrics = run_experiment(
            config_path=args.config,
            timesteps=args.timesteps,
            num_envs=args.num_envs,
            eval_episodes=args.eval_episodes,
            seed=args.seed + experiment_id,
            experiment_id=experiment_id,
            run_dir=run_dir,
            timeout=args.timeout,
            resume=args.resume,
        )

        mean_reward = safe_float(metrics.get("mean_reward"), default=-999.0)
        has_best_artifacts = (best_dir / "final.pt").exists() and best_code_path.exists()
        checkpoint_exists = (run_dir / "final.pt").exists()
        bootstrap_best = (not has_best_artifacts) and checkpoint_exists and ("error" not in metrics)
        actual_best = mean_reward > best_reward
        promoted = actual_best or bootstrap_best
        loop_time = time.time() - t0

        log_entry = {
            "experiment_id": experiment_id,
            "timestamp": datetime.now().isoformat(),
            "was_best": actual_best,
            "promoted_to_best_artifacts": promoted,
            "changes_description": changes_description,
            "loop_wall_clock_seconds": loop_time,
            "run_dir": str(run_dir),
            "candidate_code_path": str(candidate_code_path),
            **metrics,
        }

        with open(run_dir / "metrics.json", "w", encoding="utf-8") as handle:
            json.dump(log_entry, handle, indent=2)

        with open(experiments_log, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(log_entry) + "\n")

        if promoted:
            if actual_best or best_entry is None:
                best_reward = mean_reward
                best_entry = log_entry
            best_code = current_code
            promote_best(run_dir, log_entry, best_dir, best_code_path, best_metrics_path)
            TRAIN_PPO_PATH.write_text(best_code, encoding="utf-8")
            if actual_best:
                log(f"[run_loop] NEW BEST | reward={mean_reward:.2f} | progress={safe_float(metrics.get('mean_progress'), default=0.0):.4f}")
            else:
                log(
                    f"[run_loop] Bootstrapped best artifacts from experiment {experiment_id} | "
                    f"reward={mean_reward:.2f}"
                )
        else:
            TRAIN_PPO_PATH.write_text(best_code, encoding="utf-8")
            log(
                "[run_loop] Not promoted | "
                f"reward={mean_reward:.2f} | best={best_reward:.2f}"
            )

        history.append(log_entry)
        history = history[-10:]

        log(
            "[run_loop] Experiment summary | "
            f"reward={mean_reward:.2f} | "
            f"progress={safe_float(metrics.get('mean_progress'), default=0.0):.4f} | "
            f"steps_s={safe_float(metrics.get('steps_per_second'), default=0.0):.1f} | "
            f"wall={format_duration(loop_time)} | "
            f"was_best={actual_best} | promoted={promoted}"
        )

    best_checkpoint = best_dir / "final.pt"
    log("#" * 72)
    log("[run_loop] AUTORESEARCH COMPLETE")
    log(f"[run_loop] best_experiment_id={best_entry.get('experiment_id') if best_entry else 'none'}")
    log(f"[run_loop] best_reward={best_reward:.2f}")
    log(f"[run_loop] best_checkpoint={best_checkpoint if best_checkpoint.exists() else 'none'}")
    log(f"[run_loop] best_code_snapshot={best_code_path if best_code_path.exists() else 'none'}")
    log("#" * 72)


if __name__ == "__main__":
    main()
