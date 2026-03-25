"""
Recursive autoresearch loop with config + code mutations and multi-metric promotion.
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
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

AUTORESEARCH_DIR = Path(__file__).resolve().parent
TRAIN_PPO_PATH = AUTORESEARCH_DIR / "train_ppo.py"
RESULTS_ROOT = AUTORESEARCH_DIR / "results"
PROMOTED_DIRNAME = "promoted"

ALLOWLIST = {
    "ppo.learning_rate": {"type": "float", "min": 1e-5, "max": 1e-3},
    "ppo.n_steps": {"type": "int", "min": 64, "max": 4096},
    "ppo.batch_size": {"type": "int", "min": 64, "max": 4096},
    "ppo.n_epochs": {"type": "int", "min": 1, "max": 10},
    "ppo.gae_lambda": {"type": "float", "min": 0.85, "max": 0.99},
    "ppo.ent_coef": {"type": "float", "min": 0.0, "max": 0.2},
    "ppo.clip_range": {"type": "float", "min": 0.05, "max": 0.4},
    "ppo.min_log_std": {"type": "float", "min": -3.0, "max": 0.0},
    "ppo.max_log_std": {"type": "float", "min": -1.0, "max": 1.0},
    "ppo.steer_min_log_std": {"type": "float", "min": -3.0, "max": 0.0},
    "ppo.steer_max_log_std": {"type": "float", "min": -1.0, "max": 1.0},
    "reward_shaping.sharp_turn_threshold": {"type": "float", "min": 0.15, "max": 0.8},
    "reward_shaping.sharp_turn_lookahead": {"type": "int", "min": 2, "max": 16},
    "reward_shaping.corner_target_speed": {"type": "float", "min": 2.0, "max": 20.0},
    "reward_shaping.corner_overspeed_penalty_scale": {"type": "float", "min": 0.0, "max": 1.0},
    "reward_shaping.apex_decel_reward_scale": {"type": "float", "min": 0.0, "max": 1.0},
    "reward_shaping.apex_decel_reward_cap": {"type": "float", "min": 0.0, "max": 5.0},
    "reward_shaping.steer_smoothness_penalty": {"type": "float", "min": 0.0, "max": 0.2},
    "reward_shaping.steer_magnitude_penalty": {"type": "float", "min": 0.0, "max": 0.1},
    "reward_shaping.lateral_velocity_penalty": {"type": "float", "min": 0.0, "max": 2.0},
    "reward_shaping.time_penalty": {"type": "float", "min": -1.0, "max": 0.0},
    "reward_shaping.brake_penalty_scale": {"type": "float", "min": 0.0, "max": 0.5},
    "reward_shaping.off_track_mode": {"type": "enum", "choices": ["penalty", "terminate"]},
    "reward_shaping.off_track_step_penalty": {"type": "float", "min": -25.0, "max": 0.0},
    "reward_shaping.off_track_terminal_penalty": {"type": "float", "min": -200.0, "max": -1.0},
    "reward_shaping.no_progress_max_steps": {"type": "int", "min": 50, "max": 2000},
    "reward_shaping.no_progress_terminal_penalty": {"type": "float", "min": -100.0, "max": -1.0},
    "reward_shaping.yaw_rate_penalty": {"type": "float", "min": 0.0, "max": 2.0},
    "safety_governor.speed_cap_ratio": {"type": "float", "min": 0.1, "max": 1.5},
    "safety_governor.speed_cap_top_speed": {"type": "float", "min": 10.0, "max": 120.0},
    "safety_governor.speed_cap_brake": {"type": "float", "min": 0.0, "max": 1.0},
}


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds >= 3600:
        return f"{seconds / 3600:.1f}h"
    if seconds >= 60:
        return f"{seconds / 60:.1f}m"
    return f"{seconds:.0f}s"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_yaml(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def strip_fences(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```json"):
        text = text[len("```json"):].strip()
    elif text.startswith("```"):
        text = text[3:].strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    return text


def load_jsonl(path: Path, max_recent: int = 20) -> list[dict]:
    if not path.exists():
        return []
    entries: list[dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries[-max_recent:]


def set_nested(config: dict, dotted_key: str, value) -> None:
    parts = dotted_key.split(".")
    current = config
    for part in parts[:-1]:
        if part not in current or not isinstance(current[part], dict):
            current[part] = {}
        current = current[part]
    current[parts[-1]] = value


def get_nested(config: dict, dotted_key: str):
    current = config
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def apply_overrides(base_config: dict, overrides: dict) -> dict:
    cfg = deepcopy(base_config)
    for key, value in (overrides or {}).items():
        set_nested(cfg, key, value)
    return cfg


def flatten_allowlist() -> str:
    lines = []
    for key, spec in ALLOWLIST.items():
        if spec["type"] == "enum":
            lines.append(f"- {key}: choices={spec['choices']}")
        else:
            lines.append(f"- {key}: {spec['type']} in [{spec['min']}, {spec['max']}]")
    return "\n".join(lines)


def validate_overrides(overrides: dict) -> tuple[bool, list[str], dict]:
    errors: list[str] = []
    cleaned: dict = {}
    for key, value in (overrides or {}).items():
        spec = ALLOWLIST.get(key)
        if spec is None:
            errors.append(f"override not allowed: {key}")
            continue
        try:
            if spec["type"] == "enum":
                if value not in spec["choices"]:
                    raise ValueError(f"{value!r} not in {spec['choices']}")
                cleaned[key] = value
            elif spec["type"] == "int":
                ivalue = int(value)
                if ivalue < spec["min"] or ivalue > spec["max"]:
                    raise ValueError(f"{ivalue} outside [{spec['min']}, {spec['max']}]")
                cleaned[key] = ivalue
            else:
                fvalue = float(value)
                if fvalue < spec["min"] or fvalue > spec["max"]:
                    raise ValueError(f"{fvalue} outside [{spec['min']}, {spec['max']}]")
                cleaned[key] = fvalue
        except (TypeError, ValueError) as exc:
            errors.append(f"{key}: {exc}")
    return len(errors) == 0, errors, cleaned


def infer_patterns(metrics: dict) -> list[str]:
    patterns: list[str] = []
    progress = safe_float(metrics.get("mean_progress"), 0.0)
    offtrack = safe_float(metrics.get("offtrack_rate"), 0.0)
    speed = safe_float(metrics.get("mean_speed"), 0.0)
    steer_var = safe_float(metrics.get("mean_steer_variance"), 0.0)
    throttle = safe_float(metrics.get("mean_throttle"), 0.0)
    brake = safe_float(metrics.get("mean_brake"), 0.0)
    episode_len = safe_float(metrics.get("mean_episode_length"), 0.0)
    std_len = safe_float(metrics.get("std_episode_length"), 0.0)

    if progress < 0.05 and offtrack >= 0.8:
        patterns.append("early_offtrack")
    if throttle >= 0.95 and brake <= 0.05:
        patterns.append("throttle_saturation_no_brake")
    if steer_var <= 0.02 and offtrack >= 0.8:
        patterns.append("low_steer_variance_circling_or_understeer")
    if speed < 6.0 and progress < 0.1:
        patterns.append("speed_collapse")
    if speed >= 8.0 and progress < 0.15 and offtrack >= 0.6:
        patterns.append("instability_collapse")
    if episode_len > 0 and std_len <= max(10.0, 0.05 * episode_len) and progress < 0.2:
        patterns.append("same_length_failure_signature")
    if brake <= 0.02 and progress < 0.2 and offtrack >= 0.5:
        patterns.append("late_or_missing_brake")
    return patterns


def gate_candidate(metrics: dict, parent_metrics: dict | None) -> tuple[bool, list[str], float]:
    reasons: list[str] = []
    reward = safe_float(metrics.get("mean_reward"), -999.0)
    progress = safe_float(metrics.get("mean_progress"), 0.0)
    offtrack = safe_float(metrics.get("offtrack_rate"), 1.0)
    speed = safe_float(metrics.get("mean_speed"), 0.0)
    throttle = safe_float(metrics.get("mean_throttle"), 0.0)

    min_reward = -500.0
    min_progress = 0.05
    max_offtrack = 0.60
    min_speed = 5.0

    if parent_metrics:
        parent_reward = safe_float(parent_metrics.get("mean_reward"), min_reward)
        parent_progress = safe_float(parent_metrics.get("mean_progress"), min_progress)
        parent_speed = safe_float(parent_metrics.get("mean_speed"), min_speed)
        min_reward = max(min_reward, parent_reward - 150.0)
        min_progress = max(min_progress, 0.80 * parent_progress)
        min_speed = max(min_speed, 0.80 * parent_speed)
        max_offtrack = min(max_offtrack, safe_float(parent_metrics.get("offtrack_rate"), max_offtrack) + 0.15)

    if reward < min_reward:
        reasons.append(f"reward<{min_reward:.2f}")
    if progress < min_progress:
        reasons.append(f"progress<{min_progress:.3f}")
    if offtrack > max_offtrack:
        reasons.append(f"offtrack>{max_offtrack:.3f}")
    if speed < min_speed:
        reasons.append(f"speed<{min_speed:.2f}")
    if throttle >= 0.98 and progress < 0.2:
        reasons.append("throttle_saturated_without_progress")

    score = reward + 250.0 * progress + 5.0 * speed - 200.0 * offtrack - 25.0 * max(0.0, throttle - 0.95)
    return len(reasons) == 0, reasons, score


def summarize_history(history: list[dict], max_items: int = 8) -> str:
    if not history:
        return "No prior recursive generations."
    lines = []
    for item in history[-max_items:]:
        lines.append(
            f"- {item.get('candidate_name', item.get('candidate_id', 'unknown'))}: "
            f"reward={safe_float(item.get('mean_reward'), -999.0):.2f}, "
            f"progress={safe_float(item.get('mean_progress'), 0.0):.3f}, "
            f"speed={safe_float(item.get('mean_speed'), 0.0):.2f}, "
            f"offtrack={safe_float(item.get('offtrack_rate'), 1.0):.3f}, "
            f"patterns={item.get('patterns', [])}, "
            f"promoted={bool(item.get('promoted', False))}"
        )
    return "\n".join(lines)


def call_llm_for_candidates(
    parent_metrics: dict | None,
    parent_patterns: list[str],
    recent_history: list[dict],
    parent_config: dict,
    parent_code: str,
    candidates_to_generate: int,
    model: str,
) -> list[dict]:
    try:
        from google import genai
    except ImportError:
        print("[run_recursive] ERROR: Gemini SDK not installed. Run: pip install google-genai", file=sys.stderr)
        sys.exit(1)

    client = genai.Client()
    parent_summary = json.dumps(parent_metrics or {}, indent=2)
    tracked_keys = sorted(ALLOWLIST.keys())
    parent_key_values = {key: get_nested(parent_config, key) for key in tracked_keys if get_nested(parent_config, key) is not None}
    system_prompt = """You are designing candidate PPO finetuning experiments for a racing agent.
Return JSON only. No markdown.
Each candidate must make 1-2 focused changes.
Prefer config overrides over code changes unless a policy/PPO code edit is truly needed.
Do not propose keys outside the allowlist.
Keep changes bounded and incremental.
If code_change is "replace", provide the full replacement Python file contents for autoresearch/train_ppo.py.
If code is unchanged, set code_change to "keep_parent" and omit train_ppo_code.
"""
    user_prompt = f"""Create {candidates_to_generate} candidate experiments.

Current parent metrics:
{parent_summary}

Current inferred patterns:
{json.dumps(parent_patterns)}

Recent recursive history:
{summarize_history(recent_history)}

Current allowlisted config values:
{json.dumps(parent_key_values, indent=2)}

Allowlisted fields and bounds:
{flatten_allowlist()}

Current train_ppo.py:
```python
{parent_code}
```

Output JSON with this shape:
{{
  "candidates": [
    {{
      "name": "short_snake_case_name",
      "rationale": "one sentence",
      "config_overrides": {{"dotted.key": 0.1}},
      "code_change": "keep_parent" | "replace",
      "train_ppo_code": "full python file only when code_change=replace"
    }}
  ]
}}
"""
    response = client.models.generate_content(
        model=model,
        contents=f"{system_prompt}\n\n{user_prompt}",
    )
    raw_text = strip_fences(response.text or "")
    if not raw_text:
        raise RuntimeError("Gemini returned empty candidate JSON")
    payload = json.loads(raw_text)
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("Gemini output missing candidates list")
    return candidates


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


def run_candidate_experiment(
    *,
    config_path: Path,
    timesteps: int,
    num_envs: int,
    eval_episodes: int,
    seed: int,
    candidate_id: str,
    run_dir: Path,
    timeout: int,
    resume: Path | None,
    resume_mode: str,
) -> dict:
    stdout_log_path = run_dir / "stdout.log"
    stderr_log_path = run_dir / "stderr.log"
    cmd = [
        sys.executable,
        "-m",
        "autoresearch.run_experiment",
        "--config",
        str(config_path),
        "--timesteps",
        str(timesteps),
        "--eval-episodes",
        str(eval_episodes),
        "--seed",
        str(seed),
        "--checkpoint-dir",
        str(run_dir),
        "--num-envs",
        str(num_envs),
        "--experiment-id",
        candidate_id,
        "--resume-mode",
        str(resume_mode),
    ]
    if resume is not None:
        cmd.extend(["--resume", str(resume)])

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
        threads = [
            threading.Thread(target=_stream_reader, args=(proc.stdout, "stdout", output_queue, stdout_sink), daemon=True),
            threading.Thread(target=_stream_reader, args=(proc.stderr, "stderr", output_queue, stderr_sink), daemon=True),
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
                sys.stderr.write(f"  {line}")
            sys.stderr.flush()

        for thread in threads:
            thread.join(timeout=1)
        return_code = proc.wait()

    if timed_out:
        return {"mean_reward": -999.0, "error": f"Timeout after {timeout}s", "return_code": return_code}

    for line in reversed(stdout_lines):
        candidate = line.strip()
        if not candidate:
            continue
        try:
            metrics = json.loads(candidate)
            metrics["return_code"] = return_code
            return metrics
        except json.JSONDecodeError:
            continue
    return {"mean_reward": -999.0, "error": "No JSON metrics found in stdout", "return_code": return_code}


def ensure_current_artifacts(branch_dir: Path, checkpoint_path: Path, code_path: Path, config_path: Path, metrics: dict) -> None:
    promoted_dir = branch_dir / PROMOTED_DIRNAME
    promoted_dir.mkdir(parents=True, exist_ok=True)
    if checkpoint_path.exists():
        shutil.copy2(checkpoint_path, promoted_dir / "checkpoint.pt")
    if code_path.exists():
        shutil.copy2(code_path, promoted_dir / "train_ppo.py")
    if config_path.exists():
        shutil.copy2(config_path, promoted_dir / "config.yaml")
    with open(promoted_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)


def make_baseline_candidate(parent_code: str) -> dict:
    return {
        "name": "baseline_continuation",
        "rationale": "Continue training the current parent unchanged as the control candidate.",
        "config_overrides": {},
        "code_change": "keep_parent",
        "train_ppo_code": parent_code,
        "source": "baseline",
    }


def build_branch_state(
    *,
    branch_dir: Path,
    base_config_path: Path,
    base_checkpoint_path: Path,
    base_code_path: Path,
    mode: str,
    candidates_per_batch: int,
    timesteps_per_candidate: int,
    eval_episodes: int,
    generations: int,
    model: str,
) -> dict:
    return {
        "branch_dir": str(branch_dir),
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "mode": mode,
        "candidates_per_batch": candidates_per_batch,
        "timesteps_per_candidate": timesteps_per_candidate,
        "eval_episodes": eval_episodes,
        "max_generations": generations,
        "model": model,
        "current_generation": 0,
        "parent": {
            "checkpoint_path": str(base_checkpoint_path.resolve()),
            "config_path": str(base_config_path.resolve()),
            "code_path": str(base_code_path.resolve()),
            "metrics": None,
            "patterns": [],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Recursive config + code autoresearch tuner")
    parser.add_argument("--base-config", type=str, required=True, help="Base YAML config to start from")
    parser.add_argument("--base-checkpoint", type=str, required=True, help="Base checkpoint to warm-start from")
    parser.add_argument("--base-code", type=str, required=True, help="Base train_ppo.py snapshot to start from")
    parser.add_argument("--results-subdir", type=str, required=True, help="Subdirectory under autoresearch/results")
    parser.add_argument("--mode", type=str, default="fully_autonomous", choices=["fully_autonomous", "semi_autonomous"])
    parser.add_argument("--candidates-per-batch", type=int, default=4)
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--model", type=str, default="gemini-2.5-flash")
    args = parser.parse_args()

    branch_dir = RESULTS_ROOT / args.results_subdir
    branch_dir.mkdir(parents=True, exist_ok=True)
    generations_log = branch_dir / "generations.jsonl"
    branch_state_path = branch_dir / "branch_state.json"

    base_config_path = Path(args.base_config).resolve()
    base_checkpoint_path = Path(args.base_checkpoint).resolve()
    base_code_path = Path(args.base_code).resolve()
    if not base_config_path.is_file():
        raise FileNotFoundError(f"Base config not found: {base_config_path}")
    if not base_checkpoint_path.is_file():
        raise FileNotFoundError(f"Base checkpoint not found: {base_checkpoint_path}")
    if not base_code_path.is_file():
        raise FileNotFoundError(f"Base code not found: {base_code_path}")

    if branch_state_path.exists():
        branch_state = json.loads(branch_state_path.read_text(encoding="utf-8"))
        log(f"[run_recursive] Resuming existing branch: {branch_dir}")
    else:
        branch_state = build_branch_state(
            branch_dir=branch_dir,
            base_config_path=base_config_path,
            base_checkpoint_path=base_checkpoint_path,
            base_code_path=base_code_path,
            mode=args.mode,
            candidates_per_batch=args.candidates_per_batch,
            timesteps_per_candidate=args.timesteps,
            eval_episodes=args.eval_episodes,
            generations=args.generations,
            model=args.model,
        )
        branch_state_path.write_text(json.dumps(branch_state, indent=2), encoding="utf-8")

    recent_entries = load_jsonl(generations_log, max_recent=30)
    recent_history: list[dict] = []
    for entry in recent_entries:
        if isinstance(entry, dict) and isinstance(entry.get("records"), list):
            recent_history.extend(entry["records"])
        elif isinstance(entry, dict):
            recent_history.append(entry)
    recent_history = recent_history[-30:]

    for generation_index in range(int(branch_state.get("current_generation", 0)) + 1, args.generations + 1):
        parent = branch_state["parent"]
        parent_checkpoint_path = Path(parent["checkpoint_path"])
        parent_config_path = Path(parent["config_path"])
        parent_code_path = Path(parent["code_path"])
        parent_config = load_yaml(parent_config_path)
        parent_code = read_text(parent_code_path)
        parent_metrics = parent.get("metrics")
        parent_patterns = parent.get("patterns", [])

        gen_dir = branch_dir / f"generation_{generation_index:03d}"
        gen_dir.mkdir(parents=True, exist_ok=True)
        log("=" * 72)
        log(f"[run_recursive] Generation {generation_index}/{args.generations}")
        log(f"[run_recursive] Parent checkpoint: {parent_checkpoint_path}")

        candidates: list[dict] = [make_baseline_candidate(parent_code)]
        llm_needed = max(0, args.candidates_per_batch - 1)
        if llm_needed > 0:
            try:
                llm_candidates = call_llm_for_candidates(
                    parent_metrics=parent_metrics,
                    parent_patterns=parent_patterns,
                    recent_history=recent_history,
                    parent_config=parent_config,
                    parent_code=parent_code,
                    candidates_to_generate=llm_needed,
                    model=args.model,
                )
                candidates.extend(llm_candidates[:llm_needed])
            except Exception as exc:
                log(f"[run_recursive] Candidate generation failed: {exc}")

        generation_records: list[dict] = []
        for idx, candidate in enumerate(candidates, start=1):
            candidate_name = str(candidate.get("name", f"candidate_{idx:02d}")).strip().lower().replace(" ", "_")
            candidate_dir = gen_dir / f"{idx:02d}_{candidate_name}"
            candidate_dir.mkdir(parents=True, exist_ok=True)
            overrides = candidate.get("config_overrides", {}) or {}
            ok, errors, cleaned_overrides = validate_overrides(overrides)
            if not ok:
                record = {
                    "generation": generation_index,
                    "candidate_index": idx,
                    "candidate_name": candidate_name,
                    "error": "invalid_overrides",
                    "override_errors": errors,
                    "promoted": False,
                    "passed_gate": False,
                }
                generation_records.append(record)
                with open(candidate_dir / "candidate_spec.json", "w", encoding="utf-8") as handle:
                    json.dump({**candidate, "override_errors": errors}, handle, indent=2)
                continue

            candidate_code = parent_code
            code_change = str(candidate.get("code_change", "keep_parent")).strip().lower()
            if code_change == "replace":
                candidate_code = candidate.get("train_ppo_code", "") or ""
                try:
                    compile(candidate_code, "train_ppo.py", "exec")
                except SyntaxError as exc:
                    record = {
                        "generation": generation_index,
                        "candidate_index": idx,
                        "candidate_name": candidate_name,
                        "error": f"invalid_code: {exc}",
                        "promoted": False,
                        "passed_gate": False,
                    }
                    generation_records.append(record)
                    with open(candidate_dir / "candidate_spec.json", "w", encoding="utf-8") as handle:
                        json.dump({**candidate, "code_error": str(exc)}, handle, indent=2)
                    continue
            elif code_change != "keep_parent":
                generation_records.append({
                    "generation": generation_index,
                    "candidate_index": idx,
                    "candidate_name": candidate_name,
                    "error": f"unknown_code_change:{code_change}",
                    "promoted": False,
                    "passed_gate": False,
                })
                continue

            candidate_config = apply_overrides(parent_config, cleaned_overrides)
            candidate_config_path = candidate_dir / "candidate_config.yaml"
            candidate_code_path = candidate_dir / "candidate_train_ppo.py"
            write_yaml(candidate_config_path, candidate_config)
            candidate_code_path.write_text(candidate_code, encoding="utf-8")
            with open(candidate_dir / "candidate_spec.json", "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "generation": generation_index,
                        "candidate_index": idx,
                        "candidate_name": candidate_name,
                        "rationale": candidate.get("rationale", ""),
                        "source": candidate.get("source", "llm"),
                        "config_overrides": cleaned_overrides,
                        "code_change": code_change,
                        "parent_checkpoint": str(parent_checkpoint_path),
                    },
                    handle,
                    indent=2,
                )

            TRAIN_PPO_PATH.write_text(candidate_code, encoding="utf-8")
            log(f"[run_recursive] Running {candidate_name} | overrides={cleaned_overrides}")
            metrics = run_candidate_experiment(
                config_path=candidate_config_path,
                timesteps=args.timesteps,
                num_envs=args.num_envs,
                eval_episodes=args.eval_episodes,
                seed=args.seed + generation_index * 100 + idx,
                candidate_id=f"g{generation_index:03d}_c{idx:02d}",
                run_dir=candidate_dir,
                timeout=args.timeout,
                resume=parent_checkpoint_path,
                resume_mode="policy_only",
            )
            patterns = infer_patterns(metrics)
            passed_gate, gate_reasons, score = gate_candidate(metrics, parent_metrics)
            record = {
                "generation": generation_index,
                "candidate_index": idx,
                "candidate_id": f"g{generation_index:03d}_c{idx:02d}",
                "candidate_name": candidate_name,
                "rationale": candidate.get("rationale", ""),
                "source": candidate.get("source", "llm"),
                "config_overrides": cleaned_overrides,
                "code_change": code_change,
                "candidate_config_path": str(candidate_config_path),
                "candidate_code_path": str(candidate_code_path),
                "checkpoint_path": str(candidate_dir / "final.pt"),
                "passed_gate": passed_gate,
                "gate_reasons": gate_reasons,
                "patterns": patterns,
                "score": score,
                "resume_mode": "policy_only",
                "promoted": False,
                **metrics,
            }
            with open(candidate_dir / "metrics.json", "w", encoding="utf-8") as handle:
                json.dump(record, handle, indent=2)
            generation_records.append(record)

        TRAIN_PPO_PATH.write_text(parent_code, encoding="utf-8")
        passing = [r for r in generation_records if r.get("passed_gate")]
        promoted_record = max(passing, key=lambda item: safe_float(item.get("score"), -1e9)) if passing else None
        if promoted_record is not None:
            promoted_record["promoted"] = True
            promoted_checkpoint = Path(promoted_record["checkpoint_path"])
            promoted_code_path = Path(promoted_record["candidate_code_path"])
            promoted_config_path = Path(promoted_record["candidate_config_path"])
            parent.update(
                {
                    "checkpoint_path": str(promoted_checkpoint),
                    "code_path": str(promoted_code_path),
                    "config_path": str(promoted_config_path),
                    "metrics": {k: v for k, v in promoted_record.items() if k not in {"candidate_code_path", "candidate_config_path"}},
                    "patterns": promoted_record.get("patterns", []),
                }
            )
            ensure_current_artifacts(branch_dir, promoted_checkpoint, promoted_code_path, promoted_config_path, promoted_record)
            log(
                f"[run_recursive] PROMOTED {promoted_record['candidate_name']} | "
                f"reward={safe_float(promoted_record.get('mean_reward'), -999.0):.2f} | "
                f"progress={safe_float(promoted_record.get('mean_progress'), 0.0):.3f} | "
                f"speed={safe_float(promoted_record.get('mean_speed'), 0.0):.2f}"
            )
        else:
            log("[run_recursive] No candidate passed the promotion gate. Retaining current parent.")

        generation_summary = {
            "timestamp": datetime.now().isoformat(),
            "generation": generation_index,
            "mode": args.mode,
            "parent_checkpoint_before": str(parent_checkpoint_path),
            "promoted_candidate": promoted_record.get("candidate_name") if promoted_record else None,
            "records": generation_records,
        }
        with open(generations_log, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(generation_summary) + "\n")

        branch_state["current_generation"] = generation_index
        branch_state["updated_at"] = datetime.now().isoformat()
        branch_state["parent"] = parent
        branch_state_path.write_text(json.dumps(branch_state, indent=2), encoding="utf-8")

        recent_history.extend(generation_records)
        recent_history = recent_history[-30:]

        if args.mode == "semi_autonomous":
            log("[run_recursive] Semi-autonomous mode: stopping after this generation.")
            break

    log("=" * 72)
    log("[run_recursive] COMPLETE")
    log(f"[run_recursive] branch_dir={branch_dir}")
    log(f"[run_recursive] promoted_checkpoint={(branch_dir / PROMOTED_DIRNAME / 'checkpoint.pt')}")


if __name__ == "__main__":
    main()
