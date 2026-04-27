"""
Recursive autoresearch loop for the maintained multi-agent training stack.
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
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

from autoresearch.marl_search_utils import (
    MARL_ALLOWLIST,
    VALIDATION_LADDER,
    apply_overrides,
    flatten_allowlist,
    gate_candidate,
    get_nested,
    infer_patterns,
    render_code_diff,
    render_override_summary,
    safe_float,
    strip_fences,
    summarize_history,
    validate_overrides,
    write_human_review,
)
from autoresearch.llm_client import (
    LlmProviderError,
    default_model_for_provider,
    generate_text,
    infer_provider_from_model,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from autoresearch.load_env import load_project_env

    load_project_env()
except ImportError:
    pass

AUTORESEARCH_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = AUTORESEARCH_DIR / "results"
PROGRAM_PATH = AUTORESEARCH_DIR / "program_marl.md"
BASE_SURFACE_PATH = AUTORESEARCH_DIR / "marl_surface_baseline.py"
PROMOTED_DIRNAME = "promoted"


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_yaml(path: Path, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_jsonl(path: Path, max_recent: int = 40) -> list[dict]:
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


def tail_lines(path: Path, max_lines: int = 20) -> list[str]:
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8", errors="replace").splitlines()[-max_lines:]


def resolve_provider_model(branch_state: dict, cli_provider: Optional[str], cli_model: Optional[str]) -> tuple[str, str]:
    stored_provider = branch_state.get("provider")
    stored_model = branch_state.get("model")
    if cli_provider:
        provider = "openai" if str(cli_provider).strip().lower() == "codex" else str(cli_provider).strip().lower()
        model = str(cli_model or default_model_for_provider(provider)).strip()
    else:
        provider = infer_provider_from_model(stored_model, fallback="gemini" if not stored_provider else stored_provider)
        model = str(cli_model or stored_model or default_model_for_provider(provider)).strip()
    return provider, model


def call_llm_for_candidates(
    *,
    parent_metrics: Optional[dict],
    parent_patterns: list[str],
    recent_history: list[dict],
    parent_config: dict,
    parent_surface_code: str,
    program_text: str,
    candidates_to_generate: int,
    provider: str,
    model: str,
) -> tuple[list[dict], str]:
    system_prompt = """You are designing candidate multi-agent PPO experiments for a racing environment.
Return JSON only.
Each candidate should make 1-2 focused changes.
Prefer config overrides before code changes.
If you propose a surface code change, output a complete Python file for the surface module.
The editable surface module must define get_policy_variants() and may optionally define build_optimizer(policy, learning_rate).
If code is unchanged, set code_change to "keep_parent" and omit surface_code.
If you change architecture and a parent checkpoint would likely be incompatible, set resume_policy to "scratch".
"""
    user_prompt = f"""Create {candidates_to_generate} candidate experiments.

Research program:
{program_text}

Parent metrics:
{json.dumps(parent_metrics or {}, indent=2)}

Parent patterns:
{json.dumps(parent_patterns)}

Recent history:
{summarize_history(recent_history)}

Allowed config fields:
{flatten_allowlist()}

Current MARL config values for allowed keys:
{json.dumps({key: get_nested(parent_config, key) for key in sorted(MARL_ALLOWLIST) if get_nested(parent_config, key) is not None}, indent=2)}

Current editable surface:
```python
{parent_surface_code}
```

Output JSON:
{{
  "candidates": [
    {{
      "name": "short_snake_case_name",
      "rationale": "one sentence",
      "config_overrides": {{"dotted.key": 0.1}},
      "code_change": "keep_parent" | "replace_surface",
      "surface_variant": "variant_name_when_replacing_surface",
      "resume_policy": "inherit_parent" | "scratch",
      "surface_code": "full python module only when code_change=replace_surface"
    }}
  ]
}}
"""
    raw_text = strip_fences(
        generate_text(
            provider=provider,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            json_mode=True,
        )
    )
    if not raw_text:
        raise RuntimeError("Model returned empty candidate JSON")
    payload = json.loads(raw_text)
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("Candidate JSON missing candidates list")
    return candidates, raw_text


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


def run_stage_experiment(
    *,
    config_path: Path,
    timesteps: int,
    eval_episodes: int,
    seed: int,
    candidate_id: str,
    run_dir: Path,
    timeout: int,
    resume: Optional[Path],
    resume_mode: str,
) -> dict:
    stdout_log_path = run_dir / "stdout.log"
    stderr_log_path = run_dir / "stderr.log"
    cmd = [
        sys.executable,
        "-m",
        "autoresearch.run_marl_experiment",
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
        "--experiment-id",
        candidate_id,
        "--resume-mode",
        resume_mode,
        "--timeout",
        str(timeout),
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
                sys.stderr.write(f"  [child-stderr] {line}")
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
            payload = json.loads(candidate)
            payload["return_code"] = return_code
            return payload
        except json.JSONDecodeError:
            continue
    return {"mean_reward": -999.0, "error": "No JSON metrics found in stdout", "return_code": return_code}


def ensure_promoted_artifacts(
    branch_dir: Path,
    checkpoint_path: Optional[Path],
    config_path: Path,
    surface_path: Path,
    metrics: dict,
) -> None:
    promoted_dir = branch_dir / PROMOTED_DIRNAME
    promoted_dir.mkdir(parents=True, exist_ok=True)
    if checkpoint_path is not None and checkpoint_path.exists():
        shutil.copy2(checkpoint_path, promoted_dir / "checkpoint.pt")
    shutil.copy2(config_path, promoted_dir / "config.yaml")
    shutil.copy2(surface_path, promoted_dir / "surface.py")
    with open(promoted_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)


def materialize_branch_base(
    *,
    branch_dir: Path,
    base_config_path: Path,
    base_checkpoint_path: Optional[Path],
) -> tuple[Path, Path, Optional[Path]]:
    bootstrap_dir = branch_dir / "bootstrap"
    bootstrap_dir.mkdir(parents=True, exist_ok=True)
    surface_path = bootstrap_dir / "surface.py"
    config_path = bootstrap_dir / "config.yaml"
    shutil.copy2(BASE_SURFACE_PATH, surface_path)
    cfg = load_yaml(base_config_path)
    training = cfg.setdefault("training", {})
    training["trainer_backend"] = "torch"
    training["torch_policy_variant"] = "marl_surface_baseline"
    training["torch_policy_variant_source"] = str(surface_path.resolve())
    write_yaml(config_path, cfg)
    checkpoint_copy = None
    if base_checkpoint_path is not None:
        checkpoint_copy = bootstrap_dir / base_checkpoint_path.name
        shutil.copy2(base_checkpoint_path, checkpoint_copy)
    return config_path, surface_path, checkpoint_copy


def build_branch_state(
    *,
    branch_dir: Path,
    base_config_path: Path,
    base_checkpoint_path: Optional[Path],
    provider: str,
    model: str,
    generations: int,
    candidates_per_batch: int,
) -> dict:
    branch_config_path, branch_surface_path, branch_checkpoint_path = materialize_branch_base(
        branch_dir=branch_dir,
        base_config_path=base_config_path,
        base_checkpoint_path=base_checkpoint_path,
    )
    return {
        "branch_dir": str(branch_dir),
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "provider": provider,
        "model": model,
        "max_generations": generations,
        "candidates_per_batch": candidates_per_batch,
        "current_generation": 0,
        "validation_ladder": VALIDATION_LADDER,
        "parent": {
            "config_path": str(branch_config_path.resolve()),
            "surface_path": str(branch_surface_path.resolve()),
            "checkpoint_path": str(branch_checkpoint_path.resolve()) if branch_checkpoint_path else None,
            "metrics": None,
            "patterns": [],
        },
    }


def make_baseline_candidate(parent_variant: str) -> dict:
    return {
        "name": "baseline_continuation",
        "rationale": "Continue the current parent through the validation ladder as the control branch.",
        "config_overrides": {},
        "code_change": "keep_parent",
        "resume_policy": "inherit_parent",
        "surface_variant": parent_variant,
        "source": "baseline",
    }


def current_parent_variant(parent_config: dict) -> str:
    return str((parent_config.get("training", {}) or {}).get("torch_policy_variant", "marl_surface_baseline"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Recursive MARL autoresearch loop")
    parser.add_argument("--base-config", type=str, required=True)
    parser.add_argument("--base-checkpoint", type=str, default=None)
    parser.add_argument("--results-subdir", type=str, required=True)
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--candidates-per-batch", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--provider", type=str, default=None, choices=["gemini", "openai", "codex"])
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--program", type=str, default=str(PROGRAM_PATH))
    parser.add_argument("--mode", type=str, default="fully_autonomous", choices=["fully_autonomous", "semi_autonomous"])
    args = parser.parse_args()

    branch_dir = RESULTS_ROOT / args.results_subdir
    branch_dir.mkdir(parents=True, exist_ok=True)
    branch_state_path = branch_dir / "branch_state.json"
    generations_log = branch_dir / "generations.jsonl"
    program_text = read_text(Path(args.program).resolve())

    base_config_path = Path(args.base_config).resolve()
    if not base_config_path.is_file():
        raise FileNotFoundError(f"Base config not found: {base_config_path}")
    base_checkpoint_path = Path(args.base_checkpoint).resolve() if args.base_checkpoint else None
    if base_checkpoint_path is not None and not base_checkpoint_path.is_file():
        raise FileNotFoundError(f"Base checkpoint not found: {base_checkpoint_path}")

    if branch_state_path.exists():
        branch_state = json.loads(branch_state_path.read_text(encoding="utf-8"))
        log(f"[run_marl_recursive] Resuming branch: {branch_dir}")
    else:
        if args.provider:
            provider = "openai" if str(args.provider).strip().lower() == "codex" else str(args.provider).strip().lower()
        else:
            provider = infer_provider_from_model(args.model, fallback="gemini")
        model = str(args.model or default_model_for_provider(provider)).strip()
        branch_state = build_branch_state(
            branch_dir=branch_dir,
            base_config_path=base_config_path,
            base_checkpoint_path=base_checkpoint_path,
            provider=provider,
            model=model,
            generations=args.generations,
            candidates_per_batch=args.candidates_per_batch,
        )
        branch_state_path.write_text(json.dumps(branch_state, indent=2), encoding="utf-8")

    provider, model = resolve_provider_model(branch_state, args.provider, args.model)
    branch_state["provider"] = provider
    branch_state["model"] = model

    recent_entries = load_jsonl(generations_log, max_recent=20)
    recent_history: list[dict] = []
    for entry in recent_entries:
        recent_history.extend(entry.get("records", []))
    recent_history = recent_history[-30:]

    for generation_index in range(int(branch_state.get("current_generation", 0)) + 1, args.generations + 1):
        parent = branch_state["parent"]
        parent_config_path = Path(parent["config_path"])
        parent_surface_path = Path(parent["surface_path"])
        parent_checkpoint_path = Path(parent["checkpoint_path"]) if parent.get("checkpoint_path") else None
        parent_config = load_yaml(parent_config_path)
        parent_surface_code = read_text(parent_surface_path)
        parent_metrics = parent.get("metrics")
        parent_patterns = parent.get("patterns", [])

        gen_dir = branch_dir / f"generation_{generation_index:03d}"
        gen_dir.mkdir(parents=True, exist_ok=True)
        log("=" * 72)
        log(f"[run_marl_recursive] Generation {generation_index}/{args.generations}")
        log(f"[run_marl_recursive] Parent checkpoint: {parent_checkpoint_path if parent_checkpoint_path else 'none'}")
        log(f"[run_marl_recursive] Provider/model: {provider}/{model}")

        candidates: list[dict] = [make_baseline_candidate(current_parent_variant(parent_config))]
        needed = max(0, args.candidates_per_batch - 1)
        if needed > 0:
            try:
                llm_candidates, llm_raw = call_llm_for_candidates(
                    parent_metrics=parent_metrics,
                    parent_patterns=parent_patterns,
                    recent_history=recent_history,
                    parent_config=parent_config,
                    parent_surface_code=parent_surface_code,
                    program_text=program_text,
                    candidates_to_generate=needed,
                    provider=provider,
                    model=model,
                )
                (gen_dir / "llm_candidates_raw.json").write_text(llm_raw + "\n", encoding="utf-8")
                candidates.extend(llm_candidates[:needed])
            except (LlmProviderError, Exception) as exc:
                log(f"[run_marl_recursive] Candidate generation failed: {exc}")

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
                continue

            candidate_config = apply_overrides(parent_config, cleaned_overrides)
            code_change = str(candidate.get("code_change", "keep_parent")).strip().lower()
            resume_policy = str(candidate.get("resume_policy", "inherit_parent")).strip().lower()
            candidate_surface_variant = str(
                candidate.get("surface_variant", current_parent_variant(parent_config))
            ).strip()
            candidate_surface_code = parent_surface_code
            if code_change == "replace_surface":
                candidate_surface_code = candidate.get("surface_code", "") or ""
                try:
                    compile(candidate_surface_code, "candidate_surface.py", "exec")
                except SyntaxError as exc:
                    generation_records.append(
                        {
                            "generation": generation_index,
                            "candidate_index": idx,
                            "candidate_name": candidate_name,
                            "error": f"invalid_surface_code: {exc}",
                            "promoted": False,
                            "passed_gate": False,
                        }
                    )
                    continue
            elif code_change != "keep_parent":
                generation_records.append(
                    {
                        "generation": generation_index,
                        "candidate_index": idx,
                        "candidate_name": candidate_name,
                        "error": f"unknown_code_change:{code_change}",
                        "promoted": False,
                        "passed_gate": False,
                    }
                )
                continue

            candidate_surface_path = candidate_dir / "candidate_surface.py"
            candidate_surface_path.write_text(candidate_surface_code, encoding="utf-8")
            candidate_config.setdefault("training", {})
            candidate_config["training"]["trainer_backend"] = "torch"
            candidate_config["training"]["torch_policy_variant_source"] = str(candidate_surface_path.resolve())
            candidate_config["training"]["torch_policy_variant"] = candidate_surface_variant
            candidate_config_path = candidate_dir / "candidate_config.yaml"
            write_yaml(candidate_config_path, candidate_config)
            override_summary = render_override_summary(parent_config, cleaned_overrides)
            code_diff = render_code_diff(parent_surface_code, candidate_surface_code)
            write_human_review(
                candidate_dir / "human_review_pre_run.txt",
                title=f"Candidate {candidate_name} (pre-run)",
                rationale=str(candidate.get("rationale", "")),
                override_summary=override_summary,
                code_diff=code_diff,
            )

            stage_records: list[dict] = []
            stage_resume = parent_checkpoint_path if (resume_policy == "inherit_parent" and parent_checkpoint_path) else None
            stage_passed = True
            gate_reasons: list[str] = []
            for stage_index, stage in enumerate(VALIDATION_LADDER, start=1):
                stage_dir = candidate_dir / stage["name"]
                stage_dir.mkdir(parents=True, exist_ok=True)
                log(
                    f"[run_marl_recursive] Running {candidate_name} | stage={stage['name']} | "
                    f"timesteps={stage['timesteps']:,} | resume={'yes' if stage_resume else 'no'}"
                )
                metrics = run_stage_experiment(
                    config_path=candidate_config_path,
                    timesteps=int(stage["timesteps"]),
                    eval_episodes=int(stage["eval_episodes"]),
                    seed=args.seed + generation_index * 100 + idx * 10 + stage_index,
                    candidate_id=f"g{generation_index:03d}_c{idx:02d}_{stage['name']}",
                    run_dir=stage_dir,
                    timeout=args.timeout,
                    resume=stage_resume,
                    resume_mode="policy_only" if stage_resume else "full",
                )
                patterns = infer_patterns(metrics)
                passed_gate, stage_gate_reasons, score = gate_candidate(metrics, stage["name"], parent_metrics)
                stage_record = {
                    "stage": stage["name"],
                    "timesteps": stage["timesteps"],
                    "eval_episodes": stage["eval_episodes"],
                    "resume_from": str(stage_resume) if stage_resume else None,
                    "patterns": patterns,
                    "passed_gate": passed_gate,
                    "gate_reasons": stage_gate_reasons,
                    "score": score,
                    **metrics,
                }
                stage_records.append(stage_record)
                if passed_gate and metrics.get("checkpoint_path"):
                    stage_resume = Path(metrics["checkpoint_path"])
                if not passed_gate:
                    stage_passed = False
                    gate_reasons.extend(stage_gate_reasons)
                    break

            final_stage = stage_records[-1] if stage_records else {}
            final_record = {
                "generation": generation_index,
                "candidate_index": idx,
                "candidate_name": candidate_name,
                "rationale": candidate.get("rationale", ""),
                "source": candidate.get("source", "llm"),
                "config_overrides": cleaned_overrides,
                "code_change": code_change,
                "resume_policy": resume_policy,
                "candidate_config_path": str(candidate_config_path),
                "candidate_surface_path": str(candidate_surface_path),
                "checkpoint_path": final_stage.get("checkpoint_path"),
                "passed_gate": stage_passed,
                "gate_reasons": gate_reasons or final_stage.get("gate_reasons", []),
                "patterns": final_stage.get("patterns", []),
                "deepest_stage": final_stage.get("stage"),
                "score": final_stage.get("score", -1e9),
                "stage_records": stage_records,
                "stderr_tail": tail_lines((candidate_dir / (final_stage.get("stage") or "") / "stderr.log"), max_lines=20)
                if final_stage.get("stage")
                else [],
                "stdout_tail": tail_lines((candidate_dir / (final_stage.get("stage") or "") / "stdout.log"), max_lines=10)
                if final_stage.get("stage")
                else [],
                "promoted": False,
                **{k: v for k, v in final_stage.items() if k not in {"patterns", "gate_reasons"}},
            }
            with open(candidate_dir / "metrics.json", "w", encoding="utf-8") as handle:
                json.dump(final_record, handle, indent=2)
            write_human_review(
                candidate_dir / "human_review_post_run.txt",
                title=f"Candidate {candidate_name} (post-run)",
                rationale=str(candidate.get("rationale", "")),
                override_summary=override_summary,
                code_diff=code_diff,
                stages=stage_records,
                gate_reasons=final_record["gate_reasons"],
            )
            generation_records.append(final_record)

        passing = [record for record in generation_records if record.get("passed_gate")]
        promoted_record = max(passing, key=lambda item: safe_float(item.get("score"), -1e9)) if passing else None
        if promoted_record is not None:
            promoted_record["promoted"] = True
            promoted_checkpoint = Path(promoted_record["checkpoint_path"]) if promoted_record.get("checkpoint_path") else None
            promoted_config_path = Path(promoted_record["candidate_config_path"])
            promoted_surface_path = Path(promoted_record["candidate_surface_path"])
            parent.update(
                {
                    "config_path": str(promoted_config_path.resolve()),
                    "surface_path": str(promoted_surface_path.resolve()),
                    "checkpoint_path": str(promoted_checkpoint.resolve()) if promoted_checkpoint else None,
                    "metrics": {k: v for k, v in promoted_record.items() if k not in {"stage_records", "stdout_tail", "stderr_tail"}},
                    "patterns": promoted_record.get("patterns", []),
                }
            )
            ensure_promoted_artifacts(
                branch_dir,
                promoted_checkpoint,
                promoted_config_path,
                promoted_surface_path,
                promoted_record,
            )
            log(
                f"[run_marl_recursive] PROMOTED {promoted_record['candidate_name']} | "
                f"stage={promoted_record.get('deepest_stage')} | "
                f"reward={safe_float(promoted_record.get('mean_reward'), -999.0):.2f} | "
                f"progress={safe_float(promoted_record.get('mean_progress'), 0.0):.3f} | "
                f"contact={safe_float(promoted_record.get('contact_rate'), 0.0):.3f}"
            )
        else:
            log("[run_marl_recursive] No candidate passed the validation ladder.")

        generation_summary = {
            "timestamp": datetime.now().isoformat(),
            "generation": generation_index,
            "provider": provider,
            "model": model,
            "promoted_candidate": promoted_record.get("candidate_name") if promoted_record else None,
            "records": generation_records,
        }
        with open(generations_log, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(generation_summary) + "\n")

        summary_lines = [
            f"Generation {generation_index}",
            f"Promoted candidate: {promoted_record.get('candidate_name') if promoted_record else 'none'}",
            "",
        ]
        for record in generation_records:
            summary_lines.append(
                f"- {record.get('candidate_name')}: stage={record.get('deepest_stage')}, "
                f"reward={safe_float(record.get('mean_reward'), -999.0):.2f}, "
                f"progress={safe_float(record.get('mean_progress'), 0.0):.3f}, "
                f"offtrack={safe_float(record.get('offtrack_rate'), 1.0):.3f}, "
                f"contact={safe_float(record.get('contact_rate'), 0.0):.3f}, "
                f"hook={safe_float(record.get('hook_contact_rate'), 0.0):.3f}, "
                f"overtakes={safe_float(record.get('mean_overtakes'), 0.0):.2f}, "
                f"score={safe_float(record.get('score'), -1e9):.2f}, "
                f"promoted={bool(record.get('promoted', False))}, "
                f"gate={record.get('gate_reasons', [])}"
            )
        (gen_dir / "generation_review.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

        branch_state["current_generation"] = generation_index
        branch_state["updated_at"] = datetime.now().isoformat()
        branch_state["parent"] = parent
        branch_state_path.write_text(json.dumps(branch_state, indent=2), encoding="utf-8")

        recent_history.extend(generation_records)
        recent_history = recent_history[-30:]

        if args.mode == "semi_autonomous":
            log("[run_marl_recursive] Semi-autonomous mode: stopping after one generation.")
            break

    log("=" * 72)
    log("[run_marl_recursive] COMPLETE")
    log(f"[run_marl_recursive] branch_dir={branch_dir}")
    log(f"[run_marl_recursive] promoted_dir={branch_dir / PROMOTED_DIRNAME}")


if __name__ == "__main__":
    main()
