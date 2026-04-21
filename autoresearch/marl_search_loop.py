"""
MARL autoresearch: fixed warm-start checkpoint, LLM-proposed YAML (+ optional compatible surface),
single or two-stage train+eval via run_marl_experiment, promotion bundle for long training.
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

from autoresearch.llm_client import (
    LlmProviderError,
    default_model_for_provider,
    generate_text,
    infer_provider_from_model,
)
from autoresearch.marl_search_utils import (
    MARL_ALLOWLIST,
    apply_overrides,
    flatten_allowlist,
    gate_candidate,
    get_nested,
    infer_patterns,
    ladder_stage_by_name,
    render_code_diff,
    render_override_summary,
    safe_float,
    strip_fences,
    summarize_history,
    validate_overrides,
    verify_surface_compatible_with_checkpoint,
    write_human_review,
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
CANDIDATES_LOG = "candidates.jsonl"


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


def load_jsonl(path: Path, max_recent: int = 60) -> list[dict]:
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


def run_marl_experiment_subprocess(
    *,
    config_path: Path,
    timesteps: int,
    eval_episodes: int,
    seed: int,
    run_dir: Path,
    timeout: int,
    resume: Path,
    resume_mode: str,
    experiment_id: str,
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
        experiment_id,
        "--resume-mode",
        resume_mode,
        "--timeout",
        str(timeout),
        "--resume",
        str(resume.resolve()),
    ]
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
        threads = [
            threading.Thread(
                target=_stream_reader, args=(proc.stdout, "stdout", output_queue, stdout_sink), daemon=True
            ),
            threading.Thread(
                target=_stream_reader, args=(proc.stderr, "stderr", output_queue, stderr_sink), daemon=True
            ),
        ]
        for thread in threads:
            thread.start()

        timed_out = False
        deadline = time.time() + timeout
        stdout_lines: list[str] = []
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
            sys.stderr.write(f"  [child-{stream_name}] {line}")
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


def materialize_bootstrap(
    branch_dir: Path,
    base_config_path: Path,
    surface_source: Path,
) -> tuple[Path, Path]:
    bootstrap = branch_dir / "bootstrap"
    bootstrap.mkdir(parents=True, exist_ok=True)
    cfg_path = bootstrap / "config.yaml"
    surf_path = bootstrap / "surface.py"
    shutil.copy2(surface_source, surf_path)
    cfg = load_yaml(base_config_path)
    training = cfg.setdefault("training", {})
    training["trainer_backend"] = "torch"
    if "torch_policy_variant" not in training or not training.get("torch_policy_variant"):
        training["torch_policy_variant"] = "marl_surface_baseline"
    training["torch_policy_variant_source"] = str(surf_path.resolve())
    write_yaml(cfg_path, cfg)
    return cfg_path, surf_path


def build_initial_state(
    *,
    branch_dir: Path,
    warm_start_ckpt: Path,
    base_config_path: Path,
    bootstrap_config: Path,
    bootstrap_surface: Path,
    provider: str,
    model: str,
    generations: int,
    candidates_per_batch: int,
) -> dict:
    return {
        "version": 1,
        "branch_dir": str(branch_dir.resolve()),
        "warm_start_ckpt": str(warm_start_ckpt.resolve()),
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "provider": provider,
        "model": model,
        "current_generation": 0,
        "max_generations": generations,
        "candidates_per_batch": candidates_per_batch,
        "reference_metrics": None,
        "parent": {
            "config_path": str(bootstrap_config.resolve()),
            "surface_path": str(bootstrap_surface.resolve()),
            "metrics": None,
            "patterns": [],
        },
    }


def current_parent_variant(parent_config: dict) -> str:
    return str((parent_config.get("training", {}) or {}).get("torch_policy_variant", "marl_surface_baseline"))


def make_baseline_candidate() -> dict:
    return {
        "name": "baseline_continuation",
        "hypothesis": "Control: same config as parent, fine-tune from warm start.",
        "rationale": "Baseline for comparison against LLM candidates.",
        "config_overrides": {},
        "code_change": "keep_parent",
        "surface_variant": None,
        "surface_code": None,
        "source": "baseline",
    }


def call_llm_for_search_candidates(
    *,
    parent_metrics: Optional[dict],
    parent_patterns: list[str],
    recent_history: list[dict],
    parent_config: dict,
    parent_surface_code: str,
    program_text: str,
    n_candidates: int,
    provider: str,
    model: str,
    allow_surface: bool,
) -> tuple[list[dict], str]:
    surface_rule = (
        'Set "code_change": "keep_parent" and omit surface_code. '
        if not allow_surface
        else (
            'You may set "code_change": "replace_surface" only if the policy architecture stays compatible '
            "with loading the existing checkpoint (same conv/MLP shapes). Prefer config_overrides.\n"
            'If replacing surface, include full "surface_code" Python module with get_policy_variants().\n'
        )
    )
    system_prompt = f"""You are tuning multi-agent racing PPO to reduce car-to-car contact and wheel-hooking
(reward shaping, safety governor, PPO hyperparameters). Return JSON only.
Each candidate: 1–2 focused changes. Prefer reward_shaping.multi_agent.* and safety_governor.* overrides.
{surface_rule}
"""
    user_prompt = f"""Create {n_candidates} candidate experiments.

Program:
{program_text}

Parent metrics (eval after fine-tune from same warm start in prior generation, or null if first generation):
{json.dumps(parent_metrics or {}, indent=2)}

Patterns:
{json.dumps(parent_patterns)}

Recent runs:
{summarize_history(recent_history)}

Allowed config keys (dot paths):
{flatten_allowlist()}

Current values for allowed keys:
{json.dumps({key: get_nested(parent_config, key) for key in sorted(MARL_ALLOWLIST) if get_nested(parent_config, key) is not None}, indent=2)}

Parent surface (reference only):
```python
{parent_surface_code[:12000]}{"..." if len(parent_surface_code) > 12000 else ""}
```

Output JSON:
{{
  "candidates": [
    {{
      "name": "short_snake_case",
      "hypothesis": "what you are testing",
      "rationale": "one sentence",
      "config_overrides": {{"dotted.key": 0.1}},
      "code_change": "keep_parent" | "replace_surface",
      "surface_variant": "variant_name_when_replacing_surface",
      "surface_code": "full python module only when code_change is replace_surface"
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
        raise RuntimeError("Model returned empty JSON")
    payload = json.loads(raw_text)
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("missing candidates list")
    if not allow_surface:
        for c in candidates:
            c["code_change"] = "keep_parent"
            c["surface_code"] = None
    return candidates, raw_text


def ensure_promoted_bundle(
    branch_dir: Path,
    *,
    checkpoint_path: Optional[Path],
    config_path: Path,
    surface_path: Path,
    metrics: dict,
    warm_start_ckpt: Path,
    timesteps_add_hint: int,
) -> None:
    promoted = branch_dir / PROMOTED_DIRNAME
    promoted.mkdir(parents=True, exist_ok=True)
    if checkpoint_path is not None and checkpoint_path.exists():
        shutil.copy2(checkpoint_path, promoted / "checkpoint.pt")
    shutil.copy2(config_path, promoted / "effective_config.yaml")
    shutil.copy2(surface_path, promoted / "surface.py")
    with open(promoted / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    lines = [
        "# Promoted MARL experiment",
        "",
        f"Warm-start checkpoint used during search: `{warm_start_ckpt}`",
        f"Promoted post-search checkpoint: `promoted/checkpoint.pt`",
        "",
        "## Continue training (example)",
        "",
        "```bash",
        f"python train.py --config promoted/effective_config.yaml --trainer_backend torch \\",
        f"  --resume promoted/checkpoint.pt --resume_mode policy_only --timesteps_add {timesteps_add_hint}",
        "```",
        "",
        "Adjust `--timesteps_add` or use your full `training.total_timesteps` workflow as needed.",
    ]
    (promoted / "PROMOTE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def resolve_provider_model_cli(
    branch_state: dict, cli_provider: Optional[str], cli_model: Optional[str]
) -> tuple[str, str]:
    if cli_provider:
        provider = "openai" if str(cli_provider).strip().lower() == "codex" else str(cli_provider).strip().lower()
        model = str(cli_model or default_model_for_provider(provider)).strip()
    else:
        stored_provider = branch_state.get("provider")
        stored_model = branch_state.get("model")
        provider = infer_provider_from_model(
            cli_model or stored_model, fallback=stored_provider or "openai"
        )
        model = str(cli_model or stored_model or default_model_for_provider(provider)).strip()
    return provider, model


def main() -> None:
    parser = argparse.ArgumentParser(description="MARL autoresearch with fixed warm-start checkpoint")
    parser.add_argument("--warm-start-ckpt", type=str, required=True)
    parser.add_argument("--base-config", type=str, required=True)
    parser.add_argument("--results-subdir", type=str, required=True)
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--candidates-per-batch", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--provider", type=str, default=None, choices=["gemini", "openai", "codex"])
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--program", type=str, default=str(PROGRAM_PATH))
    parser.add_argument("--base-surface", type=str, default=str(BASE_SURFACE_PATH))
    parser.add_argument("--allow-surface", action="store_true", help="Allow LLM replace_surface (checked vs checkpoint)")
    parser.add_argument("--resume-mode", type=str, default="policy_only", choices=["full", "policy_only"])
    parser.add_argument("--screen-stage", type=str, default="smoke_control", help="Ladder name for score/gates")
    parser.add_argument("--screen-timesteps", type=int, default=100_000)
    parser.add_argument("--screen-eval-episodes", type=int, default=2)
    parser.add_argument("--confirm-top", type=int, default=0, help="Run second stage for top K by screen score")
    parser.add_argument("--confirm-stage", type=str, default="smoke_balanced_racing")
    parser.add_argument("--confirm-timesteps", type=int, default=200_000)
    parser.add_argument("--confirm-eval-episodes", type=int, default=3)
    parser.add_argument("--llm-retries", type=int, default=2)
    parser.add_argument("--llm-sleep-seconds", type=float, default=0.0)
    parser.add_argument("--timesteps-add-hint", type=int, default=2_000_000, help="Printed in PROMOTE.md")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="No train/eval; one synthetic candidate to verify artifact layout",
    )
    args = parser.parse_args()

    warm_start = Path(args.warm_start_ckpt).resolve()
    if not warm_start.is_file():
        raise FileNotFoundError(f"Warm-start checkpoint not found: {warm_start}")
    base_config_path = Path(args.base_config).resolve()
    if not base_config_path.is_file():
        raise FileNotFoundError(f"Base config not found: {base_config_path}")
    surface_default = Path(args.base_surface).resolve()
    if not surface_default.is_file():
        raise FileNotFoundError(f"Base surface not found: {surface_default}")

    branch_dir = (RESULTS_ROOT / args.results_subdir).resolve()
    branch_dir.mkdir(parents=True, exist_ok=True)
    state_path = branch_dir / "branch_state.json"
    candidates_log_path = branch_dir / CANDIDATES_LOG
    program_text = read_text(Path(args.program).resolve())

    ladder_stage_by_name(args.screen_stage)
    if args.confirm_top > 0:
        ladder_stage_by_name(args.confirm_stage)

    if state_path.exists():
        branch_state = json.loads(state_path.read_text(encoding="utf-8"))
        log(f"[marl_search_loop] Resuming {branch_dir}")
    else:
        boot_cfg, boot_surf = materialize_bootstrap(branch_dir, base_config_path, surface_default)
        if args.provider:
            provider = "openai" if args.provider == "codex" else args.provider
        else:
            provider = infer_provider_from_model(args.model, fallback="openai")
        model = str(args.model or default_model_for_provider(provider)).strip()
        branch_state = build_initial_state(
            branch_dir=branch_dir,
            warm_start_ckpt=warm_start,
            base_config_path=base_config_path,
            bootstrap_config=boot_cfg,
            bootstrap_surface=boot_surf,
            provider=provider,
            model=model,
            generations=args.generations,
            candidates_per_batch=args.candidates_per_batch,
        )
        state_path.write_text(json.dumps(branch_state, indent=2), encoding="utf-8")

    provider, model = resolve_provider_model_cli(branch_state, args.provider, args.model)
    branch_state["provider"] = provider
    branch_state["model"] = model

    warm_start = Path(branch_state["warm_start_ckpt"]).resolve()

    recent_flat: list[dict] = []
    for row in load_jsonl(candidates_log_path, max_recent=40):
        recent_flat.append(row)

    start_gen = int(branch_state.get("current_generation", 0)) + 1
    end_gen = int(args.generations)
    if start_gen > end_gen:
        log("[marl_search_loop] current_generation already reached --generations; nothing to do.")
        return

    for gen in range(start_gen, end_gen + 1):
        parent = branch_state["parent"]
        parent_config_path = Path(parent["config_path"])
        parent_surface_path = Path(parent["surface_path"])
        parent_config = load_yaml(parent_config_path)
        parent_surface_code = read_text(parent_surface_path)
        parent_metrics = parent.get("metrics")
        parent_patterns = list(parent.get("patterns") or [])

        gen_dir = branch_dir / f"generation_{gen:03d}"
        gen_dir.mkdir(parents=True, exist_ok=True)
        log("=" * 72)
        log(f"[marl_search_loop] Generation {gen}/{end_gen} | warm_start={warm_start.name}")

        candidates: list[dict] = [make_baseline_candidate()]
        need_llm = max(0, args.candidates_per_batch - 1)

        if args.dry_run:
            candidates = [
                make_baseline_candidate(),
                {
                    "name": "dry_run_synthetic",
                    "hypothesis": "dry run",
                    "rationale": "plumbing test",
                    "config_overrides": {},
                    "code_change": "keep_parent",
                    "surface_variant": None,
                    "surface_code": None,
                    "source": "dry_run",
                },
            ]
            need_llm = 0

        if need_llm > 0 and not args.dry_run:
            last_err: Optional[BaseException] = None
            for attempt in range(max(1, args.llm_retries + 1)):
                try:
                    llm_list, raw = call_llm_for_search_candidates(
                        parent_metrics=parent_metrics,
                        parent_patterns=parent_patterns,
                        recent_history=recent_flat[-30:],
                        parent_config=parent_config,
                        parent_surface_code=parent_surface_code,
                        program_text=program_text,
                        n_candidates=need_llm,
                        provider=provider,
                        model=model,
                        allow_surface=args.allow_surface,
                    )
                    (gen_dir / "llm_candidates_raw.json").write_text(raw + "\n", encoding="utf-8")
                    candidates.extend(llm_list[:need_llm])
                    last_err = None
                    break
                except (LlmProviderError, json.JSONDecodeError, ValueError, RuntimeError) as exc:
                    last_err = exc
                    log(f"[marl_search_loop] LLM attempt {attempt + 1} failed: {exc}")
                    time.sleep(1.0 + args.llm_sleep_seconds)
            if last_err is not None:
                log(f"[marl_search_loop] No LLM candidates this generation ({last_err})")

        if args.llm_sleep_seconds > 0:
            time.sleep(args.llm_sleep_seconds)

        screen_records: list[dict] = []

        for idx, cand in enumerate(candidates, start=1):
            name = str(cand.get("name", f"candidate_{idx}")).strip().lower().replace(" ", "_")
            cand_dir = gen_dir / f"{idx:02d}_{name}"
            cand_dir.mkdir(parents=True, exist_ok=True)

            overrides = cand.get("config_overrides") or {}
            ok, o_err, cleaned = validate_overrides(overrides)
            if not ok:
                rec = {
                    "generation": gen,
                    "candidate_index": idx,
                    "candidate_name": name,
                    "error": "invalid_overrides",
                    "override_errors": o_err,
                    "passed_gate": False,
                    "promoted": False,
                }
                screen_records.append(rec)
                with open(cand_dir / "record.json", "w", encoding="utf-8") as f:
                    json.dump(rec, f, indent=2)
                continue

            merged = apply_overrides(parent_config, cleaned)
            code_change = str(cand.get("code_change", "keep_parent")).strip().lower()
            surf_code = parent_surface_code
            variant = cand.get("surface_variant") or current_parent_variant(merged)

            if code_change == "replace_surface" and args.allow_surface:
                surf_code = str(cand.get("surface_code") or "")
                try:
                    compile(surf_code, "candidate_surface.py", "exec")
                except SyntaxError as exc:
                    rec = {
                        "generation": gen,
                        "candidate_index": idx,
                        "candidate_name": name,
                        "error": f"invalid_surface_syntax:{exc}",
                        "passed_gate": False,
                        "promoted": False,
                    }
                    screen_records.append(rec)
                    continue
                surf_path = cand_dir / "candidate_surface.py"
                surf_path.write_text(surf_code, encoding="utf-8")
                ppo_cfg = merged.get("ppo") or {}
                compat, cerr = verify_surface_compatible_with_checkpoint(
                    warm_start, surf_path, str(variant), ppo_cfg
                )
                if not compat:
                    log(f"[marl_search_loop] Surface incompatible for {name}: {cerr}; falling back to parent surface.")
                    surf_code = parent_surface_code
                    shutil.copy2(parent_surface_path, cand_dir / "candidate_surface.py")
                    surf_path = cand_dir / "candidate_surface.py"
                    variant = current_parent_variant(parent_config)
            else:
                shutil.copy2(parent_surface_path, cand_dir / "candidate_surface.py")
                surf_path = cand_dir / "candidate_surface.py"
                variant = current_parent_variant(merged)

            merged.setdefault("training", {})
            merged["training"]["trainer_backend"] = "torch"
            merged["training"]["torch_policy_variant_source"] = str(surf_path.resolve())
            merged["training"]["torch_policy_variant"] = str(variant)

            cfg_out = cand_dir / "candidate_config.yaml"
            write_yaml(cfg_out, merged)

            override_summary = render_override_summary(parent_config, cleaned)
            code_diff = render_code_diff(parent_surface_code, read_text(surf_path))
            write_human_review(
                cand_dir / "human_review_pre_run.txt",
                title=f"Candidate {name} (pre-run)",
                rationale=str(cand.get("rationale", "")),
                override_summary=override_summary,
                code_diff=code_diff,
            )

            stage_dir = cand_dir / "screen"
            stage_dir.mkdir(parents=True, exist_ok=True)

            if args.dry_run:
                metrics = {
                    "mean_reward": 0.0,
                    "mean_progress": 0.2,
                    "offtrack_rate": 0.2,
                    "contact_rate": 0.15,
                    "hook_contact_rate": 0.05,
                    "contact_termination_rate": 0.0,
                    "mean_speed": 10.0,
                    "mean_overtakes": 0.0,
                    "mean_steer_variance": 0.02,
                    "mean_throttle": 0.5,
                    "mean_brake": 0.1,
                    "checkpoint_path": str(warm_start),
                    "dry_run": True,
                }
            else:
                metrics = run_marl_experiment_subprocess(
                    config_path=cfg_out,
                    timesteps=int(args.screen_timesteps),
                    eval_episodes=int(args.screen_eval_episodes),
                    seed=args.seed + gen * 100 + idx,
                    run_dir=stage_dir,
                    timeout=args.timeout,
                    resume=warm_start,
                    resume_mode=args.resume_mode,
                    experiment_id=f"g{gen:03d}_c{idx:02d}_screen",
                )

            patterns = infer_patterns(metrics)
            passed, greasons, sc = gate_candidate(metrics, args.screen_stage, parent_metrics)

            stage_payload = {
                "stage": "screen",
                "screen_substage": args.screen_stage,
                "timesteps": args.screen_timesteps,
                "passed_gate": passed,
                "gate_reasons": greasons,
                "score": sc,
                "patterns": patterns,
                **metrics,
            }

            rec = {
                "generation": gen,
                "candidate_index": idx,
                "candidate_name": name,
                "hypothesis": cand.get("hypothesis", ""),
                "rationale": cand.get("rationale", ""),
                "source": cand.get("source", "llm"),
                "config_overrides": cleaned,
                "candidate_config_path": str(cfg_out.resolve()),
                "candidate_surface_path": str(surf_path.resolve()),
                "stage_records": [stage_payload],
                "passed_gate": passed,
                "gate_reasons": greasons,
                "score": sc,
                "patterns": patterns,
                "mean_reward": metrics.get("mean_reward", -999.0),
                "mean_progress": metrics.get("mean_progress", 0.0),
                "offtrack_rate": metrics.get("offtrack_rate"),
                "contact_rate": metrics.get("contact_rate"),
                "hook_contact_rate": metrics.get("hook_contact_rate"),
                "contact_termination_rate": metrics.get("contact_termination_rate"),
                "mean_speed": metrics.get("mean_speed"),
                "mean_overtakes": metrics.get("mean_overtakes"),
                "checkpoint_path": metrics.get("checkpoint_path"),
                "error": metrics.get("error"),
            }

            write_human_review(
                cand_dir / "human_review_post_screen.txt",
                title=f"Candidate {name} (post-screen)",
                rationale=str(cand.get("rationale", "")),
                override_summary=override_summary,
                code_diff=code_diff,
                stages=[stage_payload],
                gate_reasons=greasons,
            )
            with open(cand_dir / "metrics_screen.json", "w", encoding="utf-8") as f:
                json.dump(rec, f, indent=2)

            screen_records.append(rec)

        # Confirm top-K by screen score among those that passed screen gate
        if args.confirm_top > 0 and not args.dry_run:
            passing = [r for r in screen_records if r.get("passed_gate") and not r.get("error")]
            passing.sort(key=lambda r: safe_float(r.get("score"), -1e9), reverse=True)
            top = passing[: args.confirm_top]
            for rec in top:
                idx = rec["candidate_index"]
                name = rec["candidate_name"]
                cand_dir = gen_dir / f"{idx:02d}_{name}"
                cfg_out = Path(rec["candidate_config_path"])
                screen_ckpt = rec.get("checkpoint_path")
                if not screen_ckpt:
                    continue
                resume_ckpt = Path(screen_ckpt).resolve()
                if not resume_ckpt.is_file():
                    log(f"[marl_search_loop] Skip confirm for {name}: missing screen checkpoint")
                    continue
                confirm_dir = cand_dir / "confirm"
                confirm_dir.mkdir(parents=True, exist_ok=True)
                c_metrics = run_marl_experiment_subprocess(
                    config_path=cfg_out,
                    timesteps=int(args.confirm_timesteps),
                    eval_episodes=int(args.confirm_eval_episodes),
                    seed=args.seed + gen * 100 + idx + 50,
                    run_dir=confirm_dir,
                    timeout=args.timeout,
                    resume=resume_ckpt,
                    resume_mode=args.resume_mode,
                    experiment_id=f"g{gen:03d}_c{idx:02d}_confirm",
                )
                c_patterns = infer_patterns(c_metrics)
                c_passed, c_greasons, c_score = gate_candidate(c_metrics, args.confirm_stage, parent_metrics)
                c_stage = {
                    "stage": "confirm",
                    "confirm_substage": args.confirm_stage,
                    "timesteps": args.confirm_timesteps,
                    "passed_gate": c_passed,
                    "gate_reasons": c_greasons,
                    "score": c_score,
                    "patterns": c_patterns,
                    **c_metrics,
                }
                # attach to matching screen_records entry
                for i, sr in enumerate(screen_records):
                    if sr.get("candidate_index") == idx and sr.get("candidate_name") == name:
                        sr.setdefault("stage_records", []).append(c_stage)
                        sr["passed_gate"] = c_passed
                        sr["gate_reasons"] = c_greasons
                        sr["score"] = c_score
                        sr["patterns"] = c_patterns
                        sr["checkpoint_path"] = c_metrics.get("checkpoint_path")
                        for k, v in c_metrics.items():
                            if k not in sr:
                                sr[k] = v
                        break

        promoted_record = None
        pool = [r for r in screen_records if r.get("passed_gate") and r.get("checkpoint_path")]
        if pool:
            promoted_record = max(pool, key=lambda r: safe_float(r.get("score"), -1e9))

        if promoted_record:
            promoted_record["promoted"] = True
            p_cfg = Path(promoted_record["candidate_config_path"])
            p_surf = Path(promoted_record["candidate_surface_path"])
            p_ckpt = Path(promoted_record["checkpoint_path"]) if promoted_record.get("checkpoint_path") else None
            parent.update(
                {
                    "config_path": str(p_cfg.resolve()),
                    "surface_path": str(p_surf.resolve()),
                    "metrics": {k: v for k, v in promoted_record.items() if isinstance(k, str) and k not in {"stage_records"}},
                    "patterns": promoted_record.get("patterns", []),
                }
            )
            ensure_promoted_bundle(
                branch_dir,
                checkpoint_path=p_ckpt,
                config_path=p_cfg,
                surface_path=p_surf,
                metrics={k: v for k, v in promoted_record.items() if k != "stage_records"},
                warm_start_ckpt=warm_start,
                timesteps_add_hint=args.timesteps_add_hint,
            )
            log(
                f"[marl_search_loop] PROMOTED {promoted_record['candidate_name']} | "
                f"score={safe_float(promoted_record.get('score'), -1e9):.2f}"
            )
        else:
            log("[marl_search_loop] No candidate passed gates this generation.")

        branch_state["current_generation"] = gen
        branch_state["updated_at"] = datetime.now().isoformat()
        branch_state["parent"] = parent
        state_path.write_text(json.dumps(branch_state, indent=2), encoding="utf-8")

        summary_lines = [
            f"Generation {gen}",
            f"Promoted: {promoted_record.get('candidate_name') if promoted_record else 'none'}",
            "",
        ]
        for r in screen_records:
            summary_lines.append(
                f"- {r.get('candidate_name')}: score={safe_float(r.get('score'), -1e9):.2f}, "
                f"gate={r.get('gate_reasons', [])}, err={r.get('error')}"
            )
        (gen_dir / "generation_review.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

        for r in screen_records:
            with open(candidates_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"timestamp": datetime.now().isoformat(), **r}) + "\n")
            recent_flat.append(r)

    log("=" * 72)
    log(f"[marl_search_loop] Done. branch_dir={branch_dir} promoted={branch_dir / PROMOTED_DIRNAME}")


if __name__ == "__main__":
    main()
