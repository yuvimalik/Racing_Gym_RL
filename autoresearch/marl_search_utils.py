"""
Shared MARL autoresearch helpers: allowlist, config merge/score/gates, surface checkpoint checks.
Used by run_marl_recursive.py and marl_search_loop.py.
"""

from __future__ import annotations

import importlib.util
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

MARL_ALLOWLIST = {
    "ppo.learning_rate": {"type": "float", "min": 1e-5, "max": 1e-3},
    "ppo.n_steps": {"type": "int", "min": 128, "max": 4096},
    "ppo.batch_size": {"type": "int", "min": 128, "max": 4096},
    "ppo.n_epochs": {"type": "int", "min": 1, "max": 8},
    "ppo.gae_lambda": {"type": "float", "min": 0.85, "max": 0.99},
    "ppo.ent_coef": {"type": "float", "min": 0.0, "max": 0.2},
    "ppo.clip_range": {"type": "float", "min": 0.05, "max": 0.4},
    "ppo.min_log_std": {"type": "float", "min": -3.0, "max": 0.0},
    "ppo.max_log_std": {"type": "float", "min": -1.0, "max": 1.0},
    "ppo.steer_min_log_std": {"type": "float", "min": -3.0, "max": 0.5},
    "ppo.steer_max_log_std": {"type": "float", "min": -1.0, "max": 0.5},
    "training.success_gate.mean_reward_threshold": {"type": "float", "min": -200.0, "max": 500.0},
    "training.success_gate.mean_progress_threshold": {"type": "float", "min": 0.05, "max": 0.95},
    "training.success_gate.max_offtrack_rate": {"type": "float", "min": 0.05, "max": 1.0},
    "training.fail_fast.min_mean_progress": {"type": "float", "min": 0.0, "max": 0.5},
    "training.fail_fast.min_mean_speed": {"type": "float", "min": 0.0, "max": 20.0},
    "reward_shaping.sharp_turn_threshold": {"type": "float", "min": 0.15, "max": 0.8},
    "reward_shaping.sharp_turn_lookahead": {"type": "int", "min": 2, "max": 16},
    "reward_shaping.corner_target_speed": {"type": "float", "min": 2.0, "max": 18.0},
    "reward_shaping.corner_overspeed_penalty_scale": {"type": "float", "min": 0.0, "max": 1.5},
    "reward_shaping.apex_decel_reward_scale": {"type": "float", "min": 0.0, "max": 1.5},
    "reward_shaping.apex_decel_reward_cap": {"type": "float", "min": 0.0, "max": 8.0},
    "reward_shaping.steer_smoothness_penalty": {"type": "float", "min": 0.0, "max": 0.2},
    "reward_shaping.steer_delta_cap": {"type": "float", "min": 0.1, "max": 1.0},
    "reward_shaping.steer_magnitude_penalty": {"type": "float", "min": 0.0, "max": 0.1},
    "reward_shaping.lateral_velocity_penalty": {"type": "float", "min": 0.0, "max": 2.0},
    "reward_shaping.time_penalty": {"type": "float", "min": -1.0, "max": 0.0},
    "reward_shaping.idle_penalty": {"type": "float", "min": -2.0, "max": 0.0},
    "reward_shaping.brake_penalty_scale": {"type": "float", "min": 0.0, "max": 0.2},
    "reward_shaping.stuck_max_steps": {"type": "int", "min": 50, "max": 500},
    "reward_shaping.no_progress_max_steps": {"type": "int", "min": 100, "max": 1500},
    "reward_shaping.no_progress_terminal_penalty": {"type": "float", "min": -50.0, "max": -1.0},
    "reward_shaping.yaw_rate_penalty": {"type": "float", "min": 0.0, "max": 1.0},
    "reward_shaping.off_track_mode": {"type": "enum", "choices": ["penalty", "terminate"]},
    "reward_shaping.off_track_terminal_penalty": {"type": "float", "min": -150.0, "max": -1.0},
    "reward_shaping.off_track_step_penalty": {"type": "float", "min": -5.0, "max": 0.0},
    "reward_shaping.multi_agent.rank_reward_scale": {"type": "float", "min": 0.0, "max": 0.5},
    "reward_shaping.multi_agent.relative_velocity_scale": {"type": "float", "min": 0.0, "max": 0.2},
    "reward_shaping.multi_agent.relative_velocity_cap": {"type": "float", "min": 0.5, "max": 12.0},
    "reward_shaping.multi_agent.nearest_opponent_max_distance": {"type": "float", "min": 2.0, "max": 20.0},
    "reward_shaping.multi_agent.overtake_bonus": {"type": "float", "min": 0.0, "max": 1.0},
    "reward_shaping.multi_agent.overtake_margin": {"type": "float", "min": 0.0, "max": 0.05},
    "reward_shaping.multi_agent.collision_distance_threshold": {"type": "float", "min": 1.0, "max": 6.0},
    "reward_shaping.multi_agent.collision_overlap_distance": {"type": "float", "min": 1.0, "max": 4.0},
    "reward_shaping.multi_agent.collision_min_closing_speed": {"type": "float", "min": 0.0, "max": 4.0},
    "reward_shaping.multi_agent.collision_low_penalty": {"type": "float", "min": -10.0, "max": 0.0},
    "reward_shaping.multi_agent.collision_medium_penalty": {"type": "float", "min": -20.0, "max": 0.0},
    "reward_shaping.multi_agent.collision_high_penalty": {"type": "float", "min": -40.0, "max": 0.0},
    "reward_shaping.multi_agent.contact_penalty": {"type": "float", "min": -5.0, "max": 0.0},
    "reward_shaping.multi_agent.sustained_contact_steps": {"type": "int", "min": 1, "max": 8},
    "reward_shaping.multi_agent.sustained_contact_penalty": {"type": "float", "min": -10.0, "max": 0.0},
    "reward_shaping.multi_agent.hook_contact_steps": {"type": "int", "min": 2, "max": 8},
    "reward_shaping.multi_agent.hook_contact_speed_threshold": {"type": "float", "min": 0.5, "max": 5.0},
    "reward_shaping.multi_agent.hook_contact_penalty": {"type": "float", "min": -25.0, "max": 0.0},
    "reward_shaping.multi_agent.contact_termination_mode": {
        "type": "enum",
        "choices": ["none", "ego", "both"],
    },
    "reward_shaping.multi_agent.contact_terminate_steps": {"type": "int", "min": 1, "max": 8},
    "reward_shaping.multi_agent.terminate_on_hook_contact": {"type": "enum", "choices": [True, False]},
    "reward_shaping.multi_agent.severe_contact_speed_threshold": {"type": "float", "min": 1.0, "max": 10.0},
    "reward_shaping.multi_agent.contact_terminal_penalty": {"type": "float", "min": -150.0, "max": -1.0},
    "reward_shaping.multi_agent.shared_collision_penalty": {"type": "float", "min": -20.0, "max": 0.0},
    "safety_governor.speed_cap_ratio": {"type": "float", "min": 0.1, "max": 1.0},
    "safety_governor.speed_cap_top_speed": {"type": "float", "min": 10.0, "max": 90.0},
    "safety_governor.speed_cap_brake": {"type": "float", "min": 0.0, "max": 1.0},
}

VALIDATION_LADDER = [
    {
        "name": "smoke_control",
        "timesteps": 100_000,
        "eval_episodes": 2,
        "score_weights": {
            "reward": 1.0,
            "progress": 320.0,
            "offtrack": -240.0,
            "contact": -40.0,
            "hook_contact": -80.0,
            "contact_termination": -80.0,
            "speed": 3.0,
            "overtakes": 1.0,
        },
    },
    {
        "name": "smoke_balanced_racing",
        "timesteps": 200_000,
        "eval_episodes": 3,
        "score_weights": {
            "reward": 1.0,
            "progress": 360.0,
            "offtrack": -220.0,
            "contact": -110.0,
            "hook_contact": -180.0,
            "contact_termination": -140.0,
            "speed": 4.0,
            "overtakes": 6.0,
        },
    },
    {
        "name": "long_confirm",
        "timesteps": 400_000,
        "eval_episodes": 5,
        "score_weights": {
            "reward": 1.0,
            "progress": 420.0,
            "offtrack": -250.0,
            "contact": -150.0,
            "hook_contact": -240.0,
            "contact_termination": -180.0,
            "speed": 4.0,
            "overtakes": 8.0,
        },
    },
]


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def set_nested(config: dict, dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    current = config
    for part in parts[:-1]:
        if part not in current or not isinstance(current[part], dict):
            current[part] = {}
        current = current[part]
    current[parts[-1]] = value


def get_nested(config: dict, dotted_key: str) -> Any:
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


def render_override_summary(parent_config: dict, overrides: dict) -> str:
    if not overrides:
        return "No config overrides."
    lines = []
    for key in sorted(overrides):
        lines.append(f"- {key}: {get_nested(parent_config, key)!r} -> {overrides[key]!r}")
    return "\n".join(lines)


def render_code_diff(parent_code: str, candidate_code: str, max_lines: int = 180) -> str:
    import difflib

    if parent_code == candidate_code:
        return "No surface code changes."
    diff_lines = list(
        difflib.unified_diff(
            parent_code.splitlines(),
            candidate_code.splitlines(),
            fromfile="parent_surface.py",
            tofile="candidate_surface.py",
            lineterm="",
        )
    )
    if len(diff_lines) > max_lines:
        omitted = len(diff_lines) - max_lines
        diff_lines = diff_lines[:max_lines] + [f"... ({omitted} more diff lines omitted)"]
    return "\n".join(diff_lines)


def strip_fences(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```json"):
        text = text[len("```json") :].strip()
    elif text.startswith("```python"):
        text = text[len("```python") :].strip()
    elif text.startswith("```"):
        text = text[3:].strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    return text


def flatten_allowlist() -> str:
    lines = []
    for key, spec in sorted(MARL_ALLOWLIST.items()):
        if spec["type"] == "enum":
            lines.append(f"- {key}: choices={spec['choices']}")
        else:
            lines.append(f"- {key}: {spec['type']} in [{spec['min']}, {spec['max']}]")
    return "\n".join(lines)


def infer_patterns(metrics: dict) -> list[str]:
    patterns: list[str] = []
    progress = safe_float(metrics.get("mean_progress"), 0.0)
    offtrack = safe_float(metrics.get("offtrack_rate"), 0.0)
    speed = safe_float(metrics.get("mean_speed"), 0.0)
    steer_var = safe_float(metrics.get("mean_steer_variance"), 0.0)
    throttle = safe_float(metrics.get("mean_throttle"), 0.0)
    brake = safe_float(metrics.get("mean_brake"), 0.0)
    contact = safe_float(metrics.get("contact_rate"), 0.0)
    hook = safe_float(metrics.get("hook_contact_rate"), 0.0)
    overtakes = safe_float(metrics.get("mean_overtakes"), 0.0)

    if progress < 0.10 and offtrack >= 0.8:
        patterns.append("control_collapse")
    if throttle >= 0.95 and brake <= 0.03 and progress < 0.2:
        patterns.append("throttle_without_brake")
    if steer_var <= 0.005 and offtrack >= 0.8:
        patterns.append("dead_steering")
    if contact >= 0.4:
        patterns.append("contact_heavy")
    if hook >= 0.15:
        patterns.append("hook_contact")
    if overtakes > 0 and contact > 0.3 and progress < 0.2:
        patterns.append("chaotic_overtakes")
    if speed < 4.0 and progress < 0.1:
        patterns.append("speed_collapse")
    return patterns


def score_candidate(metrics: dict, stage_name: str, parent_metrics: Optional[dict]) -> float:
    stage_cfg = next(stage for stage in VALIDATION_LADDER if stage["name"] == stage_name)
    w = stage_cfg["score_weights"]
    score = 0.0
    score += w["reward"] * safe_float(metrics.get("mean_reward"), -999.0)
    score += w["progress"] * safe_float(metrics.get("mean_progress"), 0.0)
    score += w["offtrack"] * safe_float(metrics.get("offtrack_rate"), 1.0)
    score += w["contact"] * safe_float(metrics.get("contact_rate"), 0.0)
    score += w["hook_contact"] * safe_float(metrics.get("hook_contact_rate"), 0.0)
    score += w["contact_termination"] * safe_float(metrics.get("contact_termination_rate"), 0.0)
    score += w["speed"] * safe_float(metrics.get("mean_speed"), 0.0)
    score += w["overtakes"] * safe_float(metrics.get("mean_overtakes"), 0.0)
    if parent_metrics:
        score += 120.0 * (
            safe_float(metrics.get("mean_progress"), 0.0) - safe_float(parent_metrics.get("mean_progress"), 0.0)
        )
        score += 80.0 * (
            safe_float(parent_metrics.get("offtrack_rate"), 1.0) - safe_float(metrics.get("offtrack_rate"), 1.0)
        )
        score += 50.0 * (
            safe_float(parent_metrics.get("contact_rate"), 0.0) - safe_float(metrics.get("contact_rate"), 0.0)
        )
        score += 80.0 * (
            safe_float(parent_metrics.get("hook_contact_rate"), 0.0)
            - safe_float(metrics.get("hook_contact_rate"), 0.0)
        )
    return float(score)


def gate_candidate(metrics: dict, stage_name: str, parent_metrics: Optional[dict]) -> tuple[bool, list[str], float]:
    reasons: list[str] = []
    reward = safe_float(metrics.get("mean_reward"), -999.0)
    progress = safe_float(metrics.get("mean_progress"), 0.0)
    offtrack = safe_float(metrics.get("offtrack_rate"), 1.0)
    steer_var = safe_float(metrics.get("mean_steer_variance"), 0.0)
    throttle = safe_float(metrics.get("mean_throttle"), 0.0)
    brake = safe_float(metrics.get("mean_brake"), 0.0)
    contact = safe_float(metrics.get("contact_rate"), 0.0)
    hook = safe_float(metrics.get("hook_contact_rate"), 0.0)
    termination = safe_float(metrics.get("contact_termination_rate"), 0.0)

    if reward <= -900:
        reasons.append("experiment_failed")
    if progress < 0.03:
        reasons.append("progress_too_low")
    if offtrack > 0.98:
        reasons.append("offtrack_near_total")
    if steer_var < 0.003 and progress < 0.15:
        reasons.append("steering_variance_dead")
    if throttle > 0.97 and brake < 0.03 and progress < 0.20:
        reasons.append("throttle_saturated_without_brake")

    if stage_name == "smoke_control":
        if offtrack > 0.85:
            reasons.append("smoke_control_offtrack")
    elif stage_name == "smoke_balanced_racing":
        if contact > 0.45:
            reasons.append("contact_too_high")
        if hook > 0.20:
            reasons.append("hook_contact_too_high")
    elif stage_name == "long_confirm":
        if contact > 0.35:
            reasons.append("long_confirm_contact_too_high")
        if hook > 0.12:
            reasons.append("long_confirm_hook_contact_too_high")
        if termination > 0.20:
            reasons.append("contact_termination_too_high")

    if parent_metrics:
        parent_progress = safe_float(parent_metrics.get("mean_progress"), 0.0)
        parent_offtrack = safe_float(parent_metrics.get("offtrack_rate"), 1.0)
        parent_contact = safe_float(parent_metrics.get("contact_rate"), 0.0)
        parent_hook = safe_float(parent_metrics.get("hook_contact_rate"), 0.0)
        if progress < max(0.05, parent_progress - 0.05):
            reasons.append("worse_progress_than_parent")
        if offtrack > min(1.0, parent_offtrack + 0.15):
            reasons.append("worse_offtrack_than_parent")
        if contact > min(1.0, parent_contact + 0.12):
            reasons.append("worse_contact_than_parent")
        if hook > min(1.0, parent_hook + 0.08):
            reasons.append("worse_hook_contact_than_parent")

    score = score_candidate(metrics, stage_name, parent_metrics)
    return len(reasons) == 0, reasons, score


def validate_overrides(overrides: dict) -> tuple[bool, list[str], dict]:
    errors: list[str] = []
    cleaned: dict = {}
    for key, value in (overrides or {}).items():
        spec = MARL_ALLOWLIST.get(key)
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


def summarize_history(history: list[dict], max_items: int = 8) -> str:
    if not history:
        return "No prior MARL generations."
    lines = []
    for item in history[-max_items:]:
        lines.append(
            f"- {item.get('candidate_name', 'unknown')}: stage={item.get('deepest_stage')}, "
            f"reward={safe_float(item.get('mean_reward'), -999.0):.2f}, "
            f"progress={safe_float(item.get('mean_progress'), 0.0):.3f}, "
            f"offtrack={safe_float(item.get('offtrack_rate'), 1.0):.3f}, "
            f"contact={safe_float(item.get('contact_rate'), 0.0):.3f}, "
            f"hook={safe_float(item.get('hook_contact_rate'), 0.0):.3f}, "
            f"overtakes={safe_float(item.get('mean_overtakes'), 0.0):.2f}, "
            f"promoted={bool(item.get('promoted', False))}, "
            f"gate={item.get('gate_reasons', [])}"
        )
    return "\n".join(lines)


def write_human_review(
    path: Path,
    *,
    title: str,
    rationale: str,
    override_summary: str,
    code_diff: str,
    stages: Optional[list[dict]] = None,
    gate_reasons: Optional[list[str]] = None,
) -> None:
    lines = [
        title,
        "=" * len(title),
        "",
        f"Rationale: {rationale or 'n/a'}",
        "",
        "Config changes:",
        override_summary,
        "",
        "Surface changes:",
        code_diff,
    ]
    if stages:
        lines.extend(["", "Stage results:"])
        for stage in stages:
            lines.append(
                f"- {stage.get('stage')}: reward={safe_float(stage.get('mean_reward'), -999.0):.2f}, "
                f"progress={safe_float(stage.get('mean_progress'), 0.0):.3f}, "
                f"offtrack={safe_float(stage.get('offtrack_rate'), 1.0):.3f}, "
                f"contact={safe_float(stage.get('contact_rate'), 0.0):.3f}, "
                f"hook={safe_float(stage.get('hook_contact_rate'), 0.0):.3f}, "
                f"overtakes={safe_float(stage.get('mean_overtakes'), 0.0):.2f}, "
                f"score={safe_float(stage.get('score'), -1e9):.2f}"
            )
    if gate_reasons:
        lines.extend(["", f"Gate reasons: {gate_reasons}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _infer_obs_hw_from_baseline_flatten(n_flat: int) -> Optional[tuple[int, int]]:
    """MarlAutoresearchBaseline-style CNN: 3 conv layers then 64*h3*w3 flatten."""
    for h in range(32, 256):
        w = h
        h1 = (h - 8) // 4 + 1
        w1 = (w - 8) // 4 + 1
        h2 = (h1 - 4) // 2 + 1
        w2 = (w1 - 4) // 2 + 1
        h3 = (h2 - 3) // 1 + 1
        w3 = (w2 - 3) // 1 + 1
        if 64 * h3 * w3 == n_flat:
            return (h, w)
    return None


def verify_surface_compatible_with_checkpoint(
    checkpoint_path: Path,
    surface_path: Path,
    variant_name: str,
    ppo_cfg: dict,
) -> tuple[bool, str]:
    """
    Return True if policy weights from checkpoint load strictly into the policy
    class exported by the surface module (same tensor shapes).
    """
    try:
        import torch
    except ImportError:
        return False, "torch not installed"

    ckpt_file = Path(checkpoint_path).resolve()
    if not ckpt_file.is_file():
        return False, f"checkpoint not found: {ckpt_file}"

    try:
        try:
            payload = torch.load(str(ckpt_file), map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(str(ckpt_file), map_location="cpu")
    except Exception as exc:
        return False, f"torch.load failed: {exc}"

    sd = payload.get("policy_state_dict")
    if not isinstance(sd, dict):
        return False, "checkpoint missing policy_state_dict"

    spec = importlib.util.spec_from_file_location(
        f"marl_surface_check_{abs(hash(str(surface_path)))}", str(surface_path.resolve())
    )
    if spec is None or spec.loader is None:
        return False, "could not load surface spec"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    get_variants = getattr(module, "get_policy_variants", None)
    if not callable(get_variants):
        return False, "surface missing get_policy_variants()"
    variants = get_variants()
    if not isinstance(variants, dict) or variant_name not in variants:
        return False, f"variant {variant_name!r} not in get_policy_variants()"

    policy_cls = variants[variant_name]
    if "features.0.weight" not in sd or "policy_mlp.0.weight" not in sd:
        return False, "unrecognized policy_state_dict layout"

    c = int(sd["features.0.weight"].shape[1])
    n_flat = int(sd["policy_mlp.0.weight"].shape[1])
    hw = _infer_obs_hw_from_baseline_flatten(n_flat)
    if hw is None:
        return False, f"could not infer H,W from flatten dim {n_flat}"
    h, w = hw
    action_dim = int(sd["policy_mean.bias"].shape[0])

    min_log_std = float(ppo_cfg.get("min_log_std", -1.5))
    max_log_std = float(ppo_cfg.get("max_log_std", 1.0))
    smin = ppo_cfg.get("steer_min_log_std", None)
    smax = ppo_cfg.get("steer_max_log_std", None)
    steer_min = float(smin) if smin is not None else None
    steer_max = float(smax) if smax is not None else None

    model = policy_cls(
        (c, h, w),
        action_dim,
        min_log_std=min_log_std,
        max_log_std=max_log_std,
        steer_min_log_std=steer_min,
        steer_max_log_std=steer_max,
    )
    try:
        model.load_state_dict(sd, strict=True)
    except Exception as exc:
        return False, f"state_dict mismatch: {exc}"
    return True, ""


def ladder_stage_by_name(name: str) -> dict:
    for stage in VALIDATION_LADDER:
        if stage["name"] == name:
            return stage
    raise ValueError(f"Unknown ladder stage: {name}. Expected one of: {[s['name'] for s in VALIDATION_LADDER]}")
