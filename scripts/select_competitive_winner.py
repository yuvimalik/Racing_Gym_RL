#!/usr/bin/env python3
import json
from pathlib import Path

BASE_CONFIGS = {
    "pace": "config/prime_marl_2car_compete_pace.yaml",
    "overtake": "config/prime_marl_2car_compete_overtake.yaml",
    "combined": "config/prime_marl_2car_compete_combined.yaml",
}


def load_json(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def metric(d: dict, key: str, default: float = 0.0) -> float:
    return float(d.get(key, default))


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    seed = 42
    baseline_path = Path(
        f"{repo}/artifacts_prime_intellect_pull/visualization_budget_fast/best_seed{seed}_eval.json"
    )

    variant_eval_paths = {
        "pace": repo
        / f"artifacts/prime_marl_2car_compete_pace/results/calibration/calibration_eval_seed{seed}.json",
        "overtake": repo
        / f"artifacts/prime_marl_2car_compete_overtake/results/calibration/calibration_eval_seed{seed}.json",
        "combined": repo
        / f"artifacts/prime_marl_2car_compete_combined/results/calibration/calibration_eval_seed{seed}.json",
    }

    for path in [baseline_path, *variant_eval_paths.values()]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required metrics file: {path}")

    baseline = load_json(baseline_path)
    scored = []

    for name, path in variant_eval_paths.items():
        cand = load_json(path)

        safety_pass = (
            metric(cand, "contact_rate") <= metric(baseline, "contact_rate") + 0.01
            and metric(cand, "hook_contact_rate") <= metric(baseline, "hook_contact_rate") + 0.01
            and metric(cand, "contact_termination_rate")
            <= metric(baseline, "contact_termination_rate") + 0.01
            and metric(cand, "offtrack_rate") <= metric(baseline, "offtrack_rate") + 0.02
        )

        competitiveness_pass = (
            metric(cand, "mean_overtakes") > metric(baseline, "mean_overtakes")
            and metric(cand, "mean_speed_std") > metric(baseline, "mean_speed_std")
            and metric(cand, "mean_progress") >= metric(baseline, "mean_progress") - 0.02
            and metric(cand, "mean_reward") >= metric(baseline, "mean_reward") - 5.0
        )

        if safety_pass and competitiveness_pass:
            score = (
                (metric(cand, "mean_overtakes") - metric(baseline, "mean_overtakes")) * 2.0
                + (metric(cand, "mean_speed_std") - metric(baseline, "mean_speed_std")) * 10.0
                + (metric(cand, "mean_reward") - metric(baseline, "mean_reward")) * 0.05
                - metric(cand, "contact_rate") * 100.0
                - metric(cand, "contact_termination_rate") * 200.0
            )
            scored.append(
                {
                    "variant": name,
                    "score": score,
                    "metrics_path": str(path),
                    "base_config": BASE_CONFIGS[name],
                    "metrics": {
                        "mean_overtakes": metric(cand, "mean_overtakes"),
                        "mean_speed_std": metric(cand, "mean_speed_std"),
                        "mean_reward": metric(cand, "mean_reward"),
                        "mean_progress": metric(cand, "mean_progress"),
                        "offtrack_rate": metric(cand, "offtrack_rate"),
                        "contact_rate": metric(cand, "contact_rate"),
                        "hook_contact_rate": metric(cand, "hook_contact_rate"),
                        "contact_termination_rate": metric(cand, "contact_termination_rate"),
                    },
                }
            )

    if not scored:
        raise RuntimeError("No competitive variant passed both safety and competitiveness gates.")

    scored.sort(key=lambda x: x["score"], reverse=True)
    winner = scored[0]

    out_dir = repo / "artifacts/competitive_tuning"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "winner.json"
    with out_path.open("w") as f:
        json.dump({"baseline": str(baseline_path), "ranking": scored, "winner": winner}, f, indent=2)

    print(f"Winner: {winner['variant']} (score={winner['score']:.4f})")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
