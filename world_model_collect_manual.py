from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import yaml

from world_model.collector import collect_manual_keyboard_dataset, save_collection_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect manual keyboard teleoperation replay episodes for the world model.")
    parser.add_argument("--config", default="config/world_model_config.yaml")
    parser.add_argument("--direction", default=None)
    parser.add_argument("--regime", default=None)
    parser.add_argument("--bucket", default="manual_racing")
    parser.add_argument("--split", default=None)
    parser.add_argument("--target-frames", type=int, default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    manual_cfg = config["collector"]["manual"]
    direction = str(args.direction or manual_cfg.get("direction", "CCW")).upper()
    regime_name = str(args.regime or manual_cfg.get("regime", "medium"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    split = str(args.split or f"{args.bucket}_{timestamp}")
    regime_cfg = config["collector"]["automatic"]["regimes"][regime_name]
    overrides = {
        "environment": {
            "direction": direction,
            "use_random_direction": False,
        },
        "safety_governor": dict(regime_cfg.get("config_overrides", {}).get("safety_governor", {})),
    }

    result = collect_manual_keyboard_dataset(
        base_config_path=config["base_config_path"],
        output_dir=Path(config["paths"]["replay_dir"]),
        split=split,
        seed=int(manual_cfg["seed"]),
        target_frames=int(args.target_frames if args.target_frames is not None else manual_cfg["target_frames"]),
        render=bool(manual_cfg.get("render", True)),
        record_video=bool(manual_cfg.get("record_video", True)),
        video_fps=float(manual_cfg.get("video_fps", 30.0)),
        config_overrides=overrides,
        speed_regime=regime_name,
    )
    manifest_path = save_collection_manifest(
        output_dir=Path(config["paths"]["replay_dir"]),
        split=split,
        episode_paths=result.episode_paths,
        summary=result.summary,
    )
    print(f"Saved {len(result.episode_paths)} manual episodes to {manifest_path}")


if __name__ == "__main__":
    main()
