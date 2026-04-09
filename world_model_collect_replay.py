from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from world_model.collector import collect_automatic_maneuver_dataset, save_collection_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect automatic maneuver-based replay episodes for the world model.")
    parser.add_argument("--config", default="config/world_model_config.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    collector_cfg = config["collector"]["automatic"]
    output_dir = Path(config["paths"]["replay_dir"])
    render = bool(collector_cfg.get("render", True))
    record_video = bool(collector_cfg.get("record_video", True))
    video_fps = float(collector_cfg.get("video_fps", 30.0))
    base_seed = int(collector_cfg["seed"])
    split_frames = collector_cfg["splits"]
    split_order = [("train", int(split_frames["train_frames"])), ("val", int(split_frames["val_frames"]))]
    directions = [str(direction).upper() for direction in collector_cfg["directions"]]
    regime_names = list(collector_cfg["regimes"].keys())

    seed_offset = 0
    for regime_name in regime_names:
        regime_cfg = collector_cfg["regimes"][regime_name]
        for direction in directions:
            for split_name, frame_target in split_order:
                overrides = {
                    "environment": {
                        "direction": direction,
                        "use_random_direction": False,
                    },
                    "safety_governor": dict(regime_cfg.get("config_overrides", {}).get("safety_governor", {})),
                    "collector_runtime": {
                        "regime": {
                            "throttle_scale": float(regime_cfg.get("throttle_scale", 1.0)),
                            "steering_scale": float(regime_cfg.get("steering_scale", 1.0)),
                            "brake_scale": float(regime_cfg.get("brake_scale", 1.0)),
                            "transition_steps": int(regime_cfg.get("transition_steps", 4)),
                        }
                    },
                }
                split = f"{regime_name}_{direction.lower()}_{split_name}"
                result = collect_automatic_maneuver_dataset(
                    base_config_path=config["base_config_path"],
                    output_dir=output_dir,
                    split=split,
                    target_frames=frame_target,
                    seed=base_seed + seed_offset,
                    render=render,
                    record_video=record_video,
                    video_fps=video_fps,
                    config_overrides=overrides,
                    regime_name=regime_name,
                )
                manifest_path = save_collection_manifest(
                    output_dir=output_dir,
                    split=split,
                    episode_paths=result.episode_paths,
                    summary=result.summary,
                )
                print(f"Saved {len(result.episode_paths)} automatic episodes to {manifest_path}")
                seed_offset += 1


if __name__ == "__main__":
    main()
