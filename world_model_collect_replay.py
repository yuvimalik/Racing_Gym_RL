from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from world_model.collector import DrunkExpertConfig, collect_drunk_expert_dataset, save_collection_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect drunk-expert replay episodes for the world model.")
    parser.add_argument("--config", default="config/world_model_config.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    collector_cfg = config["collector"]
    noise_cfg = DrunkExpertConfig(**collector_cfg["noise"])
    output_dir = Path(config["paths"]["replay_dir"])

    train_paths = collect_drunk_expert_dataset(
        base_config_path=config["base_config_path"],
        output_dir=output_dir,
        checkpoint_path=collector_cfg["checkpoint_path"],
        split="train",
        target_frames=int(collector_cfg["target_train_frames"]),
        seed=int(collector_cfg["seed"]),
        noise_cfg=noise_cfg,
    )
    val_paths = collect_drunk_expert_dataset(
        base_config_path=config["base_config_path"],
        output_dir=output_dir,
        checkpoint_path=collector_cfg["checkpoint_path"],
        split="val",
        target_frames=int(collector_cfg["target_val_frames"]),
        seed=int(collector_cfg["seed"]) + 1,
        noise_cfg=noise_cfg,
    )

    train_manifest = save_collection_manifest(output_dir, "train", train_paths)
    val_manifest = save_collection_manifest(output_dir, "val", val_paths)
    print(f"Saved {len(train_paths)} training episodes to {train_manifest}")
    print(f"Saved {len(val_paths)} validation episodes to {val_manifest}")


if __name__ == "__main__":
    main()
