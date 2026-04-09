from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import yaml


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_manifest(path: Path, split: str, episodes: list[str], source_manifests: list[str]) -> None:
    payload = {
        "split": split,
        "episodes": episodes,
        "summary": {
            "num_episodes": float(len(episodes)),
        },
        "source_manifests": source_manifests,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine bucket manifests into train/val world-model manifests.")
    parser.add_argument("--config", default="config/world_model_config.yaml")
    parser.add_argument("--prefix", default="manual_", help="Only include manifests whose filename starts with this prefix.")
    parser.add_argument("--exclude", action="append", default=[], help="Filename substring to exclude. Can be passed multiple times.")
    parser.add_argument("--val-per-manifest", type=int, default=1, help="Episodes to reserve for validation from each source manifest.")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    replay_dir = Path(config["paths"]["replay_dir"])
    excluded = tuple(args.exclude)
    manifest_paths = sorted(
        path
        for path in replay_dir.glob(f"{args.prefix}*_manifest.json")
        if not any(token in path.name for token in excluded)
    )
    if not manifest_paths:
        raise ValueError(f"No manifests matched prefix {args.prefix!r} in {replay_dir}")

    rng = random.Random(args.seed)
    train_episodes: list[str] = []
    val_episodes: list[str] = []
    source_names: list[str] = []

    for manifest_path in manifest_paths:
        payload = load_manifest(manifest_path)
        episodes = list(payload.get("episodes", []))
        if not episodes:
            continue
        rng.shuffle(episodes)
        val_count = min(int(args.val_per_manifest), max(1, len(episodes) - 1))
        val_episodes.extend(episodes[:val_count])
        train_episodes.extend(episodes[val_count:])
        source_names.append(manifest_path.name)

    if not train_episodes or not val_episodes:
        raise ValueError("Need at least one train episode and one validation episode after splitting manifests.")

    save_manifest(replay_dir / "train_manifest.json", "train", train_episodes, source_names)
    save_manifest(replay_dir / "val_manifest.json", "val", val_episodes, source_names)

    print(f"Prepared train_manifest.json with {len(train_episodes)} episodes")
    print(f"Prepared val_manifest.json with {len(val_episodes)} episodes")
    print("Included source manifests:")
    for name in source_names:
        print(f"  - {name}")


if __name__ == "__main__":
    main()
