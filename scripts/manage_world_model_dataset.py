from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from world_model.replay import ReplayWriter


def _load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_manifest(path: Path, split: str, episodes: list[str], summary: dict[str, object]) -> None:
    payload = {
        "split": split,
        "episodes": episodes,
        "summary": summary,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _episode_summary(episode_paths: list[Path]) -> dict[str, object]:
    frame_counts: list[int] = []
    reward_means: list[float] = []
    progress_means: list[float] = []
    offtrack_flags: list[int] = []
    directions: Counter[str] = Counter()
    sources: Counter[str] = Counter()

    for episode_path in episode_paths:
        episode = ReplayWriter.load_episode(episode_path)
        metadata = dict(episode.metadata)
        frame_counts.append(int(episode.observations_uint8.shape[0]))
        reward_means.append(float(np.sum(episode.rewards)))
        progress_means.append(float(metadata.get("final_progress", metadata.get("progress", 0.0))))
        offtrack_flags.append(int(bool(metadata.get("offtrack_seen", metadata.get("offtrack", False)))))
        directions[str(metadata.get("direction", "UNKNOWN")).upper()] += 1
        sources[str(metadata.get("collection_source", "unknown"))] += 1

    total_frames = int(sum(frame_counts))
    return {
        "num_episodes": len(episode_paths),
        "total_frames": total_frames,
        "mean_episode_length": float(np.mean(frame_counts)) if frame_counts else 0.0,
        "mean_episode_reward": float(np.mean(reward_means)) if reward_means else 0.0,
        "mean_progress": float(np.mean(progress_means)) if progress_means else 0.0,
        "offtrack_rate": float(np.mean(offtrack_flags)) if offtrack_flags else 0.0,
        "directions": dict(directions),
        "sources": dict(sources),
    }


def _print_summary(name: str, summary: dict[str, object]) -> None:
    print(f"[{name}] episodes={summary['num_episodes']} frames={summary['total_frames']}")
    print(
        f"[{name}] mean_len={summary['mean_episode_length']:.1f} "
        f"mean_reward={summary['mean_episode_reward']:.2f} "
        f"mean_progress={summary['mean_progress']:.4f} "
        f"offtrack_rate={summary['offtrack_rate']:.4f}"
    )
    print(f"[{name}] directions={summary['directions']}")
    print(f"[{name}] sources={summary['sources']}")


def summarize_manifests(manifest_paths: list[Path]) -> None:
    for manifest_path in manifest_paths:
        payload = _load_manifest(manifest_path)
        episodes = [Path(path) for path in payload.get("episodes", [])]
        summary = _episode_summary(episodes)
        _print_summary(manifest_path.name, summary)


def merge_manifests(train_manifests: list[Path], val_manifests: list[Path], output_prefix: str, replay_dir: Path) -> None:
    train_episode_paths: list[Path] = []
    val_episode_paths: list[Path] = []
    source_manifests: list[str] = []

    for manifest_path in train_manifests:
        payload = _load_manifest(manifest_path)
        train_episode_paths.extend(Path(path) for path in payload.get("episodes", []))
        source_manifests.append(manifest_path.name)

    for manifest_path in val_manifests:
        payload = _load_manifest(manifest_path)
        val_episode_paths.extend(Path(path) for path in payload.get("episodes", []))
        source_manifests.append(manifest_path.name)

    train_summary = _episode_summary(train_episode_paths)
    val_summary = _episode_summary(val_episode_paths)
    train_summary["source_manifests"] = sorted(source_manifests)
    val_summary["source_manifests"] = sorted(source_manifests)

    train_output = replay_dir / f"{output_prefix}_train_manifest.json"
    val_output = replay_dir / f"{output_prefix}_val_manifest.json"
    _save_manifest(train_output, "train", [str(path) for path in train_episode_paths], train_summary)
    _save_manifest(val_output, "val", [str(path) for path in val_episode_paths], val_summary)

    print(f"Saved merged train manifest to {train_output}")
    print(f"Saved merged val manifest to {val_output}")
    _print_summary(train_output.name, train_summary)
    _print_summary(val_output.name, val_summary)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize or merge world-model replay manifests.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    summarize_parser = subparsers.add_parser("summarize", help="Summarize one or more manifest files.")
    summarize_parser.add_argument("manifests", nargs="+", help="Manifest files to summarize.")

    merge_parser = subparsers.add_parser("merge", help="Merge explicit train/val manifest sets into named manifests.")
    merge_parser.add_argument("--replay-dir", default="results/world_model/replay")
    merge_parser.add_argument("--output-prefix", required=True, help="Prefix for merged train/val manifest names.")
    merge_parser.add_argument("--train-manifests", nargs="+", required=True, help="Manifest files contributing train episodes.")
    merge_parser.add_argument("--val-manifests", nargs="+", required=True, help="Manifest files contributing val episodes.")

    args = parser.parse_args()

    if args.command == "summarize":
        summarize_manifests([Path(path) for path in args.manifests])
        return

    if args.command == "merge":
        merge_manifests(
            train_manifests=[Path(path) for path in args.train_manifests],
            val_manifests=[Path(path) for path in args.val_manifests],
            output_prefix=str(args.output_prefix),
            replay_dir=Path(args.replay_dir),
        )
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
