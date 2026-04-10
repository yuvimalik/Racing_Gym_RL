from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from world_model.collector import (
    collect_automatic_maneuver_dataset,
    collect_ppo_policy_dataset,
    save_collection_manifest,
)


def _run_automatic(config: dict, args: argparse.Namespace) -> None:
    """Original scripted-maneuver collection path."""
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


def _run_ppo_policy(config: dict, args: argparse.Namespace) -> None:
    """Collect episodes using a trained PPO policy checkpoint.

    Why: A policy-based collector produces more natural, correlated action
    sequences than scripted maneuvers. Using a mid-training checkpoint (e.g.
    ppo_racecar_300000_steps.zip) gives the world model diverse driving data
    including cornering, braking, and mild recovery — without being so perfect
    that it never shows edge cases.

    Supports both .pt (custom PyTorch) and .zip (SB3) checkpoints.
    """
    output_dir = Path(config["paths"]["replay_dir"])
    collector_cfg = config["collector"]["automatic"]
    base_seed = int(collector_cfg.get("seed", 42))
    directions = args.directions if args.directions else [str(d).upper() for d in collector_cfg.get("directions", ["CCW"])]

    for i, direction in enumerate(directions):
        for split_name, frame_target in [
            ("train", args.train_frames),
            ("val", args.val_frames),
        ]:
            if frame_target <= 0:
                continue

            split = f"ppo_{Path(args.policy_checkpoint).stem}_{direction.lower()}_{split_name}"
            overrides = {
                "environment": {
                    "direction": direction,
                    "use_random_direction": False,
                }
            }
            result = collect_ppo_policy_dataset(
                base_config_path=config["base_config_path"],
                output_dir=output_dir,
                split=split,
                target_frames=frame_target,
                seed=base_seed + i,
                policy_checkpoint=args.policy_checkpoint,
                policy_variant=args.policy_variant,
                deterministic=args.deterministic,
                render=args.render,
                record_video=args.record_video,
                video_fps=float(collector_cfg.get("video_fps", 30.0)),
                config_overrides=overrides,
                device=args.device,
            )
            manifest_path = save_collection_manifest(
                output_dir=output_dir,
                split=split,
                episode_paths=result.episode_paths,
                summary=result.summary,
            )
            print(f"Saved {len(result.episode_paths)} PPO policy episodes to {manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect replay episodes for the world model. "
                    "By default uses scripted maneuvers. Use --policy_checkpoint to collect with a trained PPO policy."
    )
    parser.add_argument("--config", default="config/world_model_config.yaml")

    # PPO policy collection arguments
    parser.add_argument(
        "--policy_checkpoint",
        default=None,
        help="Path to a PPO checkpoint (.pt for custom format, .zip for SB3). "
             "When provided, collects data by running this policy instead of scripted maneuvers. "
             "Example: models/ppo_racecar_300000_steps.zip or models/best_model_torch.pt",
    )
    parser.add_argument(
        "--policy_variant",
        default="autoresearch_run_008",
        choices=["legacy", "autoresearch_run_008"],
        help="Policy class variant for .pt checkpoints. "
             "Use 'autoresearch_run_008' for the tanh-squashed variant (most recent). "
             "Ignored for .zip (SB3) checkpoints.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=True,
        help="Run policy deterministically (use mean action, no sampling). "
             "Gives cleaner trajectories. Use --no-deterministic for stochastic rollouts.",
    )
    parser.add_argument("--no-deterministic", dest="deterministic", action="store_false")
    parser.add_argument(
        "--train_frames",
        type=int,
        default=5000,
        help="Number of training frames to collect (PPO mode only).",
    )
    parser.add_argument(
        "--val_frames",
        type=int,
        default=1000,
        help="Number of validation frames to collect (PPO mode only).",
    )
    parser.add_argument(
        "--directions",
        nargs="+",
        default=None,
        help="Directions to collect in (e.g. CCW CW). Defaults to config value.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        default=False,
        help="Render the environment window during collection.",
    )
    parser.add_argument(
        "--record_video",
        action="store_true",
        default=False,
        help="Save a video of the collection run.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for policy inference: 'cpu' or 'cuda'. "
             "CPU is usually fine since collection is env-bound, not compute-bound.",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if args.policy_checkpoint is not None:
        _run_ppo_policy(config, args)
    else:
        _run_automatic(config, args)


if __name__ == "__main__":
    main()
