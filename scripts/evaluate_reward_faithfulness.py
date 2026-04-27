from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from world_model.models import RSSMSequence
from world_model.training import build_replay_loader


def load_manifest(path: str | Path) -> list[str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [episode.replace("\\", "/") for episode in payload["episodes"]]


def reward_correlation(predicted: np.ndarray, target: np.ndarray) -> float:
    if predicted.size < 2 or target.size < 2:
        return float("nan")
    pred_std = float(predicted.std())
    target_std = float(target.std())
    if pred_std < 1e-8 or target_std < 1e-8:
        return float("nan")
    return float(np.corrcoef(predicted, target)[0, 1])


def masked_corr(predicted: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    keep = mask.astype(bool)
    if keep.sum() < 2:
        return float("nan")
    return reward_correlation(predicted[keep], target[keep])


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate short-horizon reward and telemetry faithfulness of a frozen RSSM world model.")
    parser.add_argument("--config", default="config/world_model_config.yaml")
    parser.add_argument("--manifest", required=True, help="Held-out replay manifest for evaluation.")
    parser.add_argument("--world-model-checkpoint", required=True, help="RSSM checkpoint to evaluate.")
    parser.add_argument("--context-length", type=int, default=25)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-batches", type=int, default=25)
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    manifest_path = Path(args.manifest)
    checkpoint_path = Path(args.world_model_checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    payload = torch.load(checkpoint_path, map_location=device)
    rssm_config = payload.get("config", {}).get("rssm", config["rssm"])
    model = RSSMSequence(**rssm_config).to(device)
    missing, unexpected = model.load_state_dict(payload["model_state_dict"], strict=False)
    model.eval()

    sequence_length = int(args.context_length) + int(args.horizon)
    loader = build_replay_loader(
        load_manifest(manifest_path),
        sequence_length=sequence_length,
        batch_size=int(args.batch_size),
        shuffle=False,
        window_stride=int(config["offline_training"].get("window_stride", 1)),
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=None,
    )

    mse_values: list[float] = []
    mae_values: list[float] = []
    corr_values: list[float] = []
    sign_match_values: list[float] = []
    rollout_bias_values: list[float] = []
    telemetry_metrics: dict[str, list[float]] = {
        "speed_mse": [],
        "speed_mae": [],
        "speed_corr": [],
        "progress_delta_mse": [],
        "progress_delta_mae": [],
        "progress_delta_corr": [],
        "steer_mse": [],
        "steer_mae": [],
        "steer_corr": [],
        "corner_angle_mse": [],
        "corner_angle_mae": [],
        "corner_angle_corr": [],
        "offtrack_bce": [],
        "offtrack_accuracy": [],
    }
    batch_summaries: list[dict[str, float]] = []

    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            if batch_index > int(args.num_batches):
                break

            images = batch["images"].to(device)
            actions = batch["actions"].to(device)
            rewards = batch["rewards"].to(device)
            is_first = batch["is_first"].to(device)
            progress_delta = batch["progress_delta"].to(device)
            speed = batch["speed"].to(device)
            steer = batch["steer"].to(device)
            corner_angle = batch["corner_angle"].to(device)
            offtrack = batch["offtrack"].to(device)
            telemetry_valid = batch["telemetry_valid"].to(device)

            output = model(images=images, actions=actions, is_first=is_first)
            predicted = output.reward[:, args.context_length : args.context_length + args.horizon, 0]
            target = rewards[:, args.context_length : args.context_length + args.horizon, 0]

            batch_mse = float(torch.mean((predicted - target) ** 2).item())
            batch_mae = float(torch.mean(torch.abs(predicted - target)).item())
            pred_np = predicted.flatten().detach().cpu().numpy()
            target_np = target.flatten().detach().cpu().numpy()
            batch_corr = reward_correlation(pred_np, target_np)
            batch_sign_match = float(np.mean(np.sign(pred_np) == np.sign(target_np)))
            batch_bias = float((pred_np - target_np).mean())

            telemetry_window = telemetry_valid[:, args.context_length : args.context_length + args.horizon, 0]
            telemetry_mask_np = telemetry_window.flatten().detach().cpu().numpy()
            telemetry_summary = {}
            if float(telemetry_window.sum().item()) > 0.0:
                telemetry_targets = {
                    "speed": speed[:, args.context_length : args.context_length + args.horizon, 0],
                    "progress_delta": progress_delta[:, args.context_length : args.context_length + args.horizon, 0],
                    "steer": steer[:, args.context_length : args.context_length + args.horizon, 0],
                    "corner_angle": corner_angle[:, args.context_length : args.context_length + args.horizon, 0],
                }
                for target_name, target_tensor in telemetry_targets.items():
                    pred_tensor = output.telemetry[target_name][:, args.context_length : args.context_length + args.horizon, 0]
                    diff = pred_tensor - target_tensor
                    denom = torch.clamp(telemetry_window.sum(), min=1.0)
                    mse = float(((diff.square() * telemetry_window).sum() / denom).item())
                    mae = float(((diff.abs() * telemetry_window).sum() / denom).item())
                    pred_target_np = pred_tensor.flatten().detach().cpu().numpy()
                    target_target_np = target_tensor.flatten().detach().cpu().numpy()
                    corr = masked_corr(pred_target_np, target_target_np, telemetry_mask_np)
                    telemetry_metrics[f"{target_name}_mse"].append(mse)
                    telemetry_metrics[f"{target_name}_mae"].append(mae)
                    telemetry_metrics[f"{target_name}_corr"].append(corr)
                    telemetry_summary[f"{target_name}_mse"] = mse
                    telemetry_summary[f"{target_name}_mae"] = mae
                    telemetry_summary[f"{target_name}_corr"] = corr

                offtrack_logits = output.telemetry["offtrack_logits"][:, args.context_length : args.context_length + args.horizon, 0]
                offtrack_target = offtrack[:, args.context_length : args.context_length + args.horizon, 0]
                offtrack_bce = float(
                    ((F.binary_cross_entropy_with_logits(offtrack_logits, offtrack_target, reduction="none") * telemetry_window).sum() / torch.clamp(telemetry_window.sum(), min=1.0)).item()
                )
                offtrack_pred = (torch.sigmoid(offtrack_logits) >= 0.5).float()
                offtrack_accuracy = float(
                    (((offtrack_pred == offtrack_target).float() * telemetry_window).sum() / torch.clamp(telemetry_window.sum(), min=1.0)).item()
                )
                telemetry_metrics["offtrack_bce"].append(offtrack_bce)
                telemetry_metrics["offtrack_accuracy"].append(offtrack_accuracy)
                telemetry_summary["offtrack_bce"] = offtrack_bce
                telemetry_summary["offtrack_accuracy"] = offtrack_accuracy

            mse_values.append(batch_mse)
            mae_values.append(batch_mae)
            corr_values.append(batch_corr)
            sign_match_values.append(batch_sign_match)
            rollout_bias_values.append(batch_bias)
            batch_summaries.append(
                {
                    "batch_index": float(batch_index),
                    "mse": batch_mse,
                    "mae": batch_mae,
                    "corr": batch_corr,
                    "sign_match": batch_sign_match,
                    "bias": batch_bias,
                    **telemetry_summary,
                }
            )

    if not batch_summaries:
        raise ValueError("Reward-faithfulness evaluation saw zero batches.")

    finite_corr = [value for value in corr_values if np.isfinite(value)]
    summary = {
        "checkpoint": str(checkpoint_path),
        "manifest": str(manifest_path),
        "missing_checkpoint_keys": list(missing),
        "unexpected_checkpoint_keys": list(unexpected),
        "context_length": int(args.context_length),
        "horizon": int(args.horizon),
        "num_batches_evaluated": len(batch_summaries),
        "reward": {
            "mean_mse": float(np.mean(mse_values)),
            "mean_mae": float(np.mean(mae_values)),
            "mean_corr": float(np.mean(finite_corr)) if finite_corr else float("nan"),
            "mean_sign_match": float(np.mean(sign_match_values)),
            "mean_bias": float(np.mean(rollout_bias_values)),
        },
        "telemetry": {
            "speed": {
                "mean_mse": float(np.mean(telemetry_metrics["speed_mse"])) if telemetry_metrics["speed_mse"] else float("nan"),
                "mean_mae": float(np.mean(telemetry_metrics["speed_mae"])) if telemetry_metrics["speed_mae"] else float("nan"),
                "mean_corr": float(np.mean([v for v in telemetry_metrics["speed_corr"] if np.isfinite(v)])) if any(np.isfinite(v) for v in telemetry_metrics["speed_corr"]) else float("nan"),
            },
            "progress_delta": {
                "mean_mse": float(np.mean(telemetry_metrics["progress_delta_mse"])) if telemetry_metrics["progress_delta_mse"] else float("nan"),
                "mean_mae": float(np.mean(telemetry_metrics["progress_delta_mae"])) if telemetry_metrics["progress_delta_mae"] else float("nan"),
                "mean_corr": float(np.mean([v for v in telemetry_metrics["progress_delta_corr"] if np.isfinite(v)])) if any(np.isfinite(v) for v in telemetry_metrics["progress_delta_corr"]) else float("nan"),
            },
            "steer": {
                "mean_mse": float(np.mean(telemetry_metrics["steer_mse"])) if telemetry_metrics["steer_mse"] else float("nan"),
                "mean_mae": float(np.mean(telemetry_metrics["steer_mae"])) if telemetry_metrics["steer_mae"] else float("nan"),
                "mean_corr": float(np.mean([v for v in telemetry_metrics["steer_corr"] if np.isfinite(v)])) if any(np.isfinite(v) for v in telemetry_metrics["steer_corr"]) else float("nan"),
            },
            "corner_angle": {
                "mean_mse": float(np.mean(telemetry_metrics["corner_angle_mse"])) if telemetry_metrics["corner_angle_mse"] else float("nan"),
                "mean_mae": float(np.mean(telemetry_metrics["corner_angle_mae"])) if telemetry_metrics["corner_angle_mae"] else float("nan"),
                "mean_corr": float(np.mean([v for v in telemetry_metrics["corner_angle_corr"] if np.isfinite(v)])) if any(np.isfinite(v) for v in telemetry_metrics["corner_angle_corr"]) else float("nan"),
            },
            "offtrack": {
                "mean_bce": float(np.mean(telemetry_metrics["offtrack_bce"])) if telemetry_metrics["offtrack_bce"] else float("nan"),
                "mean_accuracy": float(np.mean(telemetry_metrics["offtrack_accuracy"])) if telemetry_metrics["offtrack_accuracy"] else float("nan"),
            },
        },
        "batch_summaries": batch_summaries,
    }

    print(json.dumps(summary, indent=2))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
