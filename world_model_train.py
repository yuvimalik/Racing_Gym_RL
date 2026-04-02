from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from world_model.models import RSSMSequence
from world_model.training import build_replay_loader, save_hallucination_video, train_world_model_epoch


def load_manifest(path: str | Path) -> list[str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return list(payload["episodes"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the offline RSSM world model.")
    parser.add_argument("--config", default="config/world_model_config.yaml")
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    replay_dir = Path(config["paths"]["replay_dir"])
    train_paths = load_manifest(replay_dir / "train_manifest.json")
    val_paths = load_manifest(replay_dir / "val_manifest.json")

    train_loader = build_replay_loader(
        train_paths,
        sequence_length=int(config["offline_training"]["sequence_length"]),
        batch_size=int(config["offline_training"]["batch_size"]),
        shuffle=True,
    )
    val_loader = build_replay_loader(
        val_paths,
        sequence_length=int(config["offline_training"]["sequence_length"]),
        batch_size=1,
        shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RSSMSequence(**config["rssm"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["offline_training"]["learning_rate"]))

    for epoch in range(int(args.epochs)):
        metrics = train_world_model_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            free_nats=float(config["offline_training"]["free_nats"]),
            kl_scale=float(config["offline_training"]["kl_scale"]),
            reward_scale=float(config["offline_training"]["reward_scale"]),
        )
        print(f"Epoch {epoch + 1}: {metrics}")

    checkpoint_dir = Path(config["paths"]["checkpoints_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "rssm_sequence.pt"
    torch.save({"model_state_dict": model.state_dict(), "config": config}, checkpoint_path)
    print(f"Saved world-model checkpoint to {checkpoint_path}")

    val_batch = next(iter(val_loader))
    video_path = Path(config["paths"]["artifacts_dir"]) / "hallucination" / "hallucination.mp4"
    save_hallucination_video(
        model=model,
        batch=val_batch,
        output_path=video_path,
        device=device,
        context_length=int(config["offline_training"]["hallucination_context"]),
        horizon=int(config["offline_training"]["hallucination_horizon"]),
    )
    print(f"Saved hallucination video to {video_path}")


if __name__ == "__main__":
    main()
