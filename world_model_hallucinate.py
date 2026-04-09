from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from world_model.models import RSSMSequence
from world_model.training import build_replay_loader, save_hallucination_video


def load_manifest(path: str | Path) -> list[str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return list(payload["episodes"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a hallucination video from a trained RSSM.")
    parser.add_argument("--config", default="config/world_model_config.yaml")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    val_paths = load_manifest(Path(config["paths"]["replay_dir"]) / "val_manifest.json")
    loader = build_replay_loader(
        val_paths,
        sequence_length=int(config["offline_training"]["sequence_length"]),
        batch_size=1,
        shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(args.checkpoint or (Path(config["paths"]["checkpoints_dir"]) / "rssm_sequence.pt"))
    payload = torch.load(checkpoint_path, map_location=device)
    model = RSSMSequence(**config["rssm"]).to(device)
    model.load_state_dict(payload["model_state_dict"])

    batch = next(iter(loader))
    output_path = Path(config["paths"]["artifacts_dir"]) / "hallucination" / "hallucination_only.mp4"
    save_hallucination_video(
        model=model,
        batch=batch,
        output_path=output_path,
        device=device,
        context_length=int(config["offline_training"]["hallucination_context"]),
        horizon=int(config["offline_training"]["hallucination_horizon"]),
    )
    print(f"Saved hallucination video to {output_path}")


if __name__ == "__main__":
    main()
