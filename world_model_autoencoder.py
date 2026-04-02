from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from train import create_env, load_config
from world_model.training import train_autoencoder_sanity


def collect_fixed_batch(base_config_path: str | Path, batch_size: int, seed: int) -> torch.Tensor:
    config = load_config(base_config_path)
    env = create_env(config, rank=0, seed=seed)
    frames = []
    obs = env.reset()
    try:
        while len(frames) < batch_size:
            frames.append(torch.as_tensor(obs, dtype=torch.float32).permute(2, 0, 1) / 255.0)
            obs, _, done, _ = env.step(env.action_space.sample())
            if done:
                obs = env.reset()
    finally:
        env.close()
    return torch.stack(frames, dim=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the world-model autoencoder sanity check.")
    parser.add_argument("--config", default="config/world_model_config.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    images = collect_fixed_batch(
        base_config_path=config["base_config_path"],
        batch_size=int(config["autoencoder"]["batch_size"]),
        seed=int(config["collector"]["seed"]),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    artifacts = train_autoencoder_sanity(
        images=images,
        output_dir=Path(config["paths"]["artifacts_dir"]) / "autoencoder",
        device=device,
        epochs=int(config["autoencoder"]["epochs"]),
        learning_rate=float(config["autoencoder"]["learning_rate"]),
    )
    print(f"Saved autoencoder checkpoint to {artifacts.model_path}")
    print(f"Saved reconstruction grid to {artifacts.grid_path}")
    print(f"Final reconstruction loss: {artifacts.final_loss:.8f}")


if __name__ == "__main__":
    main()
