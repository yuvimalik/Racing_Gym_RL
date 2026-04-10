from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import torch
import yaml

from world_model.control import FrozenWorldModel, LatentActor, LatentCritic
from world_model.control_training import (
    evaluate_latent_actor,
    save_control_metrics,
    train_latent_actor_critic_epoch,
)
from world_model.models import RSSMSequence
from world_model.training import build_replay_loader


def load_manifest(path: str | Path) -> list[str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return list(payload["episodes"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a latent actor-critic on top of a frozen RSSM world model.")
    parser.add_argument("--config", default="config/world_model_config.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--world-model-checkpoint", default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    control_cfg = config["control_training"]
    replay_dir = Path(config["paths"]["replay_dir"])
    train_paths = load_manifest(replay_dir / "train_manifest.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Latent-control device: {device}")

    cuda_available = torch.cuda.is_available()
    num_workers = int(control_cfg.get("num_workers", config["offline_training"].get("num_workers", 0)))
    pin_memory = bool(control_cfg.get("pin_memory", config["offline_training"].get("pin_memory", False)) and cuda_available)
    persistent_workers = bool(
        control_cfg.get("persistent_workers", config["offline_training"].get("persistent_workers", False)) and num_workers > 0
    )
    prefetch_factor = control_cfg.get("prefetch_factor", config["offline_training"].get("prefetch_factor", None))

    context_length = int(control_cfg["context_length"])
    imagination_horizon = int(control_cfg["imagination_horizon"])
    batch_size = int(control_cfg["batch_size"])
    epochs = int(args.epochs or control_cfg["epochs"])
    window_stride = int(control_cfg.get("window_stride", config["offline_training"].get("window_stride", 1)))

    train_loader = build_replay_loader(
        train_paths,
        sequence_length=context_length,
        batch_size=batch_size,
        shuffle=True,
        window_stride=window_stride,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )

    checkpoint_path = Path(args.world_model_checkpoint or control_cfg.get("world_model_checkpoint") or (Path(config["paths"]["checkpoints_dir"]) / "rssm_sequence.pt"))
    if not checkpoint_path.is_absolute():
        checkpoint_path = Path.cwd() / checkpoint_path
    checkpoint_payload = torch.load(checkpoint_path, map_location=device)
    rssm_config = checkpoint_payload.get("config", {}).get("rssm", config["rssm"])
    rssm = RSSMSequence(**rssm_config).to(device)
    rssm.load_state_dict(checkpoint_payload["model_state_dict"])
    world_model = FrozenWorldModel(rssm)

    actor = LatentActor(
        latent_dim=world_model.latent_dim,
        action_dim=world_model.action_dim,
        hidden_dim=int(control_cfg.get("actor_hidden_dim", 256)),
    ).to(device)
    critic = LatentCritic(
        latent_dim=world_model.latent_dim,
        hidden_dim=int(control_cfg.get("critic_hidden_dim", 256)),
    ).to(device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=float(control_cfg["actor_learning_rate"]))
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=float(control_cfg["critic_learning_rate"]))

    run_name = str(args.run_name or f"latent_control_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    checkpoint_root = Path(control_cfg.get("checkpoints_dir", "./models/world_model_control"))
    artifacts_root = Path(control_cfg.get("artifacts_dir", "./results/world_model/control"))
    checkpoint_dir = checkpoint_root / run_name
    artifacts_dir = artifacts_root / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run name: {run_name}")
    print(f"Frozen RSSM checkpoint: {checkpoint_path}")
    print(f"Train episodes: {len(train_paths)}")
    print(f"Train windows: {len(train_loader.dataset)}")
    print(f"Train batches per epoch: {len(train_loader)}")
    print(f"Context length: {context_length}")
    print(f"Imagination horizon: {imagination_horizon}")
    print(f"Batch size: {batch_size}")
    print(f"Window stride: {window_stride}")
    print(f"DataLoader workers: {num_workers}")
    print(f"pin_memory: {pin_memory}")
    print(f"persistent_workers: {persistent_workers}")

    epoch_durations: list[float] = []
    run_start = time.perf_counter()

    for epoch in range(epochs):
        print(f"Starting latent-control epoch {epoch + 1}/{epochs}", flush=True)
        epoch_start = time.perf_counter()
        train_metrics = train_latent_actor_critic_epoch(
            world_model=world_model,
            actor=actor,
            critic=critic,
            loader=train_loader,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            device=device,
            context_length=context_length,
            imagination_horizon=imagination_horizon,
            discount=float(control_cfg.get("discount", 0.99)),
            action_l2_scale=float(control_cfg.get("action_l2_scale", 0.0)),
            log_every=int(args.log_every),
        )
        epoch_duration = time.perf_counter() - epoch_start
        epoch_durations.append(epoch_duration)
        elapsed = time.perf_counter() - run_start
        mean_epoch_time = sum(epoch_durations) / len(epoch_durations)
        eta_seconds = (epochs - (epoch + 1)) * mean_epoch_time
        memory_suffix = ""
        if device.type == "cuda":
            max_memory_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
            memory_suffix = f" | max_gpu_mem={max_memory_gb:.2f}GB"
            torch.cuda.reset_peak_memory_stats(device)
        print(
            f"Epoch {epoch + 1}: {train_metrics} "
            f"| epoch_time={epoch_duration:.1f}s "
            f"| elapsed={elapsed / 60.0:.1f}m "
            f"| eta={eta_seconds / 60.0:.1f}m"
            f"{memory_suffix}",
            flush=True,
        )

        eval_metrics: dict[str, float] | None = None
        if (epoch + 1) % int(control_cfg.get("eval_every", 1)) == 0 or (epoch + 1) == epochs:
            real_video_path = artifacts_dir / f"eval_epoch_{epoch + 1:03d}_real.mp4"
            compare_video_path = artifacts_dir / f"eval_epoch_{epoch + 1:03d}_compare.mp4"
            eval_metrics = evaluate_latent_actor(
                world_model=world_model,
                actor=actor,
                base_config_path=config["base_config_path"],
                device=device,
                episodes=int(control_cfg.get("eval_episodes", 3)),
                max_steps=int(control_cfg.get("eval_max_steps", 1000)),
                seed=int(control_cfg.get("eval_seed", 123)),
                record_video=bool(control_cfg.get("record_eval_video", True)),
                video_path=real_video_path,
                record_compare_video=bool(control_cfg.get("record_compare_video", True)),
                compare_video_path=compare_video_path,
                video_fps=float(control_cfg.get("eval_video_fps", 10.0)),
                compare_context_length=int(control_cfg.get("compare_context_length", context_length)),
                compare_horizon=int(control_cfg.get("compare_horizon", 50)),
                overlay=bool(control_cfg.get("eval_overlay", True)),
            )
            print(f"Actor evaluation: {eval_metrics}", flush=True)

        if (epoch + 1) % int(args.save_every) == 0 or (epoch + 1) == epochs:
            payload = {
                "epoch": epoch + 1,
                "train_metrics": train_metrics,
                "eval_metrics": eval_metrics,
                "actor_state_dict": actor.state_dict(),
                "critic_state_dict": critic.state_dict(),
                "world_model_checkpoint": str(checkpoint_path),
                "config": config,
            }
            checkpoint_epoch_path = checkpoint_dir / f"latent_actor_critic_epoch_{epoch + 1:03d}.pt"
            torch.save(payload, checkpoint_epoch_path)
            latest_checkpoint_path = checkpoint_root / "latent_actor_critic.pt"
            torch.save(payload, latest_checkpoint_path)
            metrics_path = artifacts_dir / f"metrics_epoch_{epoch + 1:03d}.json"
            save_control_metrics(metrics_path, {"epoch": epoch + 1, "train_metrics": train_metrics, "eval_metrics": eval_metrics})
            print(f"Saved latent-control checkpoint to {checkpoint_epoch_path}", flush=True)
            print(f"Updated latest latent-control checkpoint at {latest_checkpoint_path}", flush=True)
            print(f"Saved control metrics to {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
