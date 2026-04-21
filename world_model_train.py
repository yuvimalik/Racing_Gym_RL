from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import torch
import yaml

from world_model.models import RSSMSequence
from world_model.training import (
    build_curated_eval_batch,
    build_replay_loader,
    save_hallucination_video,
    save_side_by_side_hallucination_video,
    train_world_model_epoch,
)


def load_manifest(path: str | Path) -> list[str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return list(payload["episodes"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the offline RSSM world model.")
    parser.add_argument("--config", default="config/world_model_config.yaml")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    replay_dir = Path(config["paths"]["replay_dir"])
    offline_cfg = config["offline_training"]
    train_paths = load_manifest(replay_dir / "train_manifest.json")
    val_paths = load_manifest(replay_dir / "val_manifest.json")
    cuda_available = torch.cuda.is_available()
    num_workers = int(offline_cfg.get("num_workers", 0))
    pin_memory = bool(offline_cfg.get("pin_memory", False) and cuda_available)
    persistent_workers = bool(offline_cfg.get("persistent_workers", False) and num_workers > 0)
    prefetch_factor = offline_cfg.get("prefetch_factor", None)
    use_amp = bool(offline_cfg.get("use_amp", True) and cuda_available)

    train_loader = build_replay_loader(
        train_paths,
        sequence_length=int(offline_cfg["sequence_length"]),
        batch_size=int(offline_cfg["batch_size"]),
        shuffle=True,
        window_stride=int(offline_cfg.get("window_stride", 1)),
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    val_sequence_length = max(
        int(offline_cfg["sequence_length"]),
        int(offline_cfg["hallucination_context"]) + int(offline_cfg["hallucination_horizon"]),
    )
    val_loader = build_replay_loader(
        val_paths,
        sequence_length=val_sequence_length,
        batch_size=1,
        shuffle=False,
        window_stride=int(offline_cfg.get("window_stride", 1)),
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )

    print(f"Loaded {len(train_paths)} train episodes from {replay_dir / 'train_manifest.json'}")
    print(f"Loaded {len(val_paths)} val episodes from {replay_dir / 'val_manifest.json'}")
    print(f"Train windows: {len(train_loader.dataset)}")
    print(f"Val windows: {len(val_loader.dataset)}")
    print(f"Val sequence length for hallucination: {val_sequence_length}")
    print(f"Window stride: {int(offline_cfg.get('window_stride', 1))}")
    print(f"Train batches per epoch: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Batch size: {int(offline_cfg['batch_size'])}")
    print(f"Hallucination FPS: {float(offline_cfg.get('hallucination_video_fps', 10.0))}")
    print(f"DataLoader workers: {num_workers}")
    print(f"pin_memory: {pin_memory}")
    print(f"persistent_workers: {persistent_workers}")
    print(f"prefetch_factor: {prefetch_factor}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}")
    print(f"AMP enabled: {use_amp}")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    run_name = str(args.run_name or f"rssm_seq{int(offline_cfg['sequence_length'])}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    print(f"Run name: {run_name}")
    model = RSSMSequence(**config["rssm"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(offline_cfg["learning_rate"]))
    grad_scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    checkpoint_root = Path(config["paths"]["checkpoints_dir"])
    checkpoint_dir = checkpoint_root / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    hallucination_root = Path(config["paths"]["artifacts_dir"]) / "hallucination"
    hallucination_dir = hallucination_root / run_name
    hallucination_dir.mkdir(parents=True, exist_ok=True)
    val_batch = next(iter(val_loader))
    curated_clip_specs = list(offline_cfg.get("curated_eval_clips", []))
    curated_batches: list[dict[str, object]] = []
    if curated_clip_specs:
        print(f"Curated evaluation clips: {len(curated_clip_specs)}")
        for clip in curated_clip_specs:
            clip_name = str(clip["name"])
            clip_type = str(clip.get("clip_type", clip_name))
            clip_context = int(clip.get("context_length", offline_cfg["hallucination_context"]))
            clip_horizon = int(clip.get("horizon", offline_cfg["hallucination_horizon"]))
            clip_episode_path = Path(clip["episode_path"])
            if not clip_episode_path.is_absolute():
                clip_episode_path = Path.cwd() / clip_episode_path
            curated_batches.append(
                {
                    "name": clip_name,
                    "clip_type": clip_type,
                    "context_length": clip_context,
                    "horizon": clip_horizon,
                    "batch": build_curated_eval_batch(
                        episode_path=clip_episode_path,
                        start_index=int(clip["start_index"]),
                        context_length=clip_context,
                        horizon=clip_horizon,
                        device=device,
                    ),
                }
            )
            print(
                f"  - {clip_name} ({clip_type}) path={clip_episode_path.name} "
                f"start_index={int(clip['start_index'])} context={clip_context} horizon={clip_horizon}"
            )
    epoch_durations: list[float] = []
    run_start = time.perf_counter()

    for epoch in range(int(args.epochs)):
        print(f"Starting epoch {epoch + 1}/{int(args.epochs)}", flush=True)
        epoch_start = time.perf_counter()
        metrics = train_world_model_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            free_nats=float(offline_cfg["free_nats"]),
            kl_scale=float(offline_cfg["kl_scale"]),
            reward_scale=float(offline_cfg["reward_scale"]),
            log_every=int(args.log_every),
            use_amp=use_amp,
            grad_scaler=grad_scaler,
        )
        epoch_duration = time.perf_counter() - epoch_start
        epoch_durations.append(epoch_duration)
        elapsed = time.perf_counter() - run_start
        mean_epoch_time = sum(epoch_durations) / len(epoch_durations)
        remaining_epochs = int(args.epochs) - (epoch + 1)
        eta_seconds = remaining_epochs * mean_epoch_time
        batches_per_second = len(train_loader) / epoch_duration if epoch_duration > 0 else 0.0
        windows_per_second = len(train_loader.dataset) / epoch_duration if epoch_duration > 0 else 0.0
        memory_suffix = ""
        if device.type == "cuda":
            max_memory_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
            memory_suffix = f" | max_gpu_mem={max_memory_gb:.2f}GB"
            torch.cuda.reset_peak_memory_stats(device)
        print(
            f"Epoch {epoch + 1}: {metrics} "
            f"| epoch_time={epoch_duration:.1f}s "
            f"| elapsed={elapsed / 60.0:.1f}m "
            f"| eta={eta_seconds / 60.0:.1f}m "
            f"| batches_per_sec={batches_per_second:.2f} "
            f"| windows_per_sec={windows_per_second:.2f}"
            f"{memory_suffix}",
            flush=True,
        )

        if (epoch + 1) % int(args.save_every) == 0 or (epoch + 1) == int(args.epochs):
            checkpoint_payload = {
                "model_state_dict": model.state_dict(),
                "config": config,
                "epoch": epoch + 1,
                "metrics": metrics,
            }
            checkpoint_path = checkpoint_dir / f"rssm_sequence_epoch_{epoch + 1:03d}.pt"
            torch.save(checkpoint_payload, checkpoint_path)
            latest_checkpoint_path = checkpoint_root / "rssm_sequence.pt"
            torch.save(checkpoint_payload, latest_checkpoint_path)
            print(f"Saved world-model checkpoint to {checkpoint_path}", flush=True)
            print(f"Updated latest checkpoint at {latest_checkpoint_path}", flush=True)

            video_path = hallucination_dir / f"hallucination_epoch_{epoch + 1:03d}.mp4"
            save_hallucination_video(
                model=model,
                batch=val_batch,
                output_path=video_path,
                device=device,
                context_length=int(offline_cfg["hallucination_context"]),
                horizon=int(offline_cfg["hallucination_horizon"]),
                fps=float(offline_cfg.get("hallucination_video_fps", 10.0)),
            )
            latest_video_path = hallucination_root / "hallucination.mp4"
            save_hallucination_video(
                model=model,
                batch=val_batch,
                output_path=latest_video_path,
                device=device,
                context_length=int(offline_cfg["hallucination_context"]),
                horizon=int(offline_cfg["hallucination_horizon"]),
                fps=float(offline_cfg.get("hallucination_video_fps", 10.0)),
            )
            print(f"Saved hallucination video to {video_path}", flush=True)
            print(f"Updated latest hallucination video at {latest_video_path}", flush=True)

            for curated in curated_batches:
                curated_name = str(curated["name"])
                curated_output = hallucination_dir / f"epoch_{epoch + 1:03d}_{curated_name}_side_by_side.mp4"
                save_side_by_side_hallucination_video(
                    model=model,
                    batch=curated["batch"],  # type: ignore[arg-type]
                    output_path=curated_output,
                    device=device,
                    context_length=int(curated["context_length"]),
                    horizon=int(curated["horizon"]),
                    fps=float(offline_cfg.get("hallucination_video_fps", 10.0)),
                )
                print(f"Saved curated side-by-side video to {curated_output}", flush=True)


if __name__ == "__main__":
    main()
