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
    compute_hallucination_metrics,
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
    parser.add_argument("--train-manifest", default=None, help="Optional explicit train manifest path.")
    parser.add_argument("--val-manifest", default=None, help="Optional explicit val manifest path.")
    parser.add_argument(
        "--init-checkpoint",
        default=None,
        help="Optional checkpoint to initialize the RSSM from before training. "
             "Loads model weights with strict=False so new telemetry heads can be added "
             "on top of older checkpoints such as E3.",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--batch-log-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--run-name", default=None)

    # W&B logging
    # How W&B works: every training run is a "run" in a "project". Metrics are
    # streamed in real-time so you can watch training on wandb.ai. When disabled
    # (--no-wandb) all W&B calls are no-ops, so the rest of the code is identical.
    parser.add_argument("--wandb-project", default="racing-world-model", help="W&B project name.")
    parser.add_argument("--no-wandb", action="store_true", default=False, help="Disable W&B logging.")

    # Distributed training (multi-GPU via torchrun / DDP)
    # Launch with: torchrun --nproc_per_node=N world_model_train.py --distributed ...
    # Each GPU process gets a unique LOCAL_RANK env variable set by torchrun.
    # DDP averages gradients across all ranks after each backward pass,
    # giving identical results to single-GPU training but N× faster.
    parser.add_argument("--distributed", action="store_true", default=False,
                        help="Enable DistributedDataParallel training. "
                             "Launch with torchrun --nproc_per_node=<N>.")

    # KL annealing: free_nats starts high (model free to use any posterior) and
    # decays to a lower value over warmup_epochs. This prevents posterior collapse
    # early in training when the reconstruction loss is still large.
    parser.add_argument("--free-nats-start", type=float, default=None,
                        help="Starting free_nats value for KL annealing. "
                             "If None, uses the fixed value from config (no annealing).")
    parser.add_argument("--free-nats-target", type=float, default=None,
                        help="Target free_nats value after warmup. "
                             "If None, uses the fixed value from config (no annealing).")
    parser.add_argument("--free-nats-warmup", type=int, default=10,
                        help="Number of epochs to anneal free_nats from start to target.")

    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    replay_dir = Path(config["paths"]["replay_dir"])
    offline_cfg = config["offline_training"]
    train_manifest_path = Path(args.train_manifest) if args.train_manifest else replay_dir / "train_manifest.json"
    val_manifest_path = Path(args.val_manifest) if args.val_manifest else replay_dir / "val_manifest.json"
    train_paths = load_manifest(train_manifest_path)
    val_paths = load_manifest(val_manifest_path)
    # Distributed setup: initialise the process group so each GPU can communicate.
    # torchrun sets LOCAL_RANK, RANK, and WORLD_SIZE automatically.
    # Only rank 0 should print, save checkpoints, and log to W&B.
    use_distributed = bool(args.distributed)
    local_rank = 0
    global_rank = 0
    world_size = 1
    if use_distributed:
        import os
        from torch.distributed import init_process_group, destroy_process_group
        import torch.distributed as dist
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        global_rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    is_main_rank = (global_rank == 0)

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
        distributed=use_distributed,
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

    print(f"Loaded {len(train_paths)} train episodes from {train_manifest_path}")
    print(f"Loaded {len(val_paths)} val episodes from {val_manifest_path}")
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

    if use_distributed:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[rank {global_rank}] Training device: {device}")
    print(f"[rank {global_rank}] AMP enabled: {use_amp}")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    run_name = str(args.run_name or f"rssm_seq{int(offline_cfg['sequence_length'])}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    print(f"Run name: {run_name}")

    # W&B initialisation — only if not disabled.
    # config is passed so that all hyperparameters are stored in the run metadata,
    # making it easy to compare runs in the W&B sweep dashboard.
    use_wandb = not args.no_wandb
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                name=run_name,
                config={
                    "rssm": config.get("rssm", {}),
                    "offline_training": config.get("offline_training", {}),
                    "train_manifest": str(train_manifest_path),
                    "val_manifest": str(val_manifest_path),
                    "epochs": args.epochs,
                    "free_nats_start": args.free_nats_start,
                    "free_nats_target": args.free_nats_target,
                    "free_nats_warmup": args.free_nats_warmup,
                },
            )
            print(f"W&B run: {wandb.run.get_url()}", flush=True)
        except ImportError:
            print("WARNING: wandb not installed, disabling W&B logging. Install with: pip install wandb", flush=True)
            use_wandb = False

    # KL annealing setup: if start/target are provided via CLI, override config free_nats
    # each epoch. Otherwise use the fixed config value throughout (current behaviour).
    base_free_nats = float(offline_cfg["free_nats"])
    free_nats_start = float(args.free_nats_start) if args.free_nats_start is not None else None
    free_nats_target = float(args.free_nats_target) if args.free_nats_target is not None else None
    free_nats_warmup = int(args.free_nats_warmup)

    def _compute_free_nats(epoch: int) -> float:
        """Linearly anneal free_nats from start → target over warmup_epochs.
        After warmup, stay at target. If no annealing configured, return base value."""
        if free_nats_start is None or free_nats_target is None:
            return base_free_nats
        progress = min(1.0, epoch / max(1, free_nats_warmup))
        return free_nats_start + (free_nats_target - free_nats_start) * progress

    model = RSSMSequence(**config["rssm"]).to(device)

    if args.init_checkpoint is not None:
        checkpoint_path = Path(args.init_checkpoint)
        if not checkpoint_path.is_absolute():
            checkpoint_path = Path.cwd() / checkpoint_path
        checkpoint_payload = torch.load(checkpoint_path, map_location=device)
        missing, unexpected = model.load_state_dict(checkpoint_payload["model_state_dict"], strict=False)
        print(f"Initialized model from checkpoint: {checkpoint_path}")
        if missing:
            print(f"Missing checkpoint keys: {sorted(missing)}")
        if unexpected:
            print(f"Unexpected checkpoint keys: {sorted(unexpected)}")

    # DDP wrapping: each GPU trains an identical model, gradients are averaged across GPUs.
    # During inference (hallucination, metrics), we unwrap to the underlying module.
    if use_distributed:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # raw_model is the underlying RSSMSequence without DDP wrapper — used for inference.
    raw_model = model.module if use_distributed else model  # type: ignore[assignment]

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
        # Tell DistributedSampler which epoch we are on so it shuffles differently each time.
        # Without this, all epochs see the same data order — no benefit from shuffling.
        if use_distributed and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        free_nats_epoch = _compute_free_nats(epoch)
        if is_main_rank:
            print(f"Starting epoch {epoch + 1}/{int(args.epochs)} (free_nats={free_nats_epoch:.3f})", flush=True)
        epoch_start = time.perf_counter()
        batch_logger = None
        if use_wandb and is_main_rank:
            import wandb

            def _wandb_batch_logger(payload: dict[str, float]) -> None:
                global_batch = epoch * len(train_loader) + int(payload["batch/index"])
                wandb.log(payload, step=global_batch)

            batch_logger = _wandb_batch_logger

        metrics = train_world_model_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            free_nats=free_nats_epoch,
            kl_scale=float(offline_cfg["kl_scale"]),
            reward_scale=float(offline_cfg["reward_scale"]),
            log_every=int(args.log_every),
            use_amp=use_amp,
            grad_scaler=grad_scaler,
            batch_log_every=int(args.batch_log_every),
            batch_logger=batch_logger,
            telemetry_loss_scale=float(offline_cfg.get("telemetry_loss_scale", 0.0)),
            telemetry_weights=dict(offline_cfg.get("telemetry_weights", {})),
        )
        epoch_duration = time.perf_counter() - epoch_start
        epoch_durations.append(epoch_duration)
        elapsed = time.perf_counter() - run_start
        mean_epoch_time = sum(epoch_durations) / len(epoch_durations)
        remaining_epochs = int(args.epochs) - (epoch + 1)
        eta_seconds = remaining_epochs * mean_epoch_time
        batches_per_second = len(train_loader) / epoch_duration if epoch_duration > 0 else 0.0
        windows_per_second = len(train_loader.dataset) / epoch_duration if epoch_duration > 0 else 0.0
        epoch_global_step = (epoch + 1) * len(train_loader)
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

        # W&B: log train losses and throughput every epoch (main rank only).
        if use_wandb and is_main_rank:
            import wandb
            wandb_log = {
                "epoch": epoch + 1,
                "train/recon_loss": metrics["recon_loss"],
                "train/reward_loss": metrics["reward_loss"],
                "train/kl_loss": metrics["kl_loss"],
                "train/telemetry_loss": metrics["telemetry_loss"],
                "train/speed_loss": metrics["speed_loss"],
                "train/progress_delta_loss": metrics["progress_delta_loss"],
                "train/steer_loss": metrics["steer_loss"],
                "train/corner_angle_loss": metrics["corner_angle_loss"],
                "train/offtrack_loss": metrics["offtrack_loss"],
                "train/total_loss": metrics["total_loss"],
                "train/free_nats": free_nats_epoch,
                "perf/epoch_time_s": epoch_duration,
                "perf/batches_per_sec": batches_per_second,
                "perf/windows_per_sec": windows_per_second,
            }
            if device.type == "cuda":
                wandb_log["perf/max_gpu_mem_gb"] = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            wandb.log(wandb_log, step=epoch_global_step)

        if is_main_rank and ((epoch + 1) % int(args.save_every) == 0 or (epoch + 1) == int(args.epochs)):
            checkpoint_payload = {
                "model_state_dict": raw_model.state_dict(),
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
                model=raw_model,
                batch=val_batch,
                output_path=video_path,
                device=device,
                context_length=int(offline_cfg["hallucination_context"]),
                horizon=int(offline_cfg["hallucination_horizon"]),
                fps=float(offline_cfg.get("hallucination_video_fps", 10.0)),
            )
            latest_video_path = hallucination_root / "hallucination.mp4"
            save_hallucination_video(
                model=raw_model,
                batch=val_batch,
                output_path=latest_video_path,
                device=device,
                context_length=int(offline_cfg["hallucination_context"]),
                horizon=int(offline_cfg["hallucination_horizon"]),
                fps=float(offline_cfg.get("hallucination_video_fps", 10.0)),
            )
            print(f"Saved hallucination video to {video_path}", flush=True)
            print(f"Updated latest hallucination video at {latest_video_path}", flush=True)

            # Compute and log hallucination quality metrics (SSIM + MSE per step).
            # SSIM > 0.7 on val clips = high-quality hallucinations.
            # mse_per_step shows where the model starts to diverge (rising = degradation).
            halluc_metrics = compute_hallucination_metrics(
                model=raw_model,
                batch=val_batch,
                device=device,
                context_length=int(offline_cfg["hallucination_context"]),
                horizon=int(offline_cfg["hallucination_horizon"]),
            )
            print(
                f"Hallucination metrics: mean_mse={halluc_metrics['mean_mse']:.6f} "
                f"ssim={halluc_metrics['ssim']:.4f}",
                flush=True,
            )
            if use_wandb:
                import wandb
                hal_log: dict = {
                    "val/hallucination_mse": halluc_metrics["mean_mse"],
                    "val/hallucination_ssim": halluc_metrics["ssim"],
                }
                # Log per-step MSE as a W&B line chart so we can see where divergence happens.
                mse_steps = halluc_metrics["mse_per_step"]
                for step_i, mse_val in enumerate(mse_steps):
                    hal_log[f"val/mse_step_{step_i:03d}"] = mse_val
                # Upload hallucination video as W&B artifact.
                hal_log["val/hallucination_video"] = wandb.Video(str(video_path), fps=int(offline_cfg.get("hallucination_video_fps", 10)), format="mp4")
                wandb.log(hal_log, step=epoch_global_step)

            for curated in curated_batches:
                curated_name = str(curated["name"])
                curated_output = hallucination_dir / f"epoch_{epoch + 1:03d}_{curated_name}_side_by_side.mp4"
                save_side_by_side_hallucination_video(
                    model=raw_model,
                    batch=curated["batch"],  # type: ignore[arg-type]
                    output_path=curated_output,
                    device=device,
                    context_length=int(curated["context_length"]),
                    horizon=int(curated["horizon"]),
                    fps=float(offline_cfg.get("hallucination_video_fps", 10.0)),
                )
                print(f"Saved curated side-by-side video to {curated_output}", flush=True)

                # Compute SSIM on each curated clip — these are the "ground truth" eval clips
                # for turn fidelity. Track curated SSIM to see per-clip improvement.
                curated_metrics = compute_hallucination_metrics(
                    model=raw_model,
                    batch=curated["batch"],  # type: ignore[arg-type]
                    device=device,
                    context_length=int(curated["context_length"]),
                    horizon=int(curated["horizon"]),
                )
                print(
                    f"  [{curated_name}] mse={curated_metrics['mean_mse']:.6f} ssim={curated_metrics['ssim']:.4f}",
                    flush=True,
                )
                if use_wandb:
                    import wandb
                    wandb.log(
                        {
                            f"val/{curated_name}_mse": curated_metrics["mean_mse"],
                            f"val/{curated_name}_ssim": curated_metrics["ssim"],
                            f"val/{curated_name}_video": wandb.Video(
                                str(curated_output),
                                fps=int(offline_cfg.get("hallucination_video_fps", 10)),
                                format="mp4",
                            ),
                        },
                        step=epoch_global_step,
                    )

    if use_wandb and is_main_rank:
        import wandb
        wandb.finish()

    if use_distributed:
        from torch.distributed import destroy_process_group
        destroy_process_group()


if __name__ == "__main__":
    main()
