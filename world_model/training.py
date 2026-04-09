from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from world_model.losses import free_bits_kl, reconstruction_loss, reward_loss
from world_model.models import Decoder, Encoder, RSSMSequence
from world_model.replay import ReplayWriter, SequenceReplayDataset


@dataclass
class AutoencoderArtifacts:
    model_path: Path
    grid_path: Path
    final_loss: float


class VisionAutoencoder(torch.nn.Module):
    def __init__(self, embedding_dim: int = 512):
        super().__init__()
        self.encoder = Encoder(embedding_dim=embedding_dim)
        self.decoder = Decoder(input_dim=embedding_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(images))


def save_image_grid(images: torch.Tensor, reconstructions: torch.Tensor, path: str | Path) -> Path:
    images_np = images.detach().cpu().numpy()
    recon_np = reconstructions.detach().cpu().numpy()
    tiles = []
    for original, recon in zip(images_np, recon_np):
        tiles.append(np.concatenate([original.transpose(1, 2, 0), recon.transpose(1, 2, 0)], axis=1))
    grid = np.concatenate(tiles, axis=0)
    grid = np.clip(grid * 255.0, 0.0, 255.0).astype(np.uint8)
    output = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), output)
    return path


def train_autoencoder_sanity(
    images: torch.Tensor,
    output_dir: str | Path,
    device: str | torch.device = "cpu",
    epochs: int = 1000,
    learning_rate: float = 1e-3,
) -> AutoencoderArtifacts:
    device = torch.device(device)
    model = VisionAutoencoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    images = images.to(device)

    for _ in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        recon = model(images)
        loss = reconstruction_loss(recon, images)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        recon = model(images)
        final_loss = float(reconstruction_loss(recon, images).item())

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "autoencoder_sanity.pt"
    grid_path = output_dir / "autoencoder_recon_grid.png"
    torch.save({"model_state_dict": model.state_dict(), "final_loss": final_loss}, model_path)
    save_image_grid(images, recon, grid_path)
    return AutoencoderArtifacts(model_path=model_path, grid_path=grid_path, final_loss=final_loss)


def build_replay_loader(
    episode_paths: Iterable[str | Path],
    sequence_length: int = 50,
    batch_size: int = 8,
    shuffle: bool = True,
    window_stride: int = 1,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    prefetch_factor: int | None = None,
) -> DataLoader:
    dataset = SequenceReplayDataset(
        list(episode_paths),
        sequence_length=sequence_length,
        normalize=True,
        window_stride=window_stride,
    )
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
    }
    if int(num_workers) > 0:
        loader_kwargs["persistent_workers"] = bool(persistent_workers)
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = int(prefetch_factor)
    return DataLoader(dataset, **loader_kwargs)


def train_world_model_epoch(
    model: RSSMSequence,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str | torch.device,
    free_nats: float = 1.0,
    kl_scale: float = 1.0,
    reward_scale: float = 1.0,
    log_every: int = 0,
    use_amp: bool = False,
    grad_scaler: torch.amp.GradScaler | None = None,
) -> dict[str, float]:
    device = torch.device(device)
    device_type = "cuda" if device.type == "cuda" else "cpu"
    non_blocking = device.type == "cuda"
    model.train()
    totals = {"recon_loss": 0.0, "reward_loss": 0.0, "kl_loss": 0.0, "total_loss": 0.0}
    num_batches = 0

    for batch_index, batch in enumerate(loader, start=1):
        images = batch["images"].to(device, non_blocking=non_blocking)
        actions = batch["actions"].to(device, non_blocking=non_blocking)
        rewards = batch["rewards"].to(device, non_blocking=non_blocking)
        is_first = batch["is_first"].to(device, non_blocking=non_blocking)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=bool(use_amp and device.type == "cuda")):
            output = model(images=images, actions=actions, is_first=is_first)
            recon = reconstruction_loss(output.reconstruction, images)
            rew = reward_loss(output.reward, rewards)
            kl = free_bits_kl(
                posterior_mean=output.posterior_mean,
                posterior_std=output.posterior_std,
                prior_mean=output.prior_mean,
                prior_std=output.prior_std,
                free_nats=free_nats,
            )
            total = recon + (reward_scale * rew) + (kl_scale * kl)

        if grad_scaler is not None and bool(use_amp and device.type == "cuda"):
            grad_scaler.scale(total).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            total.backward()
            optimizer.step()

        totals["recon_loss"] += float(recon.item())
        totals["reward_loss"] += float(rew.item())
        totals["kl_loss"] += float(kl.item())
        totals["total_loss"] += float(total.item())
        num_batches += 1

        if log_every > 0 and (batch_index == 1 or batch_index % log_every == 0):
            print(
                "[RSSM][batch "
                f"{batch_index}/{len(loader)}] "
                f"recon={recon.item():.6f} "
                f"reward={rew.item():.6f} "
                f"kl={kl.item():.6f} "
                f"total={total.item():.6f}",
                flush=True,
            )

    if num_batches == 0:
        raise ValueError("Replay loader produced zero batches.")
    return {key: value / num_batches for key, value in totals.items()}


def hallucinate_future(
    model: RSSMSequence,
    images: torch.Tensor,
    actions: torch.Tensor,
    context_length: int = 50,
    horizon: int = 50,
) -> torch.Tensor:
    if images.shape[1] < context_length + horizon:
        raise ValueError("Need at least context_length + horizon frames.")

    batch_size = images.shape[0]
    state = model.initial_state(batch_size=batch_size, device=images.device)
    prev_action = torch.zeros(batch_size, model.action_dim, device=images.device, dtype=actions.dtype)

    for time_index in range(context_length):
        step = model.cell.observe_step(prev_state=state, prev_action=prev_action, image_t=images[:, time_index])
        state = step.state
        prev_action = actions[:, time_index]

    reconstructions = []
    for time_index in range(context_length, context_length + horizon):
        step = model.cell.imagine_step(prev_state=state, prev_action=prev_action)
        reconstructions.append(step.reconstruction)
        state = step.state
        prev_action = actions[:, time_index]

    return torch.stack(reconstructions, dim=1)


def save_video(frames_thwc_uint8: np.ndarray, path: str | Path, fps: float = 10.0) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _, height, width = frames_thwc_uint8.shape[:3]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {path}")
    try:
        for frame in frames_thwc_uint8:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    return path


def build_curated_eval_batch(
    episode_path: str | Path,
    start_index: int,
    context_length: int,
    horizon: int,
    device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    episode = ReplayWriter.load_episode(episode_path)
    total_length = int(context_length) + int(horizon)
    start_index = int(start_index)
    end_index = start_index + total_length
    if end_index > int(episode.observations_uint8.shape[0]):
        raise ValueError(
            f"Curated clip {episode_path} start_index={start_index} exceeds episode length "
            f"{int(episode.observations_uint8.shape[0])} for required window {total_length}."
        )

    images = torch.from_numpy(episode.observations_uint8[start_index:end_index]).permute(0, 3, 1, 2).float() / 255.0
    actions = torch.from_numpy(episode.actions[start_index:end_index]).float()
    rewards = torch.from_numpy(episode.rewards[start_index:end_index]).float().unsqueeze(-1)
    dones = torch.from_numpy(episode.dones[start_index:end_index].astype(np.float32))
    is_first = torch.zeros(total_length, dtype=torch.bool)
    if start_index == 0:
        is_first[0] = True

    batch = {
        "images": images.unsqueeze(0).to(device),
        "actions": actions.unsqueeze(0).to(device),
        "rewards": rewards.unsqueeze(0).to(device),
        "dones": dones.unsqueeze(0).to(device),
        "is_first": is_first.unsqueeze(0).to(device),
    }
    return batch


def save_hallucination_video(
    model: RSSMSequence,
    batch: dict[str, torch.Tensor],
    output_path: str | Path,
    device: str | torch.device,
    context_length: int = 50,
    horizon: int = 50,
    fps: float = 10.0,
) -> Path:
    device = torch.device(device)
    model.eval()
    with torch.no_grad():
        images = batch["images"].to(device)
        actions = batch["actions"].to(device)
        imagined = hallucinate_future(model, images, actions, context_length=context_length, horizon=horizon)
    frames = imagined[0].detach().cpu().permute(0, 2, 3, 1).numpy()
    frames_uint8 = np.clip(frames * 255.0, 0.0, 255.0).astype(np.uint8)
    return save_video(frames_uint8, output_path, fps=fps)


def save_side_by_side_hallucination_video(
    model: RSSMSequence,
    batch: dict[str, torch.Tensor],
    output_path: str | Path,
    device: str | torch.device,
    context_length: int = 50,
    horizon: int = 50,
    fps: float = 10.0,
) -> Path:
    device = torch.device(device)
    model.eval()
    with torch.no_grad():
        images = batch["images"].to(device)
        actions = batch["actions"].to(device)
        imagined = hallucinate_future(model, images, actions, context_length=context_length, horizon=horizon)
    real_frames = images[0, context_length : context_length + horizon].detach().cpu().permute(0, 2, 3, 1).numpy()
    imagined_frames = imagined[0].detach().cpu().permute(0, 2, 3, 1).numpy()
    real_uint8 = np.clip(real_frames * 255.0, 0.0, 255.0).astype(np.uint8)
    imagined_uint8 = np.clip(imagined_frames * 255.0, 0.0, 255.0).astype(np.uint8)
    side_by_side = np.concatenate([real_uint8, imagined_uint8], axis=2)
    return save_video(side_by_side, output_path, fps=fps)
