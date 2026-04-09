from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class EpisodeReplay:
    observations_uint8: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    truncated: np.ndarray
    action_mean_raw: np.ndarray
    action_noisy_raw: np.ndarray
    noise: np.ndarray
    metadata: dict[str, Any]


class ReplayWriter:
    """Stores episode chunks as compressed NPZ files."""

    def __init__(self, root_dir: str | Path, split: str = "train"):
        self.root_dir = Path(root_dir)
        self.split_dir = self.root_dir / str(split)
        self.split_dir.mkdir(parents=True, exist_ok=True)

    def save_episode(self, episode: EpisodeReplay, episode_id: int) -> Path:
        path = self.split_dir / f"episode_{episode_id:06d}.npz"
        np.savez_compressed(
            path,
            observations_uint8=episode.observations_uint8.astype(np.uint8),
            actions=episode.actions.astype(np.float32),
            rewards=episode.rewards.astype(np.float32),
            dones=episode.dones.astype(np.bool_),
            truncated=episode.truncated.astype(np.bool_),
            action_mean_raw=episode.action_mean_raw.astype(np.float32),
            action_noisy_raw=episode.action_noisy_raw.astype(np.float32),
            noise=episode.noise.astype(np.float32),
            metadata_json=np.asarray(json.dumps(episode.metadata), dtype=np.unicode_),
        )
        return path

    @staticmethod
    def load_episode(path: str | Path) -> EpisodeReplay:
        with np.load(Path(path), allow_pickle=False) as data:
            return EpisodeReplay(
                observations_uint8=data["observations_uint8"],
                actions=data["actions"],
                rewards=data["rewards"],
                dones=data["dones"].astype(np.bool_),
                truncated=data["truncated"].astype(np.bool_),
                action_mean_raw=data["action_mean_raw"],
                action_noisy_raw=data["action_noisy_raw"],
                noise=data["noise"],
                metadata=json.loads(str(data["metadata_json"])),
            )


class SequenceReplayDataset(Dataset):
    """Returns fixed-length windows without crossing episode boundaries."""

    def __init__(
        self,
        episode_paths: list[str | Path],
        sequence_length: int = 50,
        normalize: bool = True,
        window_stride: int = 1,
    ):
        self.episode_paths = [Path(path) for path in episode_paths]
        self.sequence_length = int(sequence_length)
        self.normalize = bool(normalize)
        self.window_stride = max(1, int(window_stride))
        self._episodes = [ReplayWriter.load_episode(path) for path in self.episode_paths]
        self._index: list[tuple[int, int]] = []

        for episode_index, episode in enumerate(self._episodes):
            length = int(episode.observations_uint8.shape[0])
            if length < self.sequence_length:
                continue
            for start_index in range(0, length - self.sequence_length + 1, self.window_stride):
                self._index.append((episode_index, start_index))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        episode_index, start_index = self._index[index]
        episode = self._episodes[episode_index]
        end_index = start_index + self.sequence_length

        images = torch.from_numpy(episode.observations_uint8[start_index:end_index]).permute(0, 3, 1, 2).float()
        if self.normalize:
            images = images / 255.0
        actions = torch.from_numpy(episode.actions[start_index:end_index]).float()
        rewards = torch.from_numpy(episode.rewards[start_index:end_index]).float().unsqueeze(-1)
        dones = torch.from_numpy(episode.dones[start_index:end_index].astype(np.float32))
        is_first = torch.zeros(self.sequence_length, dtype=torch.bool)
        if start_index == 0:
            is_first[0] = True

        return {
            "images": images,
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
            "is_first": is_first,
        }
