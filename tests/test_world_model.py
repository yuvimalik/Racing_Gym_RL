import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from world_model.collector import apply_drunk_expert_noise
from world_model.models import Decoder, Encoder, RSSMCell, RSSMSequence
from world_model.replay import EpisodeReplay, ReplayWriter, SequenceReplayDataset
from world_model.training import save_video


class WorldModelTests(unittest.TestCase):
    def test_encoder_decoder_shapes(self):
        encoder = Encoder()
        decoder = Decoder(input_dim=512)
        images = torch.rand(4, 3, 96, 96)
        embedding = encoder(images)
        reconstruction = decoder(embedding)
        self.assertEqual(tuple(embedding.shape), (4, 512))
        self.assertEqual(tuple(reconstruction.shape), (4, 3, 96, 96))
        self.assertTrue(torch.all(reconstruction >= 0.0).item())
        self.assertTrue(torch.all(reconstruction <= 1.0).item())

    def test_rssm_cell_shapes_and_finite_values(self):
        cell = RSSMCell()
        state = cell.initial_state(batch_size=3, device="cpu")
        action = torch.zeros(3, 3)
        image = torch.rand(3, 3, 96, 96)
        output = cell.observe_step(state, action, image)
        self.assertEqual(tuple(output.state.deterministic.shape), (3, 512))
        self.assertEqual(tuple(output.state.stochastic.shape), (3, 32))
        self.assertEqual(tuple(output.reward.shape), (3, 1))
        self.assertEqual(tuple(output.reconstruction.shape), (3, 3, 96, 96))
        self.assertTrue(torch.isfinite(output.prior.mean).all().item())
        self.assertTrue(torch.isfinite(output.posterior.std).all().item())

    def test_rssm_sequence_shapes(self):
        model = RSSMSequence()
        images = torch.rand(2, 5, 3, 96, 96)
        actions = torch.rand(2, 5, 3)
        is_first = torch.zeros(2, 5, dtype=torch.bool)
        is_first[:, 0] = True
        output = model(images=images, actions=actions, is_first=is_first)
        self.assertEqual(tuple(output.deterministic.shape), (2, 5, 512))
        self.assertEqual(tuple(output.stochastic.shape), (2, 5, 32))
        self.assertEqual(tuple(output.reward.shape), (2, 5, 1))
        self.assertEqual(tuple(output.reconstruction.shape), (2, 5, 3, 96, 96))

    def test_replay_writer_round_trip(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            episode = EpisodeReplay(
                observations_uint8=np.random.randint(0, 255, size=(12, 96, 96, 3), dtype=np.uint8),
                actions=np.random.randn(12, 3).astype(np.float32),
                rewards=np.random.randn(12).astype(np.float32),
                dones=np.zeros(12, dtype=np.bool_),
                truncated=np.zeros(12, dtype=np.bool_),
                action_mean_raw=np.random.randn(12, 3).astype(np.float32),
                action_noisy_raw=np.random.randn(12, 3).astype(np.float32),
                noise=np.random.randn(12, 3).astype(np.float32),
                metadata={"episode_id": 7},
            )
            writer = ReplayWriter(temp_dir, split="train")
            saved = writer.save_episode(episode, episode_id=7)
            loaded = ReplayWriter.load_episode(saved)
            self.assertEqual(loaded.metadata["episode_id"], 7)
            self.assertEqual(tuple(loaded.observations_uint8.shape), (12, 96, 96, 3))

    def test_sequence_dataset_windows_do_not_cross_episodes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            writer = ReplayWriter(temp_dir, split="train")
            for episode_id in range(2):
                episode = EpisodeReplay(
                    observations_uint8=np.random.randint(0, 255, size=(60, 96, 96, 3), dtype=np.uint8),
                    actions=np.random.randn(60, 3).astype(np.float32),
                    rewards=np.random.randn(60).astype(np.float32),
                    dones=np.zeros(60, dtype=np.bool_),
                    truncated=np.zeros(60, dtype=np.bool_),
                    action_mean_raw=np.random.randn(60, 3).astype(np.float32),
                    action_noisy_raw=np.random.randn(60, 3).astype(np.float32),
                    noise=np.random.randn(60, 3).astype(np.float32),
                    metadata={"episode_id": episode_id},
                )
                writer.save_episode(episode, episode_id=episode_id)

            dataset = SequenceReplayDataset(
                episode_paths=sorted((Path(temp_dir) / "train").glob("*.npz")),
                sequence_length=50,
            )
            sample = dataset[0]
            self.assertEqual(tuple(sample["images"].shape), (50, 3, 96, 96))
            self.assertEqual(tuple(sample["actions"].shape), (50, 3))
            self.assertTrue(bool(sample["is_first"][0].item()))

    def test_drunk_expert_noise_contract(self):
        rng = np.random.default_rng(123)
        mean_raw = np.array([0.0, 0.5, -0.3], dtype=np.float32)
        noise_std = np.array([0.6, 0.35, 0.15], dtype=np.float32)
        action_low = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
        action_high = np.array([1.0, 1.0, 1.0], dtype=np.float32)

        def raw_to_env_action(raw_tensor):
            mapped = raw_tensor.clone()
            mapped[..., 0] = torch.tanh(mapped[..., 0])
            mapped[..., 1] = (torch.tanh(mapped[..., 1]) + 1.0) / 2.0
            mapped[..., 2] = (torch.tanh(mapped[..., 2]) + 1.0) / 2.0
            return mapped

        noisy_raw, noise, env_action = apply_drunk_expert_noise(
            action_mean_raw=mean_raw,
            noise_std=noise_std,
            rng=rng,
            action_low=action_low,
            action_high=action_high,
            raw_to_env_action=raw_to_env_action,
        )
        np.testing.assert_allclose(noisy_raw - mean_raw, noise, atol=1e-6)
        self.assertTrue(np.all(env_action >= action_low))
        self.assertTrue(np.all(env_action <= action_high))

    def test_video_writer_smoke(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            frames = np.random.randint(0, 255, size=(5, 96, 96, 3), dtype=np.uint8)
            path = save_video(frames, Path(temp_dir) / "test.mp4", fps=5.0)
            self.assertTrue(path.is_file())


if __name__ == "__main__":
    unittest.main()
