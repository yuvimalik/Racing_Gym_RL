import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import yaml

from world_model.collector import _build_maneuver_actions, save_collection_manifest
from world_model.control import FrozenWorldModel, LatentActor, LatentCritic, discounted_bootstrap_returns, imagine_with_actor
from world_model.models import Decoder, Encoder, RSSMCell, RSSMSequence
from world_model.replay import EpisodeReplay, ReplayWriter, SequenceReplayDataset
from world_model.control_training import save_latent_actor_compare_video, train_latent_actor_critic_epoch
from world_model.training import build_curated_eval_batch, build_replay_loader, save_side_by_side_hallucination_video, save_video


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

    def test_sequence_dataset_window_stride_reduces_index_count(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            writer = ReplayWriter(temp_dir, split="train")
            episode = EpisodeReplay(
                observations_uint8=np.random.randint(0, 255, size=(60, 96, 96, 3), dtype=np.uint8),
                actions=np.random.randn(60, 3).astype(np.float32),
                rewards=np.random.randn(60).astype(np.float32),
                dones=np.zeros(60, dtype=np.bool_),
                truncated=np.zeros(60, dtype=np.bool_),
                action_mean_raw=np.random.randn(60, 3).astype(np.float32),
                action_noisy_raw=np.random.randn(60, 3).astype(np.float32),
                noise=np.random.randn(60, 3).astype(np.float32),
                metadata={"episode_id": 0},
            )
            writer.save_episode(episode, episode_id=0)
            dataset_stride_1 = SequenceReplayDataset(
                episode_paths=sorted((Path(temp_dir) / "train").glob("*.npz")),
                sequence_length=50,
                window_stride=1,
            )
            dataset_stride_4 = SequenceReplayDataset(
                episode_paths=sorted((Path(temp_dir) / "train").glob("*.npz")),
                sequence_length=50,
                window_stride=4,
            )
            self.assertEqual(len(dataset_stride_1), 11)
            self.assertEqual(len(dataset_stride_4), 3)

    def test_maneuver_generator_produces_sustained_sequences(self):
        rng = np.random.default_rng(123)
        maneuver = _build_maneuver_actions(
            family="steer_then_countersteer_recovery",
            rng=rng,
            regime={"throttle_scale": 0.8, "steering_scale": 1.0, "brake_scale": 1.0, "transition_steps": 4},
        )
        self.assertEqual(maneuver.actions.ndim, 2)
        self.assertEqual(maneuver.actions.shape[1], 3)
        self.assertGreater(maneuver.actions.shape[0], 20)
        self.assertTrue(maneuver.recovery_focused)
        unique_rows = np.unique(np.round(maneuver.actions, 3), axis=0)
        self.assertGreater(unique_rows.shape[0], 3)

    def test_video_writer_smoke(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            frames = np.random.randint(0, 255, size=(5, 96, 96, 3), dtype=np.uint8)
            path = save_video(frames, Path(temp_dir) / "test.mp4", fps=5.0)
            self.assertTrue(path.is_file())

    def test_build_curated_eval_batch_shapes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            writer = ReplayWriter(temp_dir, split="train")
            episode = EpisodeReplay(
                observations_uint8=np.random.randint(0, 255, size=(180, 96, 96, 3), dtype=np.uint8),
                actions=np.random.randn(180, 3).astype(np.float32),
                rewards=np.random.randn(180).astype(np.float32),
                dones=np.zeros(180, dtype=np.bool_),
                truncated=np.zeros(180, dtype=np.bool_),
                action_mean_raw=np.random.randn(180, 3).astype(np.float32),
                action_noisy_raw=np.random.randn(180, 3).astype(np.float32),
                noise=np.random.randn(180, 3).astype(np.float32),
                metadata={"episode_id": 0},
            )
            path = writer.save_episode(episode, episode_id=0)
            batch = build_curated_eval_batch(path, start_index=10, context_length=50, horizon=100)
            self.assertEqual(tuple(batch["images"].shape), (1, 150, 3, 96, 96))
            self.assertEqual(tuple(batch["actions"].shape), (1, 150, 3))

    def test_side_by_side_hallucination_video_smoke(self):
        model = RSSMSequence()
        batch = {
            "images": torch.rand(1, 100, 3, 96, 96),
            "actions": torch.rand(1, 100, 3),
            "rewards": torch.rand(1, 100, 1),
            "dones": torch.zeros(1, 100),
            "is_first": torch.zeros(1, 100, dtype=torch.bool),
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = save_side_by_side_hallucination_video(
                model=model,
                batch=batch,
                output_path=Path(temp_dir) / "side_by_side.mp4",
                device="cpu",
                context_length=50,
                horizon=50,
            )
            self.assertTrue(path.is_file())

    def test_replay_loader_accepts_worker_configuration(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            writer = ReplayWriter(temp_dir, split="train")
            episode = EpisodeReplay(
                observations_uint8=np.random.randint(0, 255, size=(60, 96, 96, 3), dtype=np.uint8),
                actions=np.random.randn(60, 3).astype(np.float32),
                rewards=np.random.randn(60).astype(np.float32),
                dones=np.zeros(60, dtype=np.bool_),
                truncated=np.zeros(60, dtype=np.bool_),
                action_mean_raw=np.random.randn(60, 3).astype(np.float32),
                action_noisy_raw=np.random.randn(60, 3).astype(np.float32),
                noise=np.random.randn(60, 3).astype(np.float32),
                metadata={"episode_id": 0},
            )
            writer.save_episode(episode, episode_id=0)
            loader = build_replay_loader(
                episode_paths=sorted((Path(temp_dir) / "train").glob("*.npz")),
                sequence_length=50,
                batch_size=4,
                shuffle=False,
                window_stride=1,
                num_workers=0,
                pin_memory=False,
                persistent_workers=False,
                prefetch_factor=2,
            )
            batch = next(iter(loader))
            self.assertEqual(tuple(batch["images"].shape), (4, 50, 3, 96, 96))
            self.assertEqual(tuple(batch["actions"].shape), (4, 50, 3))

    def test_frozen_world_model_encodes_context_and_flattens_state(self):
        model = RSSMSequence()
        frozen = FrozenWorldModel(model)
        images = torch.rand(2, 4, 3, 96, 96)
        actions = torch.rand(2, 4, 3)
        state = frozen.encode_context(images=images, actions=actions)
        features = frozen.flatten_state(state)
        self.assertEqual(tuple(state.deterministic.shape), (2, 512))
        self.assertEqual(tuple(state.stochastic.shape), (2, 32))
        self.assertEqual(tuple(features.shape), (2, 544))
        self.assertTrue(all(not parameter.requires_grad for parameter in frozen.model.parameters()))

    def test_latent_actor_and_critic_shapes(self):
        actor = LatentActor(latent_dim=544, action_dim=3, hidden_dim=128)
        critic = LatentCritic(latent_dim=544, hidden_dim=128)
        latent = torch.rand(5, 544)
        action = actor(latent)
        value = critic(latent)
        self.assertEqual(tuple(action.shape), (5, 3))
        self.assertEqual(tuple(value.shape), (5, 1))
        self.assertTrue(torch.all(action[:, 0] >= -1.0).item())
        self.assertTrue(torch.all(action[:, 0] <= 1.0).item())
        self.assertTrue(torch.all(action[:, 1:] >= 0.0).item())
        self.assertTrue(torch.all(action[:, 1:] <= 1.0).item())

    def test_imagined_rollout_gradients_skip_frozen_world_model(self):
        model = RSSMSequence()
        frozen = FrozenWorldModel(model)
        actor = LatentActor(latent_dim=frozen.latent_dim, action_dim=frozen.action_dim, hidden_dim=128)
        critic = LatentCritic(latent_dim=frozen.latent_dim, hidden_dim=128)
        images = torch.rand(2, 3, 3, 96, 96)
        actions = torch.rand(2, 3, 3)
        initial_state = frozen.encode_context(images=images, actions=actions)
        rollout = imagine_with_actor(frozen, actor, initial_state, horizon=4)
        rewards = torch.stack(rollout.rewards, dim=1)
        bootstrap = critic(rollout.final_feature.detach()).squeeze(-1).detach()
        actor_returns = discounted_bootstrap_returns(rewards, bootstrap, discount=0.99)
        actor_loss = -actor_returns.mean()
        actor_loss.backward()
        actor_grad_norm = sum(
            float(parameter.grad.abs().sum().item())
            for parameter in actor.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(actor_grad_norm, 0.0)
        self.assertTrue(all(parameter.grad is None for parameter in frozen.model.parameters()))

    def test_latent_control_epoch_smoke(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            writer = ReplayWriter(temp_dir, split="train")
            for episode_id in range(2):
                episode = EpisodeReplay(
                    observations_uint8=np.random.randint(0, 255, size=(40, 96, 96, 3), dtype=np.uint8),
                    actions=np.random.randn(40, 3).astype(np.float32),
                    rewards=np.random.randn(40).astype(np.float32),
                    dones=np.zeros(40, dtype=np.bool_),
                    truncated=np.zeros(40, dtype=np.bool_),
                    action_mean_raw=np.random.randn(40, 3).astype(np.float32),
                    action_noisy_raw=np.random.randn(40, 3).astype(np.float32),
                    noise=np.random.randn(40, 3).astype(np.float32),
                    metadata={"episode_id": episode_id},
                )
                writer.save_episode(episode, episode_id=episode_id)

            loader = build_replay_loader(
                episode_paths=sorted((Path(temp_dir) / "train").glob("*.npz")),
                sequence_length=20,
                batch_size=4,
                shuffle=False,
                window_stride=4,
                num_workers=0,
                pin_memory=False,
                persistent_workers=False,
                prefetch_factor=2,
            )
            frozen = FrozenWorldModel(RSSMSequence())
            actor = LatentActor(latent_dim=frozen.latent_dim, action_dim=frozen.action_dim, hidden_dim=128)
            critic = LatentCritic(latent_dim=frozen.latent_dim, hidden_dim=128)
            metrics = train_latent_actor_critic_epoch(
                world_model=frozen,
                actor=actor,
                critic=critic,
                loader=loader,
                actor_optimizer=torch.optim.Adam(actor.parameters(), lr=1e-3),
                critic_optimizer=torch.optim.Adam(critic.parameters(), lr=1e-3),
                device="cpu",
                context_length=20,
                imagination_horizon=6,
                discount=0.99,
                action_l2_scale=1e-4,
                log_every=0,
            )
            self.assertIn("actor_loss", metrics)
            self.assertIn("critic_loss", metrics)
            self.assertIn("imagined_reward_mean", metrics)
            self.assertTrue(all(parameter.grad is None for parameter in frozen.model.parameters()))

    def test_latent_actor_compare_video_smoke(self):
        frozen = FrozenWorldModel(RSSMSequence())
        observations = np.random.randint(0, 255, size=(70, 96, 96, 3), dtype=np.uint8)
        actions = np.random.randn(70, 3).astype(np.float32)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = save_latent_actor_compare_video(
                world_model=frozen,
                observations_uint8=observations,
                actions=actions,
                output_path=Path(temp_dir) / "compare.mp4",
                device="cpu",
                context_length=20,
                horizon=30,
                fps=8.0,
            )
            self.assertTrue(path.is_file())

    def test_manifest_contains_summary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            episode_paths = [Path(temp_dir) / "train" / "episode_000001.npz"]
            manifest_path = save_collection_manifest(
                output_dir=temp_dir,
                split="normal_train",
                episode_paths=episode_paths,
                summary={"mean_progress": 0.42, "offtrack_rate": 0.5},
            )
            payload = manifest_path.read_text(encoding="utf-8")
            self.assertIn("\"summary\"", payload)
            self.assertIn("normal_train", payload)

    def test_world_model_config_has_three_regimes_and_two_directions(self):
        with open("config/world_model_config.yaml", "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        automatic = config["collector"]["automatic"]
        self.assertEqual(sorted(automatic["regimes"].keys()), ["high", "medium", "slow"])
        self.assertEqual(sorted(automatic["directions"]), ["CCW", "CW"])

    def test_world_model_config_has_manual_collector(self):
        with open("config/world_model_config.yaml", "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        manual = config["collector"]["manual"]
        self.assertIn("target_frames", manual)
        self.assertIn("direction", manual)
        self.assertIn("regime", manual)

    def test_world_model_config_has_local_gpu_training_knobs(self):
        with open("config/world_model_config.yaml", "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        offline = config["offline_training"]
        self.assertEqual(offline["batch_size"], 16)
        self.assertEqual(offline["hallucination_horizon"], 100)
        self.assertEqual(offline["hallucination_video_fps"], 8.0)
        self.assertEqual(offline["window_stride"], 4)
        self.assertEqual(offline["num_workers"], 4)
        self.assertTrue(offline["pin_memory"])
        self.assertTrue(offline["persistent_workers"])
        self.assertEqual(offline["prefetch_factor"], 2)
        self.assertTrue(offline["use_amp"])
        self.assertEqual(len(offline["curated_eval_clips"]), 3)

    def test_world_model_config_has_control_training_knobs(self):
        with open("config/world_model_config.yaml", "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        control = config["control_training"]
        self.assertEqual(control["context_length"], 20)
        self.assertEqual(control["imagination_horizon"], 12)
        self.assertEqual(control["batch_size"], 32)
        self.assertEqual(control["eval_episodes"], 3)
        self.assertEqual(control["window_stride"], 4)
        self.assertTrue(control["pin_memory"])
        self.assertTrue(control["record_eval_video"])
        self.assertTrue(control["record_compare_video"])
        self.assertEqual(control["compare_context_length"], 20)
        self.assertEqual(control["compare_horizon"], 50)


if __name__ == "__main__":
    unittest.main()
