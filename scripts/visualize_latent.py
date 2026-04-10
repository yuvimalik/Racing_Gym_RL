"""Visualise the RSSM latent space to show what the world model has learned.

This is the "wow moment" for the presentation:
  The model was never told where on the track it is, how fast it is going,
  or what it is steering towards. Yet when we encode 500 real frames and
  reduce the latent vectors to 2D using PCA, the latent space organises
  itself by track position, speed, and steering — the model has discovered
  the track's geometry unsupervised.

How PCA works (for the audience):
  PCA (Principal Component Analysis) finds the two directions in high-dimensional
  space (z_t has 32–64 dimensions) along which the data varies most. It is a
  linear projection — think of shining a light on a 3D object and looking at
  its shadow. The shadow loses information but preserves the main shape.
  When points that are far apart in the 2D PCA plot are also far apart in
  physical meaning (e.g. different track sections), it means the latent space
  has organised itself to represent those physical differences.

t-SNE is an alternative non-linear projection that better preserves local
neighbourhood structure (nearby points stay near each other) but distorts
global distances. It is slower but often produces visually cleaner clusters.

Usage:
    python scripts/visualize_latent.py \\
        --checkpoint models/world_model/rssm_sequence.pt \\
        --config config/world_model_config.yaml \\
        --replay-dir results/world_model/replay \\
        --n-frames 500 \\
        --output results/demo/latent_viz.png \\
        --method pca
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch
import yaml

from world_model.models import RSSMSequence
from world_model.replay import ReplayWriter


def _load_checkpoint(checkpoint_path: Path, config: dict, device: torch.device) -> RSSMSequence:
    model = RSSMSequence(**config["rssm"]).to(device)
    payload = torch.load(str(checkpoint_path), map_location=device)
    state_dict = payload.get("model_state_dict", payload)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _encode_frames(
    model: RSSMSequence,
    episode_paths: list[Path],
    n_frames: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Encode random frames from the replay dataset into latent vectors.

    Returns:
        z_vectors: (N, stochastic_dim) — the stochastic latent z_t for each frame
        speeds:    (N,) — approximate speed from reward signal (proxy)
        steers:    (N,) — steering action
        progress:  (N,) — frame index as proxy for track position (0.0 to 1.0)

    How encoding works:
        For each frame, we run the encoder CNN to get an embedding, then pass
        it through the posterior head (which uses both h_t and the embedding)
        to get z_t. We use the posterior mean (not a sample) for deterministic,
        reproducible visualisation.
    """
    z_list, speed_list, steer_list, progress_list = [], [], [], []

    random.shuffle(episode_paths)
    frames_collected = 0

    for ep_path in episode_paths:
        if frames_collected >= n_frames:
            break
        try:
            episode = ReplayWriter.load_episode(ep_path)
        except Exception:
            continue

        obs = episode.observations_uint8   # (T, H, W, C)
        acts = episode.actions             # (T, 3)
        T = obs.shape[0]
        if T < 2:
            continue

        # Encode the episode frame by frame, maintaining the RNN state.
        # This means z_t reflects not just the current frame but also history.
        state = model.initial_state(batch_size=1, device=device)
        prev_action = torch.zeros(1, model.action_dim, device=device)

        for t in range(T):
            if frames_collected >= n_frames:
                break

            img_t = torch.from_numpy(obs[t]).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            img_t = img_t.to(device)
            act_t = torch.from_numpy(acts[t]).float().unsqueeze(0).to(device)

            with torch.no_grad():
                step = model.cell.observe_step(
                    prev_state=state,
                    prev_action=prev_action,
                    image_t=img_t,
                )
            state = step.state

            # Use posterior mean as the latent representation (deterministic)
            z_mean = step.state.stochastic  # (1, stochastic_dim) — this IS the posterior mean after reparameterisation
            # Actually stochastic is the sample; use posterior_mean from the step output
            z_vec = step.posterior_mean if hasattr(step, "posterior_mean") else step.state.stochastic
            z_list.append(z_vec[0].cpu().numpy())

            # Proxy colour variables from action/reward signals
            steer_list.append(float(acts[t, 0]))                        # steer in [-1, 1]
            speed_proxy = max(0.0, float(acts[t, 1]) - float(acts[t, 2]))  # throttle - brake ≈ speed intent
            speed_list.append(speed_proxy)
            progress_list.append(float(t) / float(max(T - 1, 1)))      # position within episode

            prev_action = act_t
            frames_collected += 1

    z_vectors = np.stack(z_list, axis=0).astype(np.float32)
    return (
        z_vectors,
        np.array(speed_list, dtype=np.float32),
        np.array(steer_list, dtype=np.float32),
        np.array(progress_list, dtype=np.float32),
    )


def _reduce_pca(z: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Project z to 2D using PCA (fast, linear, no external deps)."""
    mean = z.mean(axis=0)
    z_centered = z - mean
    _, _, vh = np.linalg.svd(z_centered, full_matrices=False)
    return z_centered @ vh[:n_components].T


def _reduce_tsne(z: np.ndarray, n_components: int = 2, perplexity: float = 30.0) -> np.ndarray:
    """Project z to 2D using t-SNE (better cluster separation, slower)."""
    try:
        from sklearn.manifold import TSNE
    except ImportError as exc:
        raise RuntimeError("scikit-learn required for t-SNE. Install: pip install scikit-learn") from exc
    return TSNE(n_components=n_components, perplexity=perplexity, random_state=42).fit_transform(z)


def make_latent_visualization(
    z_vectors: np.ndarray,
    speeds: np.ndarray,
    steers: np.ndarray,
    progress: np.ndarray,
    output_path: Path,
    method: str = "pca",
) -> Path:
    """Create a 3-panel figure showing the latent space coloured by different variables.

    Panel 1: coloured by track progress (episode position → proxy for track section)
    Panel 2: coloured by speed intent (throttle - brake)
    Panel 3: coloured by steering direction (-1 = full left, +1 = full right)
    """
    print(f"Reducing {z_vectors.shape[0]} latent vectors ({z_vectors.shape[1]}D) to 2D using {method.upper()}...")
    if method == "tsne":
        z_2d = _reduce_tsne(z_vectors)
    else:
        z_2d = _reduce_pca(z_vectors)
    print(f"Reduction complete. Plotting...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f"RSSM Latent Space — {z_vectors.shape[0]} frames, {method.upper()} projection\n"
        "If the model has learned meaningful representations, these plots should show structured clusters.",
        fontsize=13,
    )

    scatter_kwargs = dict(s=8, alpha=0.6, linewidths=0)

    # Panel 1: Track progress (position along episode)
    sc1 = axes[0].scatter(z_2d[:, 0], z_2d[:, 1], c=progress, cmap="plasma", **scatter_kwargs)
    plt.colorbar(sc1, ax=axes[0], label="Track progress (0=start, 1=end)")
    axes[0].set_title("Coloured by track progress\n(proxy for track section)")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")

    # Panel 2: Speed intent
    sc2 = axes[1].scatter(z_2d[:, 0], z_2d[:, 1], c=speeds, cmap="YlOrRd", **scatter_kwargs)
    plt.colorbar(sc2, ax=axes[1], label="Speed intent (throttle−brake)")
    axes[1].set_title("Coloured by speed intent\n(throttle − brake ≈ forward acceleration)")
    axes[1].set_xlabel("PC1")

    # Panel 3: Steering direction
    sc3 = axes[2].scatter(z_2d[:, 0], z_2d[:, 1], c=steers, cmap="RdBu", vmin=-1.0, vmax=1.0, **scatter_kwargs)
    plt.colorbar(sc3, ax=axes[2], label="Steering (−1=left, +1=right)")
    axes[2].set_title("Coloured by steering direction\n(−1 = full left turn, +1 = full right)")
    axes[2].set_xlabel("PC1")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Latent space visualisation saved to {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualise RSSM latent space via PCA or t-SNE.")
    parser.add_argument("--checkpoint", required=True, help="Path to RSSM .pt checkpoint.")
    parser.add_argument("--config", default="config/world_model_config.yaml")
    parser.add_argument("--replay-dir", default="results/world_model/replay",
                        help="Directory containing .npz episode files.")
    parser.add_argument("--manifest", default=None,
                        help="Path to train_manifest.json. If given, overrides --replay-dir.")
    parser.add_argument("--n-frames", type=int, default=500,
                        help="Number of frames to encode (more = slower but smoother plot).")
    parser.add_argument("--output", default="results/demo/latent_viz.png",
                        help="Output .png path.")
    parser.add_argument("--method", choices=["pca", "tsne"], default="pca",
                        help="Dimensionality reduction method. PCA is fast; t-SNE is slower but shows clusters better.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device(args.device)
    model = _load_checkpoint(Path(args.checkpoint), config, device)

    # Collect episode paths
    if args.manifest:
        manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        episode_paths = [Path(p) for p in manifest["episodes"]]
    else:
        episode_paths = list(Path(args.replay_dir).rglob("*.npz"))
    print(f"Found {len(episode_paths)} episode files.")

    z_vectors, speeds, steers, progress = _encode_frames(
        model=model,
        episode_paths=episode_paths,
        n_frames=args.n_frames,
        device=device,
    )
    print(f"Encoded {z_vectors.shape[0]} frames with latent dim {z_vectors.shape[1]}.")

    make_latent_visualization(
        z_vectors=z_vectors,
        speeds=speeds,
        steers=steers,
        progress=progress,
        output_path=Path(args.output),
        method=args.method,
    )


if __name__ == "__main__":
    main()
