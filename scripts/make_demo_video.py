"""Generate a side-by-side hallucination comparison video for presentation.

This script produces the primary presentation artifact:
  LEFT panel  — real episode frames (ground truth)
  RIGHT panel — world model's hallucination from the same starting state
  BOTTOM bar  — text overlay showing predicted vs. actual reward, frame index

How hallucination works (explained for the audience):
  1. We feed the model the first `context_length` real frames.
     During this phase the model uses BOTH the real image AND its own prediction
     (via the posterior distribution) to update its internal state.
  2. After the context, we cut off real frames. The model now imagines future
     frames using only its recurrent hidden state (h_t) and the prior distribution
     — no real observations. This is "imagination".
  3. We decode the imagined latent states back to pixel space and compare
     side-by-side with what actually happened.

If the model is good, the two panels look similar — same road geometry, same
car position, same corner shape. Differences reveal what the model got wrong.

Usage:
    python scripts/make_demo_video.py \\
        --checkpoint models/world_model/rssm_sequence.pt \\
        --config config/world_model_config.yaml \\
        --episode results/world_model/replay/manual_wide_line_20260404_145350/episode_000002.npz \\
        --start-index 80 \\
        --context-length 50 \\
        --horizon 150 \\
        --output results/demo/comparison.mp4 \\
        --fps 10
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from world_model.models import RSSMSequence
from world_model.replay import ReplayWriter
from world_model.training import hallucinate_future


def _load_checkpoint(checkpoint_path: Path, config: dict, device: torch.device) -> RSSMSequence:
    """Load RSSM from checkpoint.

    The checkpoint contains model_state_dict (weights) and the config that was
    used to train it. We use the config passed here (not the checkpoint's config)
    so the architecture matches whatever the user specifies.
    """
    model = RSSMSequence(**config["rssm"]).to(device)
    payload = torch.load(str(checkpoint_path), map_location=device)
    state_dict = payload.get("model_state_dict", payload)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _render_text_bar(width: int, lines: list[str], font_scale: float = 0.55, bar_height: int = 60) -> np.ndarray:
    """Render a dark text bar below the video frames."""
    bar = np.zeros((bar_height, width, 3), dtype=np.uint8)
    y = 20
    for line in lines:
        cv2.putText(bar, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (220, 220, 220), 1, cv2.LINE_AA)
        y += 22
    return bar


def _upscale(frame: np.ndarray, scale: int) -> np.ndarray:
    """Upscale a (H, W, 3) uint8 frame by integer factor for visibility."""
    h, w = frame.shape[:2]
    return cv2.resize(frame, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)


def make_comparison_video(
    model: RSSMSequence,
    episode_path: Path,
    start_index: int,
    context_length: int,
    horizon: int,
    output_path: Path,
    device: torch.device,
    fps: float = 10.0,
    upscale_factor: int = 4,
) -> Path:
    """Generate the side-by-side comparison video.

    Args:
        context_length: Number of real frames fed to the model before imagination starts.
        horizon: Number of imagined frames to generate and compare.
        upscale_factor: Integer pixel zoom (4 = 96px → 384px, much more visible on screen).
    """
    episode = ReplayWriter.load_episode(episode_path)
    total_length = context_length + horizon
    end_index = start_index + total_length

    if end_index > episode.observations_uint8.shape[0]:
        raise ValueError(
            f"Not enough frames: episode has {episode.observations_uint8.shape[0]} frames, "
            f"need {end_index} (start={start_index} + context={context_length} + horizon={horizon})."
        )

    images_np = episode.observations_uint8[start_index:end_index]           # (T, H, W, C) uint8
    actions_np = episode.actions[start_index:end_index]                      # (T, 3)
    rewards_np = episode.rewards[start_index:end_index]                      # (T,)

    images_t = torch.from_numpy(images_np).float().permute(0, 3, 1, 2) / 255.0  # (T, C, H, W)
    actions_t = torch.from_numpy(actions_np).float()
    images_t = images_t.unsqueeze(0).to(device)    # (1, T, C, H, W)
    actions_t = actions_t.unsqueeze(0).to(device)  # (1, T, 3)

    with torch.no_grad():
        imagined = hallucinate_future(
            model=model,
            images=images_t,
            actions=actions_t,
            context_length=context_length,
            horizon=horizon,
        )  # (1, horizon, C, H, W)

    real_frames = images_np[context_length:]                                  # (horizon, H, W, C) uint8
    imagined_np = imagined[0].detach().cpu().permute(0, 2, 3, 1).numpy()     # (horizon, H, W, C) float
    imagined_uint8 = np.clip(imagined_np * 255.0, 0.0, 255.0).astype(np.uint8)

    # Also run reconstruction on the context to compare reconstruction vs. hallucination quality
    # We use reward prediction from the model's imagination (not available here directly, so we skip)
    real_rewards = rewards_np[context_length:]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame_h, frame_w = real_frames.shape[1:3]
    scaled_h = frame_h * upscale_factor
    scaled_w = frame_w * upscale_factor
    text_bar_h = 65
    video_h = scaled_h + text_bar_h
    video_w = scaled_w * 2  # two panels side by side

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (video_w, video_h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {output_path}")

    try:
        for i in range(horizon):
            real_scaled = _upscale(real_frames[i], upscale_factor)
            imag_scaled = _upscale(imagined_uint8[i], upscale_factor)

            # Label each panel
            cv2.putText(real_scaled, "REAL", (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(imag_scaled, "IMAGINED", (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 100), 2, cv2.LINE_AA)

            side_by_side = np.concatenate([real_scaled, imag_scaled], axis=1)  # (H, 2W, 3)

            # Text bar with frame info
            real_r = float(real_rewards[i])
            bar = _render_text_bar(
                width=video_w,
                lines=[
                    f"Frame {i + 1}/{horizon} | Imagined from context of {context_length} real frames",
                    f"Real reward: {real_r:+.3f}",
                ],
                bar_height=text_bar_h,
            )

            frame = np.concatenate([side_by_side, bar], axis=0)
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()

    print(f"Demo video saved to {output_path} ({horizon} frames at {fps:.0f}fps)")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a side-by-side hallucination comparison video for presentation."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to RSSM .pt checkpoint.")
    parser.add_argument("--config", default="config/world_model_config.yaml")
    parser.add_argument("--episode", required=True, help="Path to .npz episode file.")
    parser.add_argument("--start-index", type=int, default=80,
                        help="Frame index in the episode to start from.")
    parser.add_argument("--context-length", type=int, default=50,
                        help="Number of real frames fed to the model before imagination.")
    parser.add_argument("--horizon", type=int, default=150,
                        help="Number of imagined frames to generate.")
    parser.add_argument("--output", default="results/demo/comparison.mp4",
                        help="Output video path.")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--upscale", type=int, default=4,
                        help="Integer pixel zoom factor (4 = 96px → 384px).")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device(args.device)
    model = _load_checkpoint(Path(args.checkpoint), config, device)

    make_comparison_video(
        model=model,
        episode_path=Path(args.episode),
        start_index=args.start_index,
        context_length=args.context_length,
        horizon=args.horizon,
        output_path=Path(args.output),
        device=device,
        fps=args.fps,
        upscale_factor=args.upscale,
    )


if __name__ == "__main__":
    main()
