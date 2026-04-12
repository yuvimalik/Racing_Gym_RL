from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Independent, Normal, kl_divergence


class PerceptualLoss(nn.Module):
    """VGG16-based perceptual loss for image reconstruction.

    Why perceptual loss instead of (or alongside) MSE?
    MSE minimises per-pixel differences. Because each pixel is penalised
    independently, the model learns to predict the *average* of uncertain
    futures, producing blurry reconstructions. Road edges, turn geometry
    and track markings are high-frequency patterns — MSE blurs them.

    Perceptual loss instead compares images in the *feature space* of a
    pretrained VGG16 network. VGG16 was trained to recognise objects and
    textures, so its intermediate activations are sensitive to edges and
    structure. Two images that look structurally similar (sharp road edge
    at the same location) will have similar VGG features even if they
    differ in a few pixel values.

    We extract features from VGG16 relu3_2 (up to layer index 16), which
    captures mid-level features (edges, corners, textures) without going so
    deep that semantic content dominates. The features are frozen — we are
    *not* training VGG16; we use it as a fixed perceptual similarity metric.

    Usage in training:
        total_recon = 0.5 * mse_loss + 0.5 * perceptual_loss
    The mix ratio can be tuned by the W&B sweep (loss_type parameter).
    """

    def __init__(self, feature_layer: int = 16, device: str | torch.device = "cpu"):
        super().__init__()
        try:
            import torchvision.models as tv_models
        except ImportError as exc:
            raise RuntimeError(
                "torchvision is required for PerceptualLoss. Install with: pip install torchvision"
            ) from exc

        vgg = tv_models.vgg16(weights=tv_models.VGG16_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(vgg.features)[:feature_layer]).eval()
        # Freeze VGG weights — we use it as a fixed feature extractor, not a learnable network.
        for param in self.features.parameters():
            param.requires_grad = False
        self.features = self.features.to(device)

        # ImageNet normalisation constants (VGG16 was trained on ImageNet with these stats).
        # Our images are already in [0, 1], so we apply normalisation inside the loss.
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std  # type: ignore[operator]

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Normalise both images to ImageNet stats before passing through VGG.
        pred_feat = self.features(self._normalize(prediction))
        tgt_feat = self.features(self._normalize(target))
        return F.mse_loss(pred_feat, tgt_feat)


def reconstruction_loss(reconstruction: torch.Tensor, target_images: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(reconstruction, target_images)


def reward_loss(predicted_reward: torch.Tensor, target_reward: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(predicted_reward, target_reward)


def masked_mse_loss(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    squared_error = (prediction - target) ** 2
    weighted_error = squared_error * mask
    normalizer = torch.clamp(mask.sum(), min=1.0)
    return weighted_error.sum() / normalizer


def masked_bce_with_logits_loss(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    raw_loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    weighted_loss = raw_loss * mask
    normalizer = torch.clamp(mask.sum(), min=1.0)
    return weighted_loss.sum() / normalizer


def free_bits_kl(
    posterior_mean: torch.Tensor,
    posterior_std: torch.Tensor,
    prior_mean: torch.Tensor,
    prior_std: torch.Tensor,
    free_nats: float = 1.0,
) -> torch.Tensor:
    posterior = Independent(Normal(posterior_mean, posterior_std), 1)
    prior = Independent(Normal(prior_mean, prior_std), 1)
    kl_values = kl_divergence(posterior, prior)
    return torch.clamp(kl_values, min=float(free_nats)).mean()
