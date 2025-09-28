"""
Noise scheduler for discrete diffusion (D3PM-style) with uniform mixing kernel.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


class DiscreteNoiseScheduler:
    """
    Discrete diffusion noise scheduler using uniform mixing kernel.
    At each step, keep token with prob (1 - beta_t), else replace with random token.
    """

    def __init__(
        self,
        num_timesteps: int = 32,
        vocab_size: int = 11,  # {0..9, PAD}
        schedule_type: str = "cosine",
        beta_min: float = 1e-4,
        beta_max: float = 2e-2
    ):
        self.num_timesteps = num_timesteps
        self.vocab_size = vocab_size
        self.schedule_type = schedule_type
        self.beta_min = beta_min
        self.beta_max = beta_max

        # Create noise schedule
        self.betas = self._create_schedule()

        # Pre-compute useful quantities
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def to(self, device: torch.device):
        """Move scheduler tensors to device."""
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bars = self.alpha_bars.to(device)
        return self

    def _create_schedule(self) -> torch.Tensor:
        """Create beta schedule (noise levels over time)."""
        if self.schedule_type == "cosine":
            # Cosine schedule from beta_min to beta_max
            timesteps = torch.arange(0, self.num_timesteps + 1, dtype=torch.float32)
            s = 0.008  # offset to prevent beta from becoming too small at t=0
            alpha_bars = torch.cos(((timesteps / self.num_timesteps) + s) / (1 + s) * np.pi / 2) ** 2
            alpha_bars = alpha_bars / alpha_bars[0]  # normalize so alpha_bar[0] = 1

            # Convert to betas, clip to [beta_min, beta_max]
            betas = 1 - alpha_bars[1:] / alpha_bars[:-1]
            betas = torch.clamp(betas, self.beta_min, self.beta_max)

        elif self.schedule_type == "linear":
            betas = torch.linspace(self.beta_min, self.beta_max, self.num_timesteps)
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")

        return betas

    def add_noise(self, x0: torch.Tensor, t: torch.Tensor, token_distribution: torch.Tensor = None) -> torch.Tensor:
        """
        Add noise to clean tokens x0 at timestep t using global mixing kernel.

        Args:
            x0: Clean tokens [batch_size, height, width]
            t: Timestep tensor [batch_size]
            token_distribution: Global token probabilities [vocab_size] or None for uniform

        Returns:
            xt: Noisy tokens [batch_size, height, width]
        """
        batch_size, height, width = x0.shape
        device = x0.device

        # Get alpha_bar for each sample in the batch
        alpha_bars_t = self.alpha_bars[t].to(device)  # [batch_size]

        # Create random mask: keep original token with prob alpha_bar_t
        keep_mask = torch.rand(batch_size, height, width, device=device) < alpha_bars_t.unsqueeze(-1).unsqueeze(-1)

        # Create random tokens for replacement using global distribution
        if token_distribution is not None:
            # Sample from global distribution
            dist = token_distribution.to(device)
            total_pixels = batch_size * height * width
            pixels = torch.multinomial(dist, num_samples=total_pixels, replacement=True)
            random_tokens = pixels.view(batch_size, height, width)
        else:
            # Fallback to uniform sampling (should only happen in tests)
            print("Warning: Using uniform token distribution fallback - this may produce unrealistic noise")
            random_tokens = torch.randint(0, self.vocab_size, (batch_size, height, width), device=device)

        # Apply noise: keep original where mask is True, replace with random elsewhere
        xt = torch.where(keep_mask, x0, random_tokens)

        return xt

    def get_schedule_info(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get alpha and alpha_bar values for timestep t."""
        device = t.device
        alphas_t = self.alphas[t].to(device)
        alpha_bars_t = self.alpha_bars[t].to(device)
        return alphas_t, alpha_bars_t


def create_timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Create sinusoidal timestep embeddings.

    Args:
        timesteps: Tensor of timesteps [batch_size]
        dim: Embedding dimension

    Returns:
        Timestep embeddings [batch_size, dim]
    """
    half_dim = dim // 2
    emb = np.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) * -emb)
    emb = timesteps.unsqueeze(-1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros(emb.shape[0], 1, device=emb.device)], dim=-1)

    return emb