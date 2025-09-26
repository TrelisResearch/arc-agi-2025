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

    def add_noise(self, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Add noise to clean tokens x0 at timestep t using uniform mixing kernel.

        Args:
            x0: Clean tokens [batch_size, height, width]
            t: Timestep tensor [batch_size]

        Returns:
            xt: Noisy tokens [batch_size, height, width]
        """
        batch_size, height, width = x0.shape
        device = x0.device

        # Get alpha_bar for each sample in the batch
        alpha_bars_t = self.alpha_bars[t].to(device)  # [batch_size]

        # Create random mask: keep original token with prob alpha_bar_t
        keep_mask = torch.rand(batch_size, height, width, device=device) < alpha_bars_t.unsqueeze(-1).unsqueeze(-1)

        # Create random tokens for replacement
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