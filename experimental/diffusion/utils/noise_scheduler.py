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

    For cosine schedule: Uses alpha_bar_final and cosine_s parameters.
    For linear schedule: Uses beta_min and beta_max parameters.
    """

    def __init__(
        self,
        num_timesteps: int = 32,
        vocab_size: int = 11,  # {0..9, PAD}
        schedule_type: str = "cosine",
        alpha_bar_final: float = 0.02,  # α_bar_T target (88.2% noise for 10-class uniform)
        cosine_s: float = 0.008,        # cosine offset
        # Legacy parameters for linear schedule (unused for cosine)
        beta_min: float = 1e-4,
        beta_max: float = 2e-2
    ):
        self.num_timesteps = num_timesteps
        self.vocab_size = vocab_size
        self.schedule_type = schedule_type
        self.alpha_bar_final = alpha_bar_final
        self.cosine_s = cosine_s

        # Only used for linear schedule
        if schedule_type == "linear":
            self.beta_min = beta_min
            self.beta_max = beta_max

        # Create noise schedule
        self.alpha_bars = self._create_alpha_bars()
        self.betas = self._compute_betas_from_alpha_bars()

        # Pre-compute useful quantities
        self.alphas = 1.0 - self.betas

    def to(self, device: torch.device):
        """Move scheduler tensors to device."""
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bars = self.alpha_bars.to(device)
        return self

    def _create_alpha_bars(self) -> torch.Tensor:
        """Create alpha_bar schedule using specified formula."""
        if self.schedule_type == "cosine":
            # Use exact formula: r_t = cos^2((t/T + s)/(1 + s) * π/2)
            # α_bar_t = α_bar_T + (1 - α_bar_T) * r_t / r_0
            # With α_bar_T = 0.02, this gives 88.2% noise level (near max 90% for 10 classes)

            timesteps = torch.arange(0, self.num_timesteps + 1, dtype=torch.float32)  # [0, 1, ..., T]

            # Compute raw cosine values r_t
            r_values = torch.cos(((timesteps / self.num_timesteps) + self.cosine_s) / (1 + self.cosine_s) * np.pi / 2) ** 2

            # Normalize and scale: α_bar_t = α_bar_T + (1 - α_bar_T) * r_t / r_0
            r_0 = r_values[0]  # r_0 for normalization
            alpha_bars = self.alpha_bar_final + (1 - self.alpha_bar_final) * r_values / r_0

            # Return α_bars for timesteps 1, 2, ..., T (exclude t=0)
            return alpha_bars[1:]

        elif self.schedule_type == "linear":
            # Traditional linear-beta schedule: betas linearly spaced, then derive α_bars
            # β_t linearly spaced from beta_min to beta_max
            betas = torch.linspace(self.beta_min, self.beta_max, self.num_timesteps)
            alphas = 1.0 - betas
            # α_bar_t = ∏(α_i) for i=1 to t
            alpha_bars = torch.cumprod(alphas, dim=0)
            return alpha_bars
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")

    def _compute_betas_from_alpha_bars(self) -> torch.Tensor:
        """Compute betas from alpha_bars: β_t = 1 - α_t = 1 - α_bar_t / α_bar_{t-1}"""
        # Prepend α_bar_0 = 1.0 for computation
        alpha_bars_with_0 = torch.cat([torch.tensor([1.0]), self.alpha_bars])

        # Compute α_t = α_bar_t / α_bar_{t-1}
        alphas = self.alpha_bars / alpha_bars_with_0[:-1]

        # Compute β_t = 1 - α_t
        betas = 1.0 - alphas

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

        # Create random tokens for replacement using uniform distribution over {0..9}
        random_tokens = torch.randint(0, 10, (batch_size, height, width), device=device)

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