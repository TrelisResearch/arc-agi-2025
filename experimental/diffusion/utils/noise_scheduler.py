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

    Uses cosine schedule with alpha_bar_final and cosine_s parameters.
    """

    def __init__(
        self,
        num_timesteps: int = 32,
        vocab_size: int = 11,  # {0..9, PAD}
        schedule_type: str = "cosine",
        alpha_bar_final: float = 0.02,  # α_bar_T target (88.2% noise for 10-class uniform)
        cosine_s: float = 0.008,        # cosine offset
    ):
        self.num_timesteps = num_timesteps
        self.vocab_size = vocab_size
        self.schedule_type = schedule_type
        self.alpha_bar_final = alpha_bar_final
        self.cosine_s = cosine_s

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
        """Create alpha_bar schedule using cosine formula."""
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

    def _compute_betas_from_alpha_bars(self) -> torch.Tensor:
        """Compute betas from alpha_bars: β_t = 1 - α_t = 1 - α_bar_t / α_bar_{t-1}"""
        # Prepend α_bar_0 = 1.0 for computation (device-matched for neatness)
        alpha_bars_with_0 = torch.cat([self.alpha_bars.new_tensor([1.0]), self.alpha_bars])

        # Compute α_t = α_bar_t / α_bar_{t-1}
        alphas = self.alpha_bars / alpha_bars_with_0[:-1]

        # Compute β_t = 1 - α_t
        betas = 1.0 - alphas

        return betas

    def add_noise(self, x0: torch.Tensor, t: torch.Tensor, masks: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Add noise to clean tokens x0 at timestep t using uniform mixing kernel.
        Only noise valid regions (where mask=1), clamp invalid regions to 0 (black).

        Args:
            x0: Clean tokens [batch_size, height, width]
            t: Timestep tensor [batch_size]
            masks: Valid region masks [batch_size, height, width], 1 for valid, 0 for invalid

        Returns:
            xt: Noisy tokens [batch_size, height, width]
        """
        batch_size, height, width = x0.shape
        device = x0.device

        # If masks provided, ensure x0 doesn't have PAD tokens outside valid region
        # (convert any PAD/10 to 0/black outside the mask)
        if masks is not None:
            x0 = torch.where(masks.bool(), x0, 0)

        # Get alpha_bar for each sample in the batch
        alpha_bars_t = self.alpha_bars[t].to(device)  # [batch_size]

        # Create random mask: keep original token with prob alpha_bar_t
        keep_mask = torch.rand(batch_size, height, width, device=device) < alpha_bars_t.unsqueeze(-1).unsqueeze(-1)

        # Create random tokens for replacement using uniform distribution over colors {0..9}
        random_tokens = torch.randint(0, 10, (batch_size, height, width), device=device)

        # Apply noise: keep original where mask is True, replace with random elsewhere
        xt = torch.where(keep_mask, x0, random_tokens)

        # If masks provided, clamp invalid regions to 0 (black)
        if masks is not None:
            xt = torch.where(masks.bool(), xt, 0)

        return xt

    def get_schedule_info(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get alpha and alpha_bar values for timestep t."""
        device = t.device
        alphas_t = self.alphas[t].to(device)
        alpha_bars_t = self.alpha_bars[t].to(device)
        return alphas_t, alpha_bars_t

    def compute_alpha_bar_from_tau(self, tau: torch.Tensor) -> torch.Tensor:
        """
        Compute alpha_bar directly from normalized time tau ∈ [0, 1].

        Uses the same cosine formula as _create_alpha_bars, but computes continuously
        from tau instead of discrete timestep indices.

        Args:
            tau: Normalized time [batch_size], where 0 = clean, 1 = maximum noise

        Returns:
            alpha_bars: Alpha_bar values [batch_size]
        """
        device = tau.device

        # Compute raw cosine value: r(tau) = cos^2((tau + s)/(1 + s) * π/2)
        r_tau = torch.cos(((tau + self.cosine_s) / (1 + self.cosine_s)) * np.pi / 2) ** 2

        # Compute r_0 for normalization (at tau=0)
        r_0 = torch.cos((self.cosine_s / (1 + self.cosine_s)) * np.pi / 2) ** 2

        # Scale: α_bar(tau) = α_bar_final + (1 - α_bar_final) * r(tau) / r_0
        alpha_bars = self.alpha_bar_final + (1 - self.alpha_bar_final) * r_tau / r_0

        return alpha_bars.to(device)


def sc_gain_from_abar(
    t_idx: torch.Tensor,
    scheduler,
    a: float = 0.3,
    b: float = 1.0,
    gamma: float = 1.0
) -> torch.Tensor:
    """
    Self-conditioning gain based on alpha_bar at absolute training indices.

    Args:
        t_idx: Absolute timestep indices [batch_size], in range [0, T-1]
        scheduler: Noise scheduler with alpha_bars
        a: Minimum gain (at high noise, low alpha_bar). Default 0.3
        b: Maximum gain (at low noise, high alpha_bar). Default 1.0
        gamma: Exponent for progress curve. Default 1.0 (linear)

    Returns:
        sc_gain: Self-conditioning gain [batch_size], in range [a, b]

    Examples:
        - Low alpha_bar (high noise, t≈T-1) → gain ≈ a (0.3)
        - High alpha_bar (low noise, t≈0) → gain ≈ b (1.0)
    """
    abar = scheduler.alpha_bars[t_idx].to(t_idx.device)  # [batch_size], in (0, 1]
    progress = abar.clamp(min=1e-6)  # Numeric safety
    gain = a + (b - a) * (progress ** gamma)
    return gain.clamp(min=a, max=b)  # [batch_size]


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