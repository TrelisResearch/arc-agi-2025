"""
Discrete diffusion model for ARC tasks with size prediction and transformer backbone.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional, Dict, Any

from ..utils.noise_scheduler import create_timestep_embedding
from ..utils.grid_utils import clamp_outside_mask, batch_create_masks


class PositionalEncoding(nn.Module):
    """2D positional encoding for grid positions."""

    def __init__(self, d_model: int, max_size: int = 30):
        super().__init__()
        self.d_model = d_model

        # Create 2D positional encoding
        pe = torch.zeros(max_size, max_size, d_model)
        pos_h = torch.arange(0, max_size).unsqueeze(1).float()  # [max_size, 1]
        pos_w = torch.arange(0, max_size).unsqueeze(0).float()  # [1, max_size]

        div_term = torch.exp(torch.arange(0, d_model // 2, 2).float() *
                           -(math.log(10000.0) / (d_model // 2)))

        # Apply sinusoidal encoding to height dimension
        sin_h = torch.sin(pos_h.unsqueeze(-1) * div_term)  # [max_size, 1, d_model//4]
        cos_h = torch.cos(pos_h.unsqueeze(-1) * div_term)  # [max_size, 1, d_model//4]

        # Apply sinusoidal encoding to width dimension
        sin_w = torch.sin(pos_w.unsqueeze(-1) * div_term)  # [1, max_size, d_model//4]
        cos_w = torch.cos(pos_w.unsqueeze(-1) * div_term)  # [1, max_size, d_model//4]

        # Broadcast and assign to appropriate positions
        pe[:, :, 0::4] = sin_h.expand(max_size, max_size, -1)
        pe[:, :, 1::4] = cos_h.expand(max_size, max_size, -1)
        pe[:, :, 2::4] = sin_w.expand(max_size, max_size, -1)
        pe[:, :, 3::4] = cos_w.expand(max_size, max_size, -1)

        self.register_buffer('pe', pe)

    def forward(self, batch_size: int) -> torch.Tensor:
        """Return positional encoding [batch_size, max_size * max_size, d_model]."""
        pe = self.pe.view(-1, self.d_model)  # [max_size^2, d_model]
        return pe.unsqueeze(0).expand(batch_size, -1, -1)


class TransformerDenoiser(nn.Module):
    """
    Transformer backbone for the diffusion denoiser.
    Processes 900 tokens (30x30 grid) with task and time conditioning.
    """

    def __init__(
        self,
        vocab_size: int = 11,
        d_model: int = 384,
        nhead: int = 6,
        num_layers: int = 8,
        max_size: int = 30,
        max_tasks: int = 1000,  # Maximum number of task IDs
    ):
        super().__init__()
        self.d_model = d_model
        self.max_size = max_size
        self.vocab_size = vocab_size

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_size)

        # Task embedding (for task conditioning)
        self.task_embedding = nn.Embedding(max_tasks, d_model)

        # Time embedding
        self.time_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )

        # Input grid encoder (separate from the main grid being denoised)
        self.input_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=2
        )

        # Main transformer
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=num_layers
        )

        # Output head (predicts logits for each cell)
        self.output_head = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        xt: torch.Tensor,  # [batch_size, max_size, max_size] - noisy output tokens
        input_grid: torch.Tensor,  # [batch_size, max_size, max_size] - input grid
        task_ids: torch.Tensor,  # [batch_size] - task IDs
        timesteps: torch.Tensor,  # [batch_size] - timesteps
        masks: Optional[torch.Tensor] = None,  # [batch_size, max_size, max_size]
    ) -> torch.Tensor:
        """
        Forward pass of the denoiser.

        Returns:
            logits: [batch_size, max_size, max_size, vocab_size] - predicted logits
        """
        batch_size = xt.shape[0]
        device = xt.device

        # Embed tokens
        xt_flat = xt.view(batch_size, -1)  # [batch_size, max_size^2]
        input_flat = input_grid.view(batch_size, -1)  # [batch_size, max_size^2]

        xt_emb = self.token_embedding(xt_flat)  # [batch_size, max_size^2, d_model]
        input_emb = self.token_embedding(input_flat)  # [batch_size, max_size^2, d_model]

        # Add positional encoding
        pos_emb = self.pos_encoding(batch_size)
        xt_emb = xt_emb + pos_emb
        input_emb = input_emb + pos_emb

        # Encode input grid
        input_encoded = self.input_encoder(input_emb)  # [batch_size, max_size^2, d_model]

        # Create task conditioning
        task_emb = self.task_embedding(task_ids)  # [batch_size, d_model]

        # Create time conditioning
        time_emb = create_timestep_embedding(timesteps, self.d_model).to(device)
        time_emb = self.time_projection(time_emb)  # [batch_size, d_model]

        # Add task and time conditioning to each token
        conditioning = task_emb + time_emb  # [batch_size, d_model]
        xt_emb = xt_emb + conditioning.unsqueeze(1)  # Broadcast to all positions

        # Cross-attention: add input encoding as conditioning
        xt_emb = xt_emb + input_encoded

        # Apply transformer
        output = self.transformer(xt_emb)  # [batch_size, max_size^2, d_model]

        # Predict logits for each cell
        logits = self.output_head(output)  # [batch_size, max_size^2, vocab_size]
        logits = logits.view(batch_size, self.max_size, self.max_size, self.vocab_size)

        return logits



class ARCDiffusionModel(nn.Module):
    """
    Complete diffusion model for ARC tasks.
    Combines the denoiser with loss computation and sampling logic.
    """

    def __init__(
        self,
        vocab_size: int = 11,
        d_model: int = 384,
        nhead: int = 6,
        num_layers: int = 8,
        max_size: int = 30,
        max_tasks: int = 1000,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_size = max_size

        self.denoiser = TransformerDenoiser(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            max_size=max_size,
            max_tasks=max_tasks
        )

    def forward(
        self,
        xt: torch.Tensor,
        input_grid: torch.Tensor,
        task_ids: torch.Tensor,
        timesteps: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass - predict x0 given xt."""
        return self.denoiser(xt, input_grid, task_ids, timesteps, masks)

    def compute_loss(
        self,
        x0: torch.Tensor,  # [batch_size, max_size, max_size] - clean output
        input_grid: torch.Tensor,  # [batch_size, max_size, max_size] - input
        task_ids: torch.Tensor,  # [batch_size] - task IDs
        xt: torch.Tensor,  # [batch_size, max_size, max_size] - noisy tokens
        timesteps: torch.Tensor,  # [batch_size] - timesteps
    ) -> Dict[str, torch.Tensor]:
        """Compute training losses."""
        batch_size = x0.shape[0]

        # Forward pass
        logits = self.forward(xt, input_grid, task_ids, timesteps)

        # Grid loss: cross-entropy on all positions
        grid_loss = F.cross_entropy(
            logits.view(-1, self.vocab_size),  # [batch_size * max_size^2, vocab_size]
            x0.view(-1),  # [batch_size * max_size^2]
            reduction='mean'
        )

        return {
            'total_loss': grid_loss,
            'grid_loss': grid_loss,
        }

