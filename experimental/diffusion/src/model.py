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

        # Size prediction head
        self.size_head = SizePredictionHead(d_model, max_size)

    def forward(
        self,
        xt: torch.Tensor,  # [batch_size, max_size, max_size] - noisy output tokens
        input_grid: torch.Tensor,  # [batch_size, max_size, max_size] - input grid
        task_ids: torch.Tensor,  # [batch_size] - task IDs
        timesteps: torch.Tensor,  # [batch_size] - timesteps
        masks: Optional[torch.Tensor] = None,  # [batch_size, max_size, max_size]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the denoiser.

        Returns:
            logits: [batch_size, max_size, max_size, vocab_size] - predicted logits
            size_logits: [batch_size, max_size * 2] - predicted height and width logits
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

        # Predict sizes using pooled representation
        pooled_features = output.mean(dim=1)  # Global average pooling
        size_features = pooled_features + conditioning  # Add conditioning
        size_logits = self.size_head(size_features)  # [batch_size, max_size * 2]

        return logits, size_logits


class SizePredictionHead(nn.Module):
    """Head for predicting grid dimensions (H, W)."""

    def __init__(self, d_model: int, max_size: int):
        super().__init__()
        self.max_size = max_size
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, max_size * 2)  # max_size logits for H, max_size logits for W
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict size logits.

        Args:
            x: Features [batch_size, d_model]

        Returns:
            Size logits [batch_size, max_size * 2] - first max_size for H, next max_size for W
        """
        return self.head(x)


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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass - predict x0 given xt."""
        return self.denoiser(xt, input_grid, task_ids, timesteps, masks)

    def compute_loss(
        self,
        x0: torch.Tensor,  # [batch_size, max_size, max_size] - clean output
        input_grid: torch.Tensor,  # [batch_size, max_size, max_size] - input
        task_ids: torch.Tensor,  # [batch_size] - task IDs
        heights: torch.Tensor,  # [batch_size] - true heights
        widths: torch.Tensor,  # [batch_size] - true widths
        xt: torch.Tensor,  # [batch_size, max_size, max_size] - noisy tokens
        timesteps: torch.Tensor,  # [batch_size] - timesteps
        size_weight: float = 0.2,
    ) -> Dict[str, torch.Tensor]:
        """Compute training losses."""
        batch_size = x0.shape[0]
        device = x0.device

        # Create masks
        masks = batch_create_masks(heights, widths, self.max_size).to(device)

        # Forward pass
        logits, size_logits = self.forward(xt, input_grid, task_ids, timesteps, masks)

        # Grid loss: cross-entropy masked to valid regions
        grid_loss = F.cross_entropy(
            logits.view(-1, self.vocab_size),  # [batch_size * max_size^2, vocab_size]
            x0.view(-1),  # [batch_size * max_size^2]
            reduction='none'
        )
        grid_loss = grid_loss.view(batch_size, self.max_size, self.max_size)
        grid_loss = (grid_loss * masks).sum() / masks.sum()  # Masked average

        # Size loss: cross-entropy for height and width
        size_logits_h = size_logits[:, :self.max_size]  # [batch_size, max_size]
        size_logits_w = size_logits[:, self.max_size:]  # [batch_size, max_size]

        # Convert sizes to 0-indexed (1-30 -> 0-29)
        size_targets_h = heights - 1
        size_targets_w = widths - 1

        size_loss_h = F.cross_entropy(size_logits_h, size_targets_h)
        size_loss_w = F.cross_entropy(size_logits_w, size_targets_w)
        size_loss = (size_loss_h + size_loss_w) / 2

        # Total loss
        total_loss = grid_loss + size_weight * size_loss

        return {
            'total_loss': total_loss,
            'grid_loss': grid_loss,
            'size_loss': size_loss,
            'size_loss_h': size_loss_h,
            'size_loss_w': size_loss_w
        }

    def predict_size(
        self,
        input_grid: torch.Tensor,
        task_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict output grid size from input grid and task.

        Returns:
            heights: [batch_size] - predicted heights (1-30)
            widths: [batch_size] - predicted widths (1-30)
        """
        with torch.no_grad():
            batch_size = input_grid.shape[0]
            device = input_grid.device

            # Use dummy values for xt and timesteps (won't affect size prediction)
            xt_dummy = torch.zeros_like(input_grid)
            timesteps_dummy = torch.zeros(batch_size, device=device, dtype=torch.long)

            _, size_logits = self.forward(xt_dummy, input_grid, task_ids, timesteps_dummy)

            size_logits_h = size_logits[:, :self.max_size]  # [batch_size, max_size]
            size_logits_w = size_logits[:, self.max_size:]  # [batch_size, max_size]

            heights = torch.argmax(size_logits_h, dim=-1) + 1  # Convert back to 1-30
            widths = torch.argmax(size_logits_w, dim=-1) + 1

            return heights, widths