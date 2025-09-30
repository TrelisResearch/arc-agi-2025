"""
Discrete diffusion model for ARC tasks with size prediction and transformer backbone.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict

from ..utils.noise_scheduler import create_timestep_embedding


class CoordinatePositionalEncoding(nn.Module):
    """2D coordinate-based positional encoding for grid positions."""

    def __init__(self, max_size: int = 20, d_model: int = 256):
        super().__init__()
        assert d_model % 2 == 0
        self.max_size = max_size
        self.row_embed = nn.Embedding(max_size, d_model // 2)
        self.col_embed = nn.Embedding(max_size, d_model // 2)

    def forward(self, batch_size: int) -> torch.Tensor:
        """Return positional encoding [batch_size, max_size * max_size, d_model]."""
        rows = torch.arange(self.max_size, device=self.row_embed.weight.device)
        cols = torch.arange(self.max_size, device=self.col_embed.weight.device)

        # Create grid of indices
        row_idx = rows.view(-1, 1).expand(-1, self.max_size).flatten()
        col_idx = cols.view(1, -1).expand(self.max_size, -1).flatten()

        # Get embeddings and concatenate
        pos_embed = torch.cat([
            self.row_embed(row_idx),
            self.col_embed(col_idx)
        ], dim=-1)

        return pos_embed.unsqueeze(0).expand(batch_size, -1, -1)


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
        embedding_dropout: float = 0.1,
        input_grid_dropout: float = 0.0,  # Dropout probability for input grid conditioning
        sc_dropout_prob: float = 0.5,  # Self-conditioning dropout probability
    ):
        super().__init__()
        self.d_model = d_model
        self.max_size = max_size
        self.vocab_size = vocab_size
        self.input_grid_dropout = input_grid_dropout
        self.sc_dropout_prob = sc_dropout_prob

        # Token embedding with padding_idx for PAD token
        # Input/output grids use PAD (10) which gets auto-zeroed
        # Noised grids (xt) use 0 for invalid regions and need explicit masking
        self.PAD_ID = 10
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=self.PAD_ID)

        # Positional encoding
        self.pos_encoding = CoordinatePositionalEncoding(max_size, d_model)

        # Task embedding (for task conditioning)
        self.task_embedding = nn.Embedding(max_tasks, d_model)

        # Time embedding
        self.time_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )

        # Self-conditioning projection
        # Maps probability distributions (10 classes) to features
        self.sc_proj = nn.Linear(10, d_model)

        # Embedding dropout for regularization
        self.embedding_dropout = nn.Dropout(embedding_dropout)

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

        # Output head (predicts logits for colors 0-9 only, not PAD)
        # PAD regions are handled via masking, not prediction
        self.output_head = nn.Linear(d_model, 10)

    def forward(
        self,
        xt: torch.Tensor,  # [batch_size, max_size, max_size] - noisy output tokens
        input_grid: torch.Tensor,  # [batch_size, max_size, max_size] - input grid
        task_ids: torch.Tensor,  # [batch_size] - task IDs
        timesteps: torch.Tensor,  # [batch_size] - timesteps
        masks: Optional[torch.Tensor] = None,  # [batch_size, max_size, max_size]
        sc_p0: Optional[torch.Tensor] = None,  # [batch_size, max_size, max_size, 10] - self-conditioning probs
        sc_gain: float = 1.0,  # Self-conditioning gain factor
    ) -> torch.Tensor:
        """
        Forward pass of the denoiser.

        Returns:
            logits: [batch_size, max_size, max_size, 10] - predicted logits for colors 0-9
        """
        # Verify tokens are valid
        # xt contains only 0-9 (uses 0/black for invalid regions)
        # input_grid can contain 0-10 (uses 10/PAD for invalid regions)
        assert xt.max().item() <= 9, f"xt must contain tokens 0-9, got max {xt.max().item()}"
        assert xt.min().item() >= 0, f"xt must contain only non-negative values, got min {xt.min().item()}"
        assert input_grid.max().item() <= 10, f"input_grid must contain tokens 0-10, got max {input_grid.max().item()}"

        batch_size = xt.shape[0]
        device = xt.device

        # Flatten grids
        input_flat = input_grid.view(batch_size, -1)  # [batch_size, max_size^2]
        xt_flat = xt.view(batch_size, -1)  # [batch_size, max_size^2]

        # Embed tokens
        input_emb = self.token_embedding(input_flat)  # [batch_size, max_size^2, d_model]
        xt_emb = self.token_embedding(xt_flat)  # [batch_size, max_size^2, d_model]

        # Add positional encoding to both
        pos_emb = self.pos_encoding(batch_size)  # [batch_size, max_size^2, d_model]
        input_emb = input_emb + pos_emb
        xt_emb = xt_emb + pos_emb

        # Apply embedding dropout
        input_emb = self.embedding_dropout(input_emb)
        xt_emb = self.embedding_dropout(xt_emb)

        # Handle self-conditioning
        if sc_p0 is not None:
            # Reshape and project previous predictions
            sc_p0_flat = sc_p0.view(batch_size, -1, 10)  # [batch_size, max_size^2, 10]
            sc_features = self.sc_proj(sc_p0_flat)  # [batch_size, max_size^2, d_model]

            # Add to xt embeddings with gain factor
            xt_emb = xt_emb + sc_gain * sc_features
        elif self.training and torch.rand(1).item() > self.sc_dropout_prob:
            # During training without sc_p0, randomly apply zero self-conditioning
            # to train the model to work without self-conditioning
            zero_sc = torch.zeros(batch_size, self.max_size * self.max_size, 10, device=device)
            sc_features = self.sc_proj(zero_sc)
            xt_emb = xt_emb + sc_gain * sc_features

        # Apply masking to xt features if masks provided
        # Zero out embeddings outside valid regions
        if masks is not None:
            masks_flat = masks.view(batch_size, -1, 1).float()  # [batch_size, max_size^2, 1]
            xt_emb = xt_emb * masks_flat  # Zero out invalid regions

        # Apply input grid conditioning dropout (training only)
        if self.training and self.input_grid_dropout > 0:
            # Sample Bernoulli gates for each batch item
            # b ~ Bernoulli(1 - p) where p is dropout probability
            keep_prob = 1.0 - self.input_grid_dropout
            dropout_mask = torch.bernoulli(torch.full((batch_size, 1, 1), keep_prob, device=device))
            # Apply dropout with scaling to maintain expectation
            input_emb = input_emb * dropout_mask / keep_prob

        # Create separate conditioning tokens
        task_token = self.task_embedding(task_ids).unsqueeze(1)  # [batch_size, 1, d_model]

        time_emb = create_timestep_embedding(timesteps, self.d_model).to(device)
        time_token = self.time_projection(time_emb).unsqueeze(1)  # [batch_size, 1, d_model]

        # Concatenate in sequence dimension: [task, time, input_grid, noised_output]
        sequence = torch.cat([
            task_token,    # [batch_size, 1, d_model]
            time_token,    # [batch_size, 1, d_model]
            input_emb,     # [batch_size, max_size^2, d_model]
            xt_emb         # [batch_size, max_size^2, d_model]
        ], dim=1)  # [batch_size, 2 + 2*max_size^2, d_model]

        # Single transformer processes the entire sequence
        output = self.transformer(sequence)  # [batch_size, 2 + 2*max_size^2, d_model]

        # Extract predictions for noised output positions (skip task + time + input)
        output_start_idx = 2 + self.max_size * self.max_size
        output_preds = output[:, output_start_idx:, :]  # [batch_size, max_size^2, d_model]

        # Predict logits for each cell (only for colors 0-9)
        logits = self.output_head(output_preds)  # [batch_size, max_size^2, 10]
        logits = logits.view(batch_size, self.max_size, self.max_size, 10)

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
        embedding_dropout: float = 0.1,
        input_grid_dropout: float = 0.0,
        sc_dropout_prob: float = 0.5,
        include_size_head: bool = True,
        size_head_hidden_dim: int = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_size = max_size
        self.include_size_head = include_size_head
        self.d_model = d_model

        self.denoiser = TransformerDenoiser(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            max_size=max_size,
            max_tasks=max_tasks,
            embedding_dropout=embedding_dropout,
            input_grid_dropout=input_grid_dropout,
            sc_dropout_prob=sc_dropout_prob
        )

        # Integrated size prediction head (auxiliary task)
        if include_size_head:
            if size_head_hidden_dim is None:
                size_head_hidden_dim = int(d_model * 0.67)  # Default to 2/3 of d_model

            self.size_head = nn.Sequential(
                nn.Linear(d_model, size_head_hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(size_head_hidden_dim, size_head_hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
            self.height_head = nn.Linear(size_head_hidden_dim, max_size)
            self.width_head = nn.Linear(size_head_hidden_dim, max_size)

    def forward(
        self,
        xt: torch.Tensor,
        input_grid: torch.Tensor,
        task_ids: torch.Tensor,
        timesteps: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
        sc_p0: Optional[torch.Tensor] = None,
        sc_gain: float = 1.0,
    ) -> torch.Tensor:
        """Forward pass - predict x0 given xt."""
        return self.denoiser(xt, input_grid, task_ids, timesteps, masks, sc_p0, sc_gain)

    def _compute_bucket_metrics(
        self,
        correct: torch.Tensor,
        logits: torch.Tensor,
        targets: torch.Tensor,
        timesteps_norm: torch.Tensor,
        bucket_name: str
    ) -> Dict[str, float]:
        """Helper to compute metrics for a specific noise bucket."""
        # Define bucket ranges
        bucket_ranges = {
            'low_noise': (0.0, 0.33),
            'mid_noise': (0.33, 0.66),
            'high_noise': (0.66, 1.0)
        }

        if bucket_name not in bucket_ranges:
            return {}

        low, high = bucket_ranges[bucket_name]
        if high == 1.0:
            bucket_mask = timesteps_norm >= low
        else:
            bucket_mask = (timesteps_norm >= low) & (timesteps_norm < high)

        if bucket_mask.sum() == 0:
            return {}

        bucket_correct = correct[bucket_mask]
        bucket_logits = logits[bucket_mask]
        bucket_targets = targets[bucket_mask]

        bucket_acc = bucket_correct.mean().item()
        bucket_chance_adj = bucket_acc - 0.10
        bucket_ce = F.cross_entropy(bucket_logits, bucket_targets, reduction='mean').item()
        bucket_count = bucket_mask.sum().item()

        return {
            f'{bucket_name}_accuracy': bucket_acc,
            f'{bucket_name}_chance_adj_acc': bucket_chance_adj,
            f'{bucket_name}_cross_entropy': bucket_ce,
            f'{bucket_name}_count': bucket_count,
        }

    def compute_loss(
        self,
        x0: torch.Tensor,  # [batch_size, max_size, max_size] - clean output
        input_grid: torch.Tensor,  # [batch_size, max_size, max_size] - input
        task_ids: torch.Tensor,  # [batch_size] - task IDs
        xt: torch.Tensor,  # [batch_size, max_size, max_size] - noisy tokens
        timesteps: torch.Tensor,  # [batch_size] - timesteps
        heights: Optional[torch.Tensor] = None,  # [batch_size] - grid heights
        widths: Optional[torch.Tensor] = None,   # [batch_size] - grid widths
        auxiliary_size_loss_weight: float = 0.1,  # Weight for auxiliary size loss
        sc_p0: Optional[torch.Tensor] = None,  # [batch_size, max_size, max_size, 10] - self-conditioning probs
        sc_gain: float = 1.0,  # Self-conditioning gain factor
    ) -> Dict[str, torch.Tensor]:
        """Compute training losses with optional masking for pad regions."""
        batch_size = x0.shape[0]
        max_size = x0.shape[1]

        # Create masks for valid positions if heights/widths provided
        masks = None
        if heights is not None and widths is not None:
            masks = torch.zeros(batch_size, max_size, max_size, dtype=torch.float32, device=x0.device)
            for i in range(batch_size):
                h, w = heights[i].item(), widths[i].item()
                masks[i, :h, :w] = 1.0

        # Forward pass with masks and self-conditioning
        logits = self.forward(xt, input_grid, task_ids, timesteps, masks, sc_p0, sc_gain)

        # Create mask for valid positions if heights/widths provided
        if heights is not None and widths is not None:
            # Create mask [batch_size, max_size, max_size] - True for valid positions
            mask = torch.zeros(batch_size, max_size, max_size, dtype=torch.bool, device=x0.device)
            for i in range(batch_size):
                h, w = heights[i].item(), widths[i].item()
                mask[i, :h, :w] = True

            # Apply mask to flatten only valid positions
            valid_logits = logits[mask]  # [num_valid_positions, vocab_size]
            valid_targets = x0[mask]     # [num_valid_positions]

            # Compute loss only on valid positions
            grid_loss = F.cross_entropy(valid_logits, valid_targets, reduction='mean')

            # Compute accuracy metrics on valid positions only
            with torch.no_grad():
                predictions = torch.argmax(valid_logits, dim=-1)
                correct = (predictions == valid_targets).float()
                accuracy = correct.mean().item()
                chance_adjusted_acc = accuracy - 0.10  # 10 colors = 10% chance

                # Expand timesteps to match mask shape for per-timestep metrics
                timesteps_expanded = timesteps.unsqueeze(1).unsqueeze(2).expand_as(mask)
                valid_timesteps = timesteps_expanded[mask]  # [num_valid_positions]

                # Normalize timesteps to [0, 1] range (timesteps are 0-indexed)
                timesteps_norm = valid_timesteps.float() / 127.0  # 128 timesteps: 0-127

                # Compute metrics for each bucket
                low_metrics = self._compute_bucket_metrics(correct, valid_logits, valid_targets, timesteps_norm, 'low_noise')
                mid_metrics = self._compute_bucket_metrics(correct, valid_logits, valid_targets, timesteps_norm, 'mid_noise')
                high_metrics = self._compute_bucket_metrics(correct, valid_logits, valid_targets, timesteps_norm, 'high_noise')
        else:
            # Fallback to original behavior (all positions)
            grid_loss = F.cross_entropy(
                logits.view(-1, 10),  # Model outputs 10 classes (0-9), not vocab_size
                x0.view(-1),
                reduction='mean'
            )

            # Compute accuracy metrics on all positions
            with torch.no_grad():
                predictions = torch.argmax(logits, dim=-1)
                correct = (predictions == x0).float()
                accuracy = correct.mean().item()
                chance_adjusted_acc = accuracy - 0.10

                # Flatten for per-timestep metrics
                correct_flat = correct.view(-1)
                logits_flat = logits.view(-1, 10)  # Model outputs 10 classes
                targets_flat = x0.view(-1)

                # Expand timesteps to match flattened shape
                timesteps_flat = timesteps.unsqueeze(1).unsqueeze(2).expand_as(x0).view(-1)
                timesteps_norm = timesteps_flat.float() / 127.0  # 128 timesteps: 0-127

                # Compute metrics for each bucket
                low_metrics = self._compute_bucket_metrics(correct_flat, logits_flat, targets_flat, timesteps_norm, 'low_noise')
                mid_metrics = self._compute_bucket_metrics(correct_flat, logits_flat, targets_flat, timesteps_norm, 'mid_noise')
                high_metrics = self._compute_bucket_metrics(correct_flat, logits_flat, targets_flat, timesteps_norm, 'high_noise')

        # Compute auxiliary size loss if size head is included
        size_loss = torch.tensor(0.0, device=x0.device)
        size_metrics = {}

        if self.include_size_head and heights is not None and widths is not None:
            # Get size predictions
            height_logits, width_logits = self.predict_size(input_grid, task_ids)

            # Convert 1-indexed targets to 0-indexed for cross-entropy
            height_targets = heights - 1  # [batch_size]
            width_targets = widths - 1    # [batch_size]

            # Compute losses
            height_loss = F.cross_entropy(height_logits, height_targets)
            width_loss = F.cross_entropy(width_logits, width_targets)
            size_loss = height_loss + width_loss

            # Compute size accuracies
            with torch.no_grad():
                height_preds = torch.argmax(height_logits, dim=-1) + 1
                width_preds = torch.argmax(width_logits, dim=-1) + 1
                height_acc = (height_preds == heights).float().mean().item()
                width_acc = (width_preds == widths).float().mean().item()
                size_acc = ((height_preds == heights) & (width_preds == widths)).float().mean().item()

            size_metrics = {
                'size_loss': size_loss.item(),
                'height_loss': height_loss.item(),
                'width_loss': width_loss.item(),
                'height_accuracy': height_acc,
                'width_accuracy': width_acc,
                'size_accuracy': size_acc,
            }

        # Combine all metrics
        metrics = {
            'total_loss': grid_loss + auxiliary_size_loss_weight * size_loss,  # Add auxiliary loss with weight
            'grid_loss': grid_loss,
            'accuracy': accuracy,
            'chance_adjusted_accuracy': chance_adjusted_acc,
        }

        # Add bucket-specific metrics
        metrics.update(low_metrics)
        metrics.update(mid_metrics)
        metrics.update(high_metrics)

        # Add size metrics
        metrics.update(size_metrics)

        return metrics

    def predict_size(
        self,
        input_grid: torch.Tensor,
        task_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict output grid size using integrated size head.

        Returns:
            height_logits: [batch_size, max_size] - logits for height
            width_logits: [batch_size, max_size] - logits for width
        """
        if not self.include_size_head:
            raise ValueError("Size head not included in model")

        batch_size = input_grid.shape[0]

        # Flatten input grid
        input_flat = input_grid.view(batch_size, -1)  # [batch_size, max_size^2]

        # Embed tokens
        input_emb = self.denoiser.token_embedding(input_flat)  # [batch_size, max_size^2, d_model]

        # Add positional encoding
        pos_emb = self.denoiser.pos_encoding(batch_size)  # [batch_size, max_size^2, d_model]
        input_emb = input_emb + pos_emb

        # Apply embedding dropout
        input_emb = self.denoiser.embedding_dropout(input_emb)

        # Create task embedding
        task_emb = self.denoiser.task_embedding(task_ids)  # [batch_size, d_model]
        task_token = task_emb.unsqueeze(1)  # [batch_size, 1, d_model]

        # Create dummy time token (zeros for size prediction)
        time_token = torch.zeros_like(task_token)

        # Create sequence similar to main forward but simpler for size prediction
        sequence = torch.cat([
            task_token,     # [batch_size, 1, d_model]
            time_token,     # [batch_size, 1, d_model]
            input_emb,      # [batch_size, max_size^2, d_model]
        ], dim=1)

        # Process through transformer
        encoded_features = self.denoiser.transformer(sequence)

        # Extract features from input positions (skip task and time)
        input_features = encoded_features[:, 2:2+self.max_size*self.max_size, :]

        # Global average pooling
        pooled_features = input_features.mean(dim=1)  # [batch_size, d_model]

        # Pass through size head
        features = self.size_head(pooled_features)  # [batch_size, hidden_dim]

        # Get height and width logits
        height_logits = self.height_head(features)  # [batch_size, max_size]
        width_logits = self.width_head(features)    # [batch_size, max_size]

        return height_logits, width_logits

    def predict_sizes(
        self,
        input_grid: torch.Tensor,
        task_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict grid sizes (returns actual height/width values, not logits).

        Returns:
            heights: [batch_size] - predicted heights (1-indexed)
            widths: [batch_size] - predicted widths (1-indexed)
        """
        height_logits, width_logits = self.predict_size(input_grid, task_ids)

        # Get predictions
        predicted_heights = torch.argmax(height_logits, dim=-1) + 1
        predicted_widths = torch.argmax(width_logits, dim=-1) + 1

        return predicted_heights, predicted_widths


