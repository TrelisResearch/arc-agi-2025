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
    ):
        super().__init__()
        self.d_model = d_model
        self.max_size = max_size
        self.vocab_size = vocab_size

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)

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

        # Predict logits for each cell
        logits = self.output_head(output_preds)  # [batch_size, max_size^2, vocab_size]
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
        embedding_dropout: float = 0.1,
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
            max_tasks=max_tasks,
            embedding_dropout=embedding_dropout
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
        heights: Optional[torch.Tensor] = None,  # [batch_size] - grid heights
        widths: Optional[torch.Tensor] = None,   # [batch_size] - grid widths
    ) -> Dict[str, torch.Tensor]:
        """Compute training losses with optional masking for pad regions."""
        batch_size = x0.shape[0]
        max_size = x0.shape[1]

        # Forward pass
        logits = self.forward(xt, input_grid, task_ids, timesteps)

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

                # Create timestep buckets: low [0, 0.33), mid [0.33, 0.66), high [0.66, 1.0]
                low_mask = timesteps_norm < 0.33
                mid_mask = (timesteps_norm >= 0.33) & (timesteps_norm < 0.66)
                high_mask = timesteps_norm >= 0.66

                def compute_bucket_metrics(bucket_mask, bucket_name):
                    if bucket_mask.sum() == 0:
                        return {}

                    bucket_correct = correct[bucket_mask]
                    bucket_logits = valid_logits[bucket_mask]
                    bucket_targets = valid_targets[bucket_mask]

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

                # Compute metrics for each bucket
                low_metrics = compute_bucket_metrics(low_mask, 'low_noise')
                mid_metrics = compute_bucket_metrics(mid_mask, 'mid_noise')
                high_metrics = compute_bucket_metrics(high_mask, 'high_noise')
        else:
            # Fallback to original behavior (all positions)
            grid_loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
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
                predictions_flat = predictions.view(-1)
                targets_flat = x0.view(-1)
                correct_flat = correct.view(-1)

                # Expand timesteps to match flattened shape
                timesteps_flat = timesteps.unsqueeze(1).unsqueeze(2).expand_as(x0).view(-1)
                timesteps_norm = timesteps_flat.float() / 127.0  # 128 timesteps: 0-127

                # Create timestep buckets
                low_mask = timesteps_norm < 0.33
                mid_mask = (timesteps_norm >= 0.33) & (timesteps_norm < 0.66)
                high_mask = timesteps_norm >= 0.66

                def compute_bucket_metrics(bucket_mask, bucket_name):
                    if bucket_mask.sum() == 0:
                        return {}

                    bucket_correct = correct_flat[bucket_mask]
                    bucket_logits = logits.view(-1, self.vocab_size)[bucket_mask]
                    bucket_targets = targets_flat[bucket_mask]

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

                # Compute metrics for each bucket
                low_metrics = compute_bucket_metrics(low_mask, 'low_noise')
                mid_metrics = compute_bucket_metrics(mid_mask, 'mid_noise')
                high_metrics = compute_bucket_metrics(high_mask, 'high_noise')

        # Combine all metrics
        metrics = {
            'total_loss': grid_loss,
            'grid_loss': grid_loss,
            'accuracy': accuracy,
            'chance_adjusted_accuracy': chance_adjusted_acc,
        }

        # Add bucket-specific metrics
        metrics.update(low_metrics)
        metrics.update(mid_metrics)
        metrics.update(high_metrics)

        return metrics



class GridSizePredictionHead(nn.Module):
    """
    Neural network head for predicting output grid dimensions (height, width).

    Takes input grid + task embedding features from frozen diffusion model and
    predicts the height and width of the expected output grid.
    """

    def __init__(
        self,
        diffusion_model: ARCDiffusionModel,
        hidden_dim: int = 256,
        max_size: int = 30
    ):
        super().__init__()
        self.diffusion_model = diffusion_model
        self.max_size = max_size

        # Freeze the diffusion model parameters
        for param in self.diffusion_model.parameters():
            param.requires_grad = False

        # Feature extraction from diffusion model's input encoding
        self.feature_extractor = nn.Sequential(
            nn.Linear(diffusion_model.denoiser.d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Separate heads for height and width prediction
        self.height_head = nn.Linear(hidden_dim, max_size)
        self.width_head = nn.Linear(hidden_dim, max_size)

    def forward(
        self,
        input_grid: torch.Tensor,  # [batch_size, max_size, max_size]
        task_ids: torch.Tensor     # [batch_size]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict grid dimensions from input grid and task ID.

        Returns:
            height_logits: [batch_size, max_size] - logits over possible heights
            width_logits: [batch_size, max_size] - logits over possible widths
        """
        batch_size = input_grid.shape[0]

        with torch.no_grad():
            # Extract features using frozen diffusion model's embeddings
            # Flatten input grid
            input_flat = input_grid.view(batch_size, -1)  # [batch_size, max_size^2]

            # Embed tokens
            input_emb = self.diffusion_model.denoiser.token_embedding(input_flat)  # [batch_size, max_size^2, d_model]

            # Add positional encoding
            pos_emb = self.diffusion_model.denoiser.pos_encoding(batch_size)  # [batch_size, max_size^2, d_model]
            input_emb = input_emb + pos_emb

            # Add task conditioning
            task_emb = self.diffusion_model.denoiser.task_embedding(task_ids)  # [batch_size, d_model]
            task_emb_expanded = task_emb.unsqueeze(1).expand(-1, input_emb.shape[1], -1)  # [batch_size, max_size^2, d_model]
            input_features = input_emb + task_emb_expanded

            # Apply dropout like in main model
            input_features = self.diffusion_model.denoiser.embedding_dropout(input_features)

            # Global average pooling to get single feature vector per example
            pooled_features = input_features.mean(dim=1)  # [batch_size, d_model]

        # Extract features for size prediction (trainable)
        features = self.feature_extractor(pooled_features)  # [batch_size, hidden_dim]

        # Predict height and width
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
        height_logits, width_logits = self.forward(input_grid, task_ids)

        # Convert to probabilities and get most likely size
        height_probs = F.softmax(height_logits, dim=-1)
        width_probs = F.softmax(width_logits, dim=-1)

        # Get predicted indices and convert to 1-indexed sizes
        predicted_heights = torch.argmax(height_probs, dim=-1) + 1  # [batch_size]
        predicted_widths = torch.argmax(width_probs, dim=-1) + 1    # [batch_size]

        return predicted_heights, predicted_widths

    def compute_size_loss(
        self,
        input_grid: torch.Tensor,
        task_ids: torch.Tensor,
        target_heights: torch.Tensor,  # [batch_size] - 1-indexed heights
        target_widths: torch.Tensor    # [batch_size] - 1-indexed widths
    ) -> torch.Tensor:
        """Compute size prediction loss."""
        height_logits, width_logits = self.forward(input_grid, task_ids)

        # Convert 1-indexed targets to 0-indexed for cross-entropy
        height_targets = target_heights - 1  # [batch_size]
        width_targets = target_widths - 1    # [batch_size]

        height_loss = F.cross_entropy(height_logits, height_targets)
        width_loss = F.cross_entropy(width_logits, width_targets)

        return height_loss + width_loss

