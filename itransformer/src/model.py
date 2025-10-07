"""
Iterative refinement transformer for ARC tasks with size prediction.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Union


class CoordinatePositionalEncoding(nn.Module):
    """2D coordinate-based positional encoding for grid positions."""

    def __init__(self, max_size: int = 30, d_model: int = 256):
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


class TransformerRefiner(nn.Module):
    """
    Transformer backbone for iterative refinement.
    Processes 900 tokens (30x30 grid) with task and step conditioning.
    """

    def __init__(
        self,
        vocab_size: int = 11,
        d_model: int = 384,
        nhead: int = 6,
        num_layers: int = 8,
        max_size: int = 30,
        max_tasks: int = 1000,  # Maximum number of task IDs
        max_steps: int = 20,  # Maximum refinement steps
        embedding_dropout: float = 0.1,
        input_grid_dropout: float = 0.0,  # Dropout probability for input grid conditioning
    ):
        super().__init__()
        self.d_model = d_model
        self.max_size = max_size
        self.vocab_size = vocab_size
        self.input_grid_dropout = input_grid_dropout

        # Token embedding with padding_idx for PAD token
        self.PAD_ID = 10
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=self.PAD_ID)

        # Positional encoding
        self.pos_encoding = CoordinatePositionalEncoding(max_size, d_model)

        # Task embedding (for task conditioning)
        self.task_embedding = nn.Embedding(max_tasks, d_model)


        # Augmentation embeddings
        self.d4_embedding = nn.Embedding(8, d_model)  # D4 group: 8 spatial transformations (0-7)
        self.color_shift_embedding = nn.Embedding(9, d_model)  # 0-8 color cycle offset

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
        self.output_head = nn.Linear(d_model, 10)

    def forward(
        self,
        x_prev: torch.Tensor,  # [batch_size, max_size, max_size] - previous prediction
        input_grid: torch.Tensor,  # [batch_size, max_size, max_size] - input grid
        task_ids: torch.Tensor,  # [batch_size] - task IDs
        step_idx: torch.Tensor,  # [batch_size] - refinement step (0 to K-1)
        d4_idx: Optional[torch.Tensor] = None,  # [batch_size] - D4 transformation index (0-7)
        color_shift: Optional[torch.Tensor] = None,  # [batch_size] - color shift (0-8)
        masks: Optional[torch.Tensor] = None,  # [batch_size, max_size, max_size]
    ) -> torch.Tensor:
        """
        Forward pass of the refiner.

        Returns:
            logits: [batch_size, max_size, max_size, 10] - predicted logits for colors 0-9
        """
        batch_size = x_prev.shape[0]
        device = x_prev.device

        # Flatten grids
        input_flat = input_grid.view(batch_size, -1)  # [batch_size, max_size^2]
        x_prev_flat = x_prev.view(batch_size, -1)  # [batch_size, max_size^2]

        # Embed tokens
        input_emb = self.token_embedding(input_flat)  # [batch_size, max_size^2, d_model]
        x_prev_emb = self.token_embedding(x_prev_flat)  # [batch_size, max_size^2, d_model]

        # Add positional encoding to both
        pos_emb = self.pos_encoding(batch_size)  # [batch_size, max_size^2, d_model]
        input_emb = input_emb + pos_emb
        x_prev_emb = x_prev_emb + pos_emb

        # Apply embedding dropout
        input_emb = self.embedding_dropout(input_emb)
        x_prev_emb = self.embedding_dropout(x_prev_emb)

        # Apply masking to x_prev features if masks provided
        # Zero out embeddings outside valid regions
        if masks is not None:
            masks_flat = masks.view(batch_size, -1, 1).float()  # [batch_size, max_size^2, 1]
            x_prev_emb = x_prev_emb * masks_flat  # Zero out invalid regions

        # Apply input grid conditioning dropout (training only)
        if self.training and self.input_grid_dropout > 0:
            # Sample Bernoulli gates for each batch item
            keep_prob = 1.0 - self.input_grid_dropout
            dropout_mask = torch.bernoulli(torch.full((batch_size, 1, 1), keep_prob, device=device))
            # Apply dropout with scaling to maintain expectation
            input_emb = input_emb * dropout_mask / keep_prob

        # Create separate conditioning tokens
        task_token = self.task_embedding(task_ids).unsqueeze(1)  # [batch_size, 1, d_model]

        # Add augmentation tokens if provided, otherwise use zeros (no augmentation)
        if d4_idx is None:
            d4_idx = torch.zeros(batch_size, dtype=torch.long, device=device)
        if color_shift is None:
            color_shift = torch.zeros(batch_size, dtype=torch.long, device=device)

        d4_token = self.d4_embedding(d4_idx).unsqueeze(1)  # [batch_size, 1, d_model]
        color_shift_token = self.color_shift_embedding(color_shift).unsqueeze(1)  # [batch_size, 1, d_model]

        # Concatenate in sequence dimension: [task, d4, color_shift, input_grid, prev_output]
        sequence = torch.cat([
            task_token,         # [batch_size, 1, d_model]
            d4_token,           # [batch_size, 1, d_model]
            color_shift_token,  # [batch_size, 1, d_model]
            input_emb,          # [batch_size, max_size^2, d_model]
            x_prev_emb          # [batch_size, max_size^2, d_model]
        ], dim=1)  # [batch_size, 3 + 2*max_size^2, d_model]

        # Single transformer processes the entire sequence
        output = self.transformer(sequence)  # [batch_size, 3 + 2*max_size^2, d_model]

        # Extract predictions for previous output positions (skip task + d4 + color_shift + input)
        output_start_idx = 3 + self.max_size * self.max_size
        output_preds = output[:, output_start_idx:, :]  # [batch_size, max_size^2, d_model]

        # Predict logits for each cell (only for colors 0-9)
        logits = self.output_head(output_preds)  # [batch_size, max_size^2, 10]
        logits = logits.view(batch_size, self.max_size, self.max_size, 10)

        return logits


class ARCIterativeModel(nn.Module):
    """
    Complete iterative refinement model for ARC tasks.
    Combines the refiner with loss computation and sampling logic.
    """

    def __init__(
        self,
        vocab_size: int = 11,
        d_model: int = 384,
        nhead: int = 6,
        num_layers: int = 8,
        max_size: int = 30,
        max_tasks: int = 1000,
        max_steps: int = 20,
        embedding_dropout: float = 0.1,
        input_grid_dropout: float = 0.0,
        include_size_head: bool = True,
        size_head_hidden_dim: int = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_size = max_size
        self.include_size_head = include_size_head
        self.d_model = d_model

        self.refiner = TransformerRefiner(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            max_size=max_size,
            max_tasks=max_tasks,
            max_steps=max_steps,
            embedding_dropout=embedding_dropout,
            input_grid_dropout=input_grid_dropout,
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
        x_prev: torch.Tensor,
        input_grid: torch.Tensor,
        task_ids: torch.Tensor,
        step_idx: torch.Tensor,
        d4_idx: Optional[torch.Tensor] = None,
        color_shift: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass - predict next refinement given previous."""
        return self.refiner(x_prev, input_grid, task_ids, step_idx, d4_idx, color_shift, masks)

    def compute_loss(
        self,
        x0: torch.Tensor,  # [batch_size, max_size, max_size] - ground truth
        input_grid: torch.Tensor,  # [batch_size, max_size, max_size] - input
        task_ids: torch.Tensor,  # [batch_size] - task IDs
        x_prev: torch.Tensor,  # [batch_size, max_size, max_size] - previous prediction
        step_idx: torch.Tensor,  # [batch_size] - refinement step
        d4_idx: Optional[torch.Tensor] = None,  # [batch_size] - D4 transformation index
        color_shift: Optional[torch.Tensor] = None,  # [batch_size] - color shift
        heights: Optional[torch.Tensor] = None,  # [batch_size] - grid heights
        widths: Optional[torch.Tensor] = None,   # [batch_size] - grid widths
        masks: Optional[torch.Tensor] = None,  # [batch_size, max_size, max_size] - precomputed masks
        auxiliary_size_loss_weight: float = 0.1,  # Weight for auxiliary size loss
        return_logits: bool = False,  # If True, return (metrics, logits) tuple
    ) -> Union[Dict[str, torch.Tensor], Tuple[Dict[str, torch.Tensor], torch.Tensor]]:
        """Compute training losses with optional masking for pad regions."""
        batch_size = x0.shape[0]
        max_size = x0.shape[1]

        # Create masks for valid positions if not provided
        mask_bool = None
        if masks is None and heights is not None and widths is not None:
            from ..utils.grid_utils import batch_create_masks
            masks = batch_create_masks(heights, widths, max_size)
            mask_bool = masks.bool()
        elif masks is not None:
            # Masks passed in are already boolean from training.py
            mask_bool = masks

        # Forward pass with masks
        logits = self.forward(
            x_prev=x_prev,
            input_grid=input_grid,
            task_ids=task_ids,
            step_idx=step_idx,
            d4_idx=d4_idx,
            color_shift=color_shift,
            masks=masks,
        )

        # Apply mask for loss computation if provided
        if mask_bool is not None:
            # Apply mask to flatten only valid positions
            valid_logits = logits[mask_bool]  # [num_valid_positions, vocab_size]
            valid_targets = x0[mask_bool]     # [num_valid_positions]

            # Compute loss only on valid positions
            grid_loss = F.cross_entropy(valid_logits, valid_targets, reduction='mean')

            # Compute simple accuracy only
            with torch.no_grad():
                predictions = torch.argmax(valid_logits, dim=-1)
                accuracy = (predictions == valid_targets).float().mean().item()
        else:
            # Fallback to original behavior (all positions)
            grid_loss = F.cross_entropy(
                logits.view(-1, 10),
                x0.view(-1),
                reduction='mean'
            )

            # Compute simple accuracy only
            with torch.no_grad():
                predictions = torch.argmax(logits, dim=-1)
                accuracy = (predictions == x0).float().mean().item()

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
            'total_loss': grid_loss + auxiliary_size_loss_weight * size_loss,
            'grid_loss': grid_loss,
            'accuracy': accuracy,
        }

        # Add size metrics
        metrics.update(size_metrics)

        if return_logits:
            return metrics, logits
        return metrics

    def predict_size(
        self,
        input_grid: torch.Tensor,
        task_ids: torch.Tensor,
        d4_idx: Optional[torch.Tensor] = None,
        color_shift: Optional[torch.Tensor] = None
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
        device = input_grid.device

        # Flatten input grid
        input_flat = input_grid.view(batch_size, -1)  # [batch_size, max_size^2]

        # Embed tokens
        input_emb = self.refiner.token_embedding(input_flat)  # [batch_size, max_size^2, d_model]

        # Add positional encoding
        pos_emb = self.refiner.pos_encoding(batch_size)  # [batch_size, max_size^2, d_model]
        input_emb = input_emb + pos_emb

        # Apply embedding dropout
        input_emb = self.refiner.embedding_dropout(input_emb)

        # Create task embedding
        task_emb = self.refiner.task_embedding(task_ids)  # [batch_size, d_model]
        task_token = task_emb.unsqueeze(1)  # [batch_size, 1, d_model]

        # Add augmentation tokens if provided, otherwise use zeros (no augmentation)
        if d4_idx is None:
            d4_idx = torch.zeros(batch_size, dtype=torch.long, device=device)
        if color_shift is None:
            color_shift = torch.zeros(batch_size, dtype=torch.long, device=device)

        d4_token = self.refiner.d4_embedding(d4_idx).unsqueeze(1)  # [batch_size, 1, d_model]
        color_shift_token = self.refiner.color_shift_embedding(color_shift).unsqueeze(1)  # [batch_size, 1, d_model]

        # Create sequence matching the main forward pass structure
        sequence = torch.cat([
            task_token,         # [batch_size, 1, d_model]
            d4_token,           # [batch_size, 1, d_model]
            color_shift_token,  # [batch_size, 1, d_model]
            input_emb,          # [batch_size, max_size^2, d_model]
        ], dim=1)

        # Process through transformer
        encoded_features = self.refiner.transformer(sequence)

        # Extract features from input positions (skip task, d4, color_shift)
        input_features = encoded_features[:, 3:3+self.max_size*self.max_size, :]

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
        task_ids: torch.Tensor,
        d4_idx: Optional[torch.Tensor] = None,
        color_shift: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict grid sizes (returns actual height/width values, not logits).

        Returns:
            heights: [batch_size] - predicted heights (1-indexed)
            widths: [batch_size] - predicted widths (1-indexed)
        """
        height_logits, width_logits = self.predict_size(input_grid, task_ids, d4_idx, color_shift)

        # Get predictions
        predicted_heights = torch.argmax(height_logits, dim=-1) + 1
        predicted_widths = torch.argmax(width_logits, dim=-1) + 1

        return predicted_heights, predicted_widths
