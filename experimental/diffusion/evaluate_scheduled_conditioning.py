#!/usr/bin/env python3
"""
ARC Diffusion Model Evaluation with Scheduled Input Conditioning

This variant turns off input grid conditioning for the first 60% of denoising steps,
then linearly ramps it up to full strength by the end. This allows the model to:
1. Generate freely from noise in early steps
2. Gradually incorporate input constraints in later steps

Usage:
    python experimental/diffusion/evaluate_scheduled_conditioning.py \
        --config experimental/diffusion/configs/smol_config.json \
        --limit 10
"""
import json
import argparse
import sys
from pathlib import Path
from typing import Optional
from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from experimental.diffusion.src.model import ARCDiffusionModel
from experimental.diffusion.src.dataset import ARCDataset, load_arc_data_paths
from experimental.diffusion.utils.noise_scheduler import DiscreteNoiseScheduler
from experimental.diffusion.utils.grid_utils import grid_to_tokens, tokens_to_grid


class ScheduledConditioningSampler:
    """Sampler that applies input conditioning on a schedule during denoising."""

    def __init__(
        self,
        model: ARCDiffusionModel,
        noise_scheduler: DiscreteNoiseScheduler,
        device: torch.device,
        conditioning_start_fraction: float = 0.6,  # Start conditioning at 60% through
        debug: bool = False
    ):
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.device = device
        self.conditioning_start_fraction = conditioning_start_fraction
        self.debug = debug

    def _get_conditioning_weight(self, step: int, total_steps: int) -> float:
        """
        Calculate input conditioning weight for current step.

        Returns 0 for first 60% of steps, then linearly ramps to 1.0 by end.
        """
        progress = step / (total_steps - 1)  # 0 to 1

        if progress < self.conditioning_start_fraction:
            return 0.0
        else:
            # Linear ramp from 0 to 1 over remaining steps
            ramp_progress = (progress - self.conditioning_start_fraction) / (1.0 - self.conditioning_start_fraction)
            return ramp_progress

    @torch.no_grad()
    def sample(
        self,
        input_grids: torch.Tensor,
        task_indices: torch.Tensor,
        num_inference_steps: Optional[int] = None,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Sample outputs with scheduled input conditioning.

        Args:
            input_grids: [batch_size, max_size, max_size]
            task_indices: [batch_size]
            num_inference_steps: Number of denoising steps
            temperature: Sampling temperature

        Returns:
            predictions: [batch_size, max_size, max_size]
        """
        self.model.eval()

        batch_size = input_grids.shape[0]
        max_size = input_grids.shape[1]

        if num_inference_steps is None:
            num_inference_steps = self.noise_scheduler.num_timesteps

        # Predict output grid sizes
        predicted_heights = None
        predicted_widths = None
        if hasattr(self.model, 'include_size_head') and self.model.include_size_head:
            predicted_heights, predicted_widths = self.model.predict_sizes(input_grids, task_indices)
            if self.debug:
                print(f"Predicted sizes: heights={predicted_heights.cpu().tolist()}, widths={predicted_widths.cpu().tolist()}")

        # Initialize with uniform random noise
        x_t = torch.randint(0, 10, (batch_size, max_size, max_size), device=self.device)

        # Create a "null" input grid (all zeros) for unconditional generation
        null_input = torch.zeros_like(input_grids)

        # Denoising loop
        timesteps = torch.linspace(num_inference_steps - 1, 0, num_inference_steps, dtype=torch.long, device=self.device)

        for i, t in enumerate(tqdm(timesteps, desc="Sampling (scheduled conditioning)", disable=(not self.debug))):
            t_batch = t.repeat(batch_size)

            # Get conditioning weight for this step
            weight = self._get_conditioning_weight(i, num_inference_steps)

            if self.debug and i % (num_inference_steps // 10) == 0:
                print(f"\nStep {i}/{num_inference_steps}, t={t.item()}, conditioning weight={weight:.3f}")

            # Interpolate between null input and actual input based on weight
            conditioned_input = weight * input_grids + (1 - weight) * null_input

            # Forward pass with scheduled conditioning
            logits = self.model(x_t, conditioned_input, task_indices, t_batch)

            # Apply temperature scaling
            logits_scaled = logits / temperature
            probs = F.softmax(logits_scaled, dim=-1)

            # Sample
            if t > 0:
                x_t = torch.multinomial(probs.view(-1, probs.shape[-1]), 1).view(x_t.shape)
            else:
                x_t = torch.argmax(logits_scaled, dim=-1)

            # Apply size masking
            if predicted_heights is not None and predicted_widths is not None:
                for b in range(batch_size):
                    h, w = predicted_heights[b].item(), predicted_widths[b].item()
                    if h < max_size:
                        x_t[b, h:, :] = 0
                    if w < max_size:
                        x_t[b, :, w:] = 0

        return x_t


def main():
    parser = argparse.ArgumentParser(description="Evaluate ARC diffusion with scheduled conditioning")
    parser.add_argument("--config", required=True, help="Path to config JSON file")
    parser.add_argument("--limit", type=int, default=None, help="Limit evaluation to N tasks")
    parser.add_argument("--conditioning-start", type=float, default=0.6,
                       help="Start conditioning at this fraction through denoising (default: 0.6)")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")

    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path, 'r') as f:
        config = json.load(f)

    # Flatten config
    flat_config = {}
    for section in ['model', 'training', 'data', 'output']:
        if section in config:
            flat_config.update(config[section])
    flat_config['auxiliary_loss'] = config.get('auxiliary_loss', {})

    # Set up device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f"Using device: {device}")
    print(f"Conditioning starts at {args.conditioning_start*100:.0f}% through denoising")

    # Load model
    output_dir = Path(flat_config['output_dir'])
    model_path = output_dir / 'best_model.pt'

    if not model_path.exists():
        print(f"Model not found: {model_path}")
        sys.exit(1)

    checkpoint = torch.load(model_path, map_location=device)
    model_config = checkpoint['config']
    dataset_info = checkpoint['dataset_info']

    # Get auxiliary loss config
    aux_config = model_config.get('auxiliary_loss', {})

    # Create model
    model = ARCDiffusionModel(
        vocab_size=model_config['vocab_size'],
        d_model=model_config['d_model'],
        nhead=model_config['nhead'],
        num_layers=model_config['num_layers'],
        max_size=model_config['max_size'],
        max_tasks=dataset_info['num_tasks'],
        embedding_dropout=model_config.get('embedding_dropout', 0.1),
        include_size_head=aux_config.get('include_size_head', True),
        size_head_hidden_dim=aux_config.get('size_head_hidden_dim', None)
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"Loaded model from {model_path}")
    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")

    # Create noise scheduler
    noise_scheduler = DiscreteNoiseScheduler(
        num_timesteps=model_config['num_timesteps'],
        vocab_size=model_config['vocab_size'],
        schedule_type=model_config['schedule_type']
    )
    noise_scheduler.to(device)

    # Create sampler with scheduled conditioning
    sampler = ScheduledConditioningSampler(
        model=model,
        noise_scheduler=noise_scheduler,
        device=device,
        conditioning_start_fraction=args.conditioning_start,
        debug=args.debug
    )

    # Load dataset for evaluation
    data_paths = load_arc_data_paths(
        data_dir=flat_config.get('data_dir', 'data/arc-prize-2024'),
        datasets=['training_challenges']
    )

    dataset = ARCDataset(
        data_paths=data_paths['train'],
        max_size=model_config['max_size'],
        augment=False,
        include_training_test_examples=False  # Only training examples
    )

    task_id_to_idx = dataset.task_id_to_idx

    # Load task data
    with open('data/arc-prize-2024/arc-agi_training_challenges.json', 'r') as f:
        task_data = json.load(f)

    # Evaluate on tasks
    num_tasks = min(args.limit, len(task_data)) if args.limit else len(task_data)
    num_correct = 0
    num_total = 0

    for task_idx, (task_id, task) in enumerate(task_data.items()):
        if args.limit and task_idx >= args.limit:
            break

        if task_id not in task_id_to_idx:
            continue

        print(f"\n[{task_idx+1}/{num_tasks}] Task {task_id}")

        # Get task index
        task_tensor_idx = task_id_to_idx[task_id]

        # Process each test example
        for test_idx, test_example in enumerate(task['test']):
            input_grid = np.array(test_example['input'])

            # Check if we have ground truth
            if 'output' not in test_example:
                print(f"  Test {test_idx}: No ground truth, skipping")
                continue

            expected = np.array(test_example['output'])

            # Convert to tensors
            input_tokens, input_h, input_w = grid_to_tokens(input_grid, model_config['max_size'])
            input_batch = input_tokens.unsqueeze(0).to(device)
            task_batch = torch.tensor([task_tensor_idx], device=device)

            # Sample with scheduled conditioning
            prediction = sampler.sample(
                input_grids=input_batch,
                task_indices=task_batch,
                num_inference_steps=model_config['num_timesteps']
            )

            # Extract predicted grid
            pred_grid = tokens_to_grid(
                prediction[0].cpu(),
                height=expected.shape[0],
                width=expected.shape[1]
            )

            # Check correctness
            correct = np.array_equal(pred_grid, expected)
            num_total += 1
            if correct:
                num_correct += 1

            print(f"  Test {test_idx}: {'✓ CORRECT' if correct else '✗ WRONG'}")

    # Print final results
    accuracy = num_correct / num_total if num_total > 0 else 0
    print(f"\n{'='*60}")
    print(f"Final Results with Scheduled Conditioning")
    print(f"Conditioning: OFF for first {args.conditioning_start*100:.0f}% → linearly ramps to ON")
    print(f"{'='*60}")
    print(f"Correct: {num_correct}/{num_total} ({accuracy*100:.1f}%)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()