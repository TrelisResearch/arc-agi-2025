#!/usr/bin/env python3
"""
Diffusion Visualization Script

Runs inference on a single example and displays the progression of diffusion
denoising step by step. Shows how the output grid evolves from noise to final prediction.
"""
import json
import argparse
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from experimental.diffusion.src.model import ARCDiffusionModel, GridSizePredictionHead
from experimental.diffusion.src.training import ARCDiffusionSampler
from experimental.diffusion.src.dataset import ARCDataset, load_arc_data_paths
from experimental.diffusion.utils.noise_scheduler import DiscreteNoiseScheduler
from experimental.diffusion.utils.grid_utils import grid_to_tokens, tokens_to_grid
from llm_python.utils.task_loader import TaskData, get_task_loader


def create_arc_colormap():
    """Create ARC-style colormap for visualization."""
    # ARC colors: black, blue, red, green, yellow, gray, magenta, orange, sky, brown
    arc_colors = [
        '#000000',  # 0: black
        '#0074D9',  # 1: blue
        '#FF4136',  # 2: red
        '#2ECC40',  # 3: green
        '#FFDC00',  # 4: yellow
        '#AAAAAA',  # 5: gray
        '#F012BE',  # 6: magenta
        '#FF851B',  # 7: orange
        '#7FDBFF',  # 8: sky blue
        '#870C25',  # 9: brown
        '#FFFFFF'   # 10: white (for PAD, shouldn't appear in valid regions)
    ]

    cmap = mcolors.ListedColormap(arc_colors[:10])  # Only use 0-9
    return cmap


class DiffusionVisualizer:
    """Visualizer for diffusion denoising progression."""

    def __init__(
        self,
        model_path: str,
        size_head_path: Optional[str] = None,
        device: Optional[str] = None,
        num_visualization_steps: int = 8,
        use_ground_truth_size: bool = False,
        dataset: str = "arc-prize-2024",
        subset: str = "evaluation"
    ):
        self.model_path = model_path
        self.size_head_path = size_head_path
        self.num_viz_steps = num_visualization_steps
        self.use_ground_truth_size = use_ground_truth_size
        self.dataset_name = dataset
        self.subset_name = subset

        # Set up device
        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')

        print(f"🔥 Loading model from {model_path}")
        print(f"🖥️ Using device: {self.device}")

        # Load model
        self.model, self.config = self._load_model()
        self.noise_scheduler = DiscreteNoiseScheduler(
            num_timesteps=self.config['num_timesteps'],
            vocab_size=self.config['vocab_size'],
            schedule_type=self.config['schedule_type']
        )
        self.noise_scheduler.to(self.device)

        # Create dataset for task-specific distributions
        data_paths = load_arc_data_paths(
            data_dir="data/arc-prize-2024",
            datasets=["training_challenges", "evaluation_challenges"]
        )
        self.dataset = ARCDataset(
            data_paths=data_paths['train'],
            max_size=self.config['max_size'],
            augment=False,  # No augmentation for inference
            include_training_test_examples=True
        )

        # Load size prediction head if provided
        self.size_head = None
        if self.size_head_path:
            print(f"🧠 Loading size prediction head from {self.size_head_path}")
            self.size_head = GridSizePredictionHead(
                diffusion_model=self.model,
                hidden_dim=256,
                max_size=self.config['max_size']
            )
            self.size_head.load_state_dict(torch.load(self.size_head_path, map_location=self.device))
            self.size_head.to(self.device)
            self.size_head.eval()

        # Create custom sampler for visualization
        self.sampler = VisualizationSampler(
            self.model,
            self.noise_scheduler,
            self.device,
            dataset=self.dataset,
            size_predictor=self.size_head
        )
        # Pass the ground truth preference to the sampler
        self.sampler.use_ground_truth_size = self.use_ground_truth_size

        print(f"✨ Model loaded: {self.model.__class__.__name__}")
        print(f"📊 Parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def _load_model(self) -> Tuple[ARCDiffusionModel, Dict]:
        """Load trained model from checkpoint"""
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        checkpoint = torch.load(self.model_path, map_location=self.device)
        config = checkpoint['config']
        dataset_info = checkpoint['dataset_info']

        # Recreate model
        model = ARCDiffusionModel(
            vocab_size=config['vocab_size'],
            d_model=config['d_model'],
            nhead=config['nhead'],
            num_layers=config['num_layers'],
            max_size=config['max_size'],
            max_tasks=dataset_info['num_tasks']
        )

        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()

        return model, config

    def visualize_task(self, task_id: str = None, test_idx: int = 0):
        """
        Visualize diffusion progression for a specific task.

        Args:
            task_id: Task ID to visualize (if None, uses first task)
            test_idx: Test example index within the task
        """
        # Load tasks using same approach as inference script
        task_loader = get_task_loader()
        tasks = task_loader.get_dataset_subset(f"{self.dataset_name}/{self.subset_name}", max_rows=10)

        if not tasks:
            raise ValueError(f"No tasks found for dataset: {self.dataset_name}/{self.subset_name}")

        # Use first task if none specified
        if task_id is None:
            task_id, task_data = tasks[0]
        else:
            # Find specified task
            task_data = None
            for tid, tdata in tasks:
                if tid == task_id:
                    task_data = tdata
                    break
            if task_data is None:
                raise ValueError(f"Task {task_id} not found")

        print(f"🎯 Visualizing task: {task_id}")
        print(f"📋 Test example: {test_idx}")
        print(f"🔍 Task data type: {type(task_data)}")
        if isinstance(task_data, dict):
            print(f"🔍 Task data keys: {list(task_data.keys())}")

        # Handle task data format (dict vs object)
        if hasattr(task_data, 'test'):
            # TaskData object format
            test_examples = task_data.test
        elif isinstance(task_data, dict) and 'test' in task_data:
            # Dict format
            test_examples = task_data['test']
        else:
            raise ValueError(f"Unknown task data format: {type(task_data)}")

        # Get test example
        if test_idx >= len(test_examples):
            raise ValueError(f"Test index {test_idx} out of range (task has {len(test_examples)} test examples)")

        test_example = test_examples[test_idx]

        # Handle test example format
        if hasattr(test_example, 'input'):
            # Object format
            input_grid = np.array(test_example.input)
            expected_output = np.array(test_example.output) if hasattr(test_example, 'output') and test_example.output else None
        elif isinstance(test_example, dict):
            # Dict format
            input_grid = np.array(test_example['input'])
            expected_output = np.array(test_example['output']) if 'output' in test_example and test_example['output'] else None
        else:
            raise ValueError(f"Unknown test example format: {type(test_example)}")

        print(f"📐 Input shape: {input_grid.shape}")
        if expected_output is not None:
            print(f"📐 Expected output shape: {expected_output.shape}")

        # Run visualization
        progression, timesteps, size_source = self.sampler.sample_with_progression(
            input_grid=input_grid,
            task_id=task_id,
            num_steps=self.num_viz_steps,
            ground_truth_size=(expected_output.shape if expected_output is not None else None)
        )

        # Create visualization
        self._create_visualization(
            input_grid=input_grid,
            expected_output=expected_output,
            progression=progression,
            timesteps=timesteps,
            size_source=size_source,
            task_id=task_id,
            test_idx=test_idx
        )

    def _create_visualization(
        self,
        input_grid: np.ndarray,
        expected_output: Optional[np.ndarray],
        progression: List[np.ndarray],
        timesteps: List[int],
        size_source: str,
        task_id: str,
        test_idx: int
    ):
        """Create and display the visualization plot."""
        cmap = create_arc_colormap()

        # Calculate layout
        num_steps = len(progression)
        num_cols = min(6, num_steps + 2)  # Input + expected + progression steps
        num_rows = (num_steps + 3) // num_cols  # Round up division

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(2*num_cols, 2*num_rows))
        if num_rows == 1:
            axes = axes.reshape(1, -1)

        # Flatten axes for easier indexing
        axes_flat = axes.flatten()

        # Plot input
        ax = axes_flat[0]
        ax.imshow(input_grid, cmap=cmap, vmin=0, vmax=9)
        ax.set_title("Input", fontsize=10, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])

        # Plot expected output if available
        plot_idx = 1
        if expected_output is not None:
            ax = axes_flat[plot_idx]
            ax.imshow(expected_output, cmap=cmap, vmin=0, vmax=9)
            ax.set_title("Expected", fontsize=10, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
            plot_idx += 1

        # Plot progression steps
        for i, step_grid in enumerate(progression):
            if plot_idx >= len(axes_flat):
                break

            ax = axes_flat[plot_idx]
            ax.imshow(step_grid, cmap=cmap, vmin=0, vmax=9)

            # Label steps with actual timesteps
            if i == 0:
                # Initial noise
                ax.set_title(f"Step 0\n(Initial Noise)", fontsize=9, fontweight='bold')
            else:
                # Use actual timestep from the sampling
                timestep = timesteps[i]
                ax.set_title(f"Step {i}\n(t={timestep})", fontsize=9)

            ax.set_xticks([])
            ax.set_yticks([])
            plot_idx += 1

        # Hide unused subplots
        for i in range(plot_idx, len(axes_flat)):
            axes_flat[i].set_visible(False)

        # Main title with size source information
        fig.suptitle(f"Diffusion Progression: {task_id} (test {test_idx})\n{size_source}", fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.show()

        # Also save the plot
        output_path = f"diffusion_viz_{task_id}_test{test_idx}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"💾 Saved visualization to: {output_path}")


class VisualizationSampler(ARCDiffusionSampler):
    """Extended sampler that captures intermediate steps for visualization."""

    def sample_with_progression(
        self,
        input_grid: np.ndarray,
        task_id: str,
        num_steps: int = 8,
        ground_truth_size: Optional[Tuple[int, int]] = None
    ) -> Tuple[List[np.ndarray], List[int], str]:
        """
        Sample with progression capture.

        Returns:
            Tuple of (progression grids, timesteps, size source description)
        """
        self.model.eval()

        # Convert input to tensor
        input_tokens, input_h, input_w = grid_to_tokens(input_grid, self.dataset.max_size)
        input_tensor = input_tokens.unsqueeze(0).to(self.device)  # [1, max_size, max_size]

        # Get task index
        if hasattr(self.dataset, 'task_id_to_idx') and task_id in self.dataset.task_id_to_idx:
            task_idx = self.dataset.task_id_to_idx[task_id]
        else:
            task_idx = 0  # Fallback
        task_tensor = torch.tensor([task_idx], device=self.device)

        # Determine sizes to use - prioritize size prediction unless ground truth explicitly requested
        predicted_heights = None
        predicted_widths = None
        size_source = None

        if ground_truth_size is not None and self.use_ground_truth_size:
            # Explicitly requested ground truth size
            h, w = ground_truth_size
            predicted_heights = torch.tensor([h], device=self.device)
            predicted_widths = torch.tensor([w], device=self.device)
            size_source = f"Ground Truth: {h} x {w}"
            print(f"🎯 Using ground truth size: {h} x {w}")
        elif self.size_predictor is not None:
            # Use size prediction (preferred method)
            predicted_heights, predicted_widths = self.size_predictor.predict_sizes(input_tensor, task_tensor)
            h, w = predicted_heights[0].item(), predicted_widths[0].item()
            size_source = f"Size Head Prediction: {h} x {w}"
            print(f"🧠 Predicted size: {h} x {w}")

            # Show expected size for comparison if available
            if ground_truth_size is not None:
                exp_h, exp_w = ground_truth_size
                accuracy_indicator = "✅" if (h == exp_h and w == exp_w) else "❌"
                print(f"   Expected: {exp_h} x {exp_w} {accuracy_indicator}")
        elif ground_truth_size is not None:
            # Fallback to ground truth if no size head
            h, w = ground_truth_size
            predicted_heights = torch.tensor([h], device=self.device)
            predicted_widths = torch.tensor([w], device=self.device)
            size_source = f"Ground Truth (fallback): {h} x {w}"
            print(f"🎯 Using ground truth size (no size head): {h} x {w}")
        else:
            size_source = "No size constraint (full 30x30 grid)"
            print("⚠️ No size information available - using full grid")

        # Initialize noise
        batch_size = 1
        max_size = input_tensor.shape[1]

        if self.dataset is not None:
            global_distribution = self.dataset.get_global_distribution().to(self.device)
            total_pixels = batch_size * max_size * max_size
            pixels = torch.multinomial(global_distribution, num_samples=total_pixels, replacement=True)
            x_t = pixels.view(batch_size, max_size, max_size)
        else:
            x_t = torch.randint(0, self.noise_scheduler.vocab_size, (batch_size, max_size, max_size), device=self.device)

        # Denoising with progression capture
        total_timesteps = self.noise_scheduler.num_timesteps
        # Use requested number of steps, interpolating between available timesteps
        timesteps = torch.linspace(total_timesteps - 1, 0, num_steps, dtype=torch.long, device=self.device)

        progression = []
        progression_timesteps = []  # Track which timesteps we capture
        # Capture all steps (initial noise + all denoising steps)
        capture_indices = np.arange(len(timesteps))  # Capture every step

        # Capture initial noise (step 0)
        grid_np = x_t[0].cpu().numpy()
        if predicted_heights is not None and predicted_widths is not None:
            h, w = predicted_heights[0].item(), predicted_widths[0].item()
            valid_grid = grid_np[:h, :w]
        else:
            # Use full grid when no size predictions available
            valid_grid = grid_np
        progression.append(valid_grid)
        progression_timesteps.append(total_timesteps)  # Initial noise is "before" timestep 0

        for i, t in enumerate(tqdm(timesteps, desc="Diffusion sampling")):
            t_batch = t.repeat(batch_size)

            # Forward pass
            logits = self.model(x_t, input_tensor, task_tensor, t_batch)
            x_t = torch.argmax(logits, dim=-1)

            # Apply size masking
            if predicted_heights is not None and predicted_widths is not None:
                h, w = predicted_heights[0].item(), predicted_widths[0].item()
                if h < max_size:
                    x_t[0, h:, :] = 0
                if w < max_size:
                    x_t[0, :, w:] = 0

            # Capture progression at selected steps
            if i in capture_indices:
                # Extract valid region
                grid_np = x_t[0].cpu().numpy()
                if predicted_heights is not None and predicted_widths is not None:
                    h, w = predicted_heights[0].item(), predicted_widths[0].item()
                    valid_grid = grid_np[:h, :w]
                else:
                    # Use full grid when no size predictions available
                    valid_grid = grid_np

                progression.append(valid_grid)
                progression_timesteps.append(t.item())

        return progression, progression_timesteps, size_source


def main():
    parser = argparse.ArgumentParser(description="Visualize ARC Diffusion Progression")

    parser.add_argument("--model-path", required=True, help="Path to trained diffusion model")
    parser.add_argument("--size-head-path", help="Path to trained size prediction head (optional)")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps", "auto"], default="auto", help="Device to use")
    parser.add_argument("--dataset", default="arc-prize-2024", help="Dataset to use (default: arc-prize-2024)")
    parser.add_argument("--subset", default="evaluation", help="Subset to use (default: evaluation)")
    parser.add_argument("--task-id", help="Specific task ID to visualize (default: first task)")
    parser.add_argument("--test-idx", type=int, default=0, help="Test example index (default: 0)")
    parser.add_argument("--steps", type=int, default=8, help="Number of visualization steps (default: 8)")
    parser.add_argument("--use-ground-truth-size", action="store_true", help="Use ground truth output size instead of size prediction")

    args = parser.parse_args()

    # Validate model path
    if not Path(args.model_path).exists():
        print(f"❌ Model not found: {args.model_path}")
        sys.exit(1)

    if args.size_head_path and not Path(args.size_head_path).exists():
        print(f"❌ Size head not found: {args.size_head_path}")
        sys.exit(1)

    print("🎨 Diffusion Visualization")
    print("=" * 50)
    print(f"Model: {args.model_path}")
    if args.size_head_path:
        print(f"Size head: {args.size_head_path}")
    print(f"Device: {args.device}")
    print(f"Visualization steps: {args.steps}")
    if args.use_ground_truth_size:
        print("🎯 Using ground truth size (not size prediction)")
    print("=" * 50)

    # Create visualizer
    visualizer = DiffusionVisualizer(
        model_path=args.model_path,
        size_head_path=args.size_head_path,
        device=args.device if args.device != "auto" else None,
        num_visualization_steps=args.steps,
        use_ground_truth_size=args.use_ground_truth_size,
        dataset=args.dataset,
        subset=args.subset
    )

    # Run visualization
    try:
        visualizer.visualize_task(
            task_id=args.task_id,
            test_idx=args.test_idx
        )
        print("✅ Visualization complete!")
    except Exception as e:
        print(f"❌ Visualization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()