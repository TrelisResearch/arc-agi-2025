#!/usr/bin/env python3
"""
Detailed timing breakdown of a single training step.

Usage:
    uv run python experimental/diffusion/benchmark_step.py --config experimental/diffusion/configs/test_config.json
"""
import argparse
import json
import sys
import torch
from pathlib import Path
import time
from contextlib import contextmanager

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from experimental.diffusion.src.model import ARCDiffusionModel
from experimental.diffusion.src.dataset import ARCDataset, load_arc_data_paths, collate_fn
from experimental.diffusion.utils.noise_scheduler import DiscreteNoiseScheduler
from experimental.diffusion.train_diffusion_backbone import load_and_flatten_config
from torch.utils.data import DataLoader


@contextmanager
def timer(name, timings):
    """Context manager to time operations."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        torch.mps.synchronize()

    start = time.perf_counter()
    yield

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        torch.mps.synchronize()

    elapsed = (time.perf_counter() - start) * 1000  # Convert to ms
    timings[name] = elapsed


def benchmark_step(config_path: str, num_iterations: int = 10):
    """Benchmark a single training step with detailed timing."""

    config = load_and_flatten_config(config_path)

    # Set up device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f"🔍 Benchmarking Training Step")
    print(f"🖥️ Device: {device}")
    print(f"🔄 Iterations: {num_iterations}")
    print(f"=" * 60)

    # Load minimal dataset
    data_paths = load_arc_data_paths(
        data_dir=config.get('data_dir', 'data/arc-prize-2025'),
        datasets=config.get('datasets', None)
    )

    dataset = ARCDataset(
        data_paths=data_paths['train'][:100],  # Just 100 tasks
        max_size=config['max_size'],
        augment=False,  # No augmentation for benchmarking
        n_augment=1
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )

    # Create model
    aux_config = config.get('auxiliary_loss', {})
    model = ARCDiffusionModel(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        max_size=config['max_size'],
        embedding_dropout=config.get('embedding_dropout', 0.1),
        input_grid_dropout=config.get('input_grid_dropout', 0.0),
        sc_dropout_prob=config.get('sc_dropout_prob', 0.5),
        include_size_head=aux_config.get('include_size_head', False),
        size_head_hidden_dim=aux_config.get('size_head_hidden_dim', 256)
    ).to(device)

    noise_scheduler = DiscreteNoiseScheduler(
        num_timesteps=config['num_timesteps'],
        schedule_type=config['schedule_type']
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config.get('weight_decay', 0.01)
    )

    use_mixed_precision = config.get('use_mixed_precision', True) and device.type in ['cuda', 'mps']
    amp_dtype = torch.bfloat16 if device.type == 'cuda' and torch.cuda.is_bf16_supported() else torch.float16

    # Get a batch
    batch = next(iter(dataloader))

    # Warm up
    print("🔥 Warming up...")
    for _ in range(5):
        input_grids = batch['input_grid'].to(device)
        output_grids = batch['output_grid'].to(device)
        task_indices = batch['task_idx'].to(device)
        heights = batch['height'].to(device)
        widths = batch['width'].to(device)
        rotations = batch['rotation'].to(device)
        flips = batch['flip'].to(device)
        color_shifts = batch['color_shift'].to(device)

        timesteps = torch.randint(0, noise_scheduler.num_timesteps, (input_grids.shape[0],), device=device)
        from experimental.diffusion.utils.grid_utils import batch_create_masks
        masks = batch_create_masks(heights, widths, model.max_size).to(device)
        noisy_grids = noise_scheduler.add_noise(output_grids, timesteps, masks)

        if use_mixed_precision:
            with torch.autocast(device_type=device.type, dtype=amp_dtype):
                losses = model.compute_loss(
                    x0=output_grids,
                    input_grid=input_grids,
                    task_ids=task_indices,
                    xt=noisy_grids,
                    timesteps=timesteps,
                    rotation=rotations,
                    flip=flips,
                    color_shift=color_shifts,
                    heights=heights,
                    widths=widths,
                    auxiliary_size_loss_weight=0.1
                )
        else:
            losses = model.compute_loss(
                x0=output_grids,
                input_grid=input_grids,
                task_ids=task_indices,
                xt=noisy_grids,
                timesteps=timesteps,
                rotation=rotations,
                flip=flips,
                color_shift=color_shifts,
                heights=heights,
                widths=widths,
                auxiliary_size_loss_weight=0.1
            )

        optimizer.zero_grad()
        losses['total_loss'].backward()
        optimizer.step()

    print("✅ Warmup complete\n")

    # Benchmark
    all_timings = []

    for i in range(num_iterations):
        timings = {}

        with timer("1_data_to_device", timings):
            input_grids = batch['input_grid'].to(device)
            output_grids = batch['output_grid'].to(device)
            task_indices = batch['task_idx'].to(device)
            heights = batch['height'].to(device)
            widths = batch['width'].to(device)
            rotations = batch['rotation'].to(device)
            flips = batch['flip'].to(device)
            color_shifts = batch['color_shift'].to(device)

        with timer("2_noise_setup", timings):
            timesteps = torch.randint(0, noise_scheduler.num_timesteps, (input_grids.shape[0],), device=device)
            from experimental.diffusion.utils.grid_utils import batch_create_masks
            masks = batch_create_masks(heights, widths, model.max_size).to(device)
            noisy_grids = noise_scheduler.add_noise(output_grids, timesteps, masks)

        with timer("3_forward_pass", timings):
            if use_mixed_precision:
                with torch.autocast(device_type=device.type, dtype=amp_dtype):
                    losses = model.compute_loss(
                        x0=output_grids,
                        input_grid=input_grids,
                        task_ids=task_indices,
                        xt=noisy_grids,
                        timesteps=timesteps,
                        rotation=rotations,
                        flip=flips,
                        color_shift=color_shifts,
                        heights=heights,
                        widths=widths,
                        auxiliary_size_loss_weight=0.1
                    )
            else:
                losses = model.compute_loss(
                    x0=output_grids,
                    input_grid=input_grids,
                    task_ids=task_indices,
                    xt=noisy_grids,
                    timesteps=timesteps,
                    rotation=rotations,
                    flip=flips,
                    color_shift=color_shifts,
                    heights=heights,
                    widths=widths,
                    auxiliary_size_loss_weight=0.1
                )

        with timer("4_backward_pass", timings):
            optimizer.zero_grad()
            losses['total_loss'].backward()

        with timer("5_optimizer_step", timings):
            optimizer.step()

        all_timings.append(timings)

    # Compute statistics
    print("=" * 60)
    print("⏱️ Timing Breakdown (avg ± std over {} iterations):".format(num_iterations))
    print("=" * 60)

    categories = sorted(all_timings[0].keys())
    total_time = 0

    for cat in categories:
        times = [t[cat] for t in all_timings]
        avg = sum(times) / len(times)
        std = (sum((t - avg) ** 2 for t in times) / len(times)) ** 0.5
        pct = (avg / sum(avg for avg in [sum(t[c] for c in categories) / num_iterations for t in all_timings]) * 100)
        total_time += avg

        name = cat.split('_', 1)[1].replace('_', ' ').title()
        print(f"{name:20s}: {avg:6.2f} ± {std:5.2f} ms  ({pct:5.1f}%)")

    print("-" * 60)
    print(f"{'Total per step':20s}: {total_time:6.2f} ms")
    print(f"{'Throughput':20s}: {1000/total_time:6.2f} steps/sec")
    print(f"{'Est. 96k steps':20s}: {(96000 * total_time / 1000 / 3600):6.1f} hours")
    print("=" * 60)

    # GPU memory info
    if device.type == 'cuda':
        print(f"\n💾 GPU Memory:")
        print(f"   Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"   Reserved:  {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        print(f"   Max:       {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")


def main():
    parser = argparse.ArgumentParser(description="Benchmark training step timing")
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument("--iterations", type=int, default=10, help="Number of iterations (default: 10)")

    args = parser.parse_args()
    benchmark_step(args.config, args.iterations)


if __name__ == "__main__":
    main()
