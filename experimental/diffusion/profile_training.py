#!/usr/bin/env python3
"""
Profile training to identify bottlenecks.

Usage:
    uv run python experimental/diffusion/profile_training.py --config experimental/diffusion/configs/test_config.json
"""
import argparse
import json
import sys
import torch
from pathlib import Path
import time

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from experimental.diffusion.src.training import train_arc_diffusion
from experimental.diffusion.train_diffusion_backbone import load_and_flatten_config


def profile_training(config_path: str, num_steps: int = 100):
    """Profile training loop for bottlenecks."""

    # Load config
    config = load_and_flatten_config(config_path)

    # Override settings for profiling
    config['optimizer_steps'] = num_steps
    config['use_wandb'] = False
    config['val_every_steps'] = 999999  # Disable validation during profiling
    config['vis_every_steps'] = 999999  # Disable visualization
    config['log_every'] = 10

    print(f"🔍 Profiling Training Loop")
    print(f"📊 Steps: {num_steps}")
    print(f"🎯 Config: {config_path}")
    print(f"=" * 60)

    # Check device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"✓ MPS available")
    else:
        device = torch.device('cpu')
        print(f"⚠️ Using CPU (profiling will be slow)")

    # Profile with torch profiler
    if device.type == 'cuda':
        print("\n📈 Running CUDA profiler...")
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            train_arc_diffusion(config=config)

        # Print results
        print("\n" + "=" * 60)
        print("🔥 Top 10 CUDA operations by time:")
        print("=" * 60)
        print(prof.key_averages().table(
            sort_by="cuda_time_total",
            row_limit=10
        ))

        print("\n" + "=" * 60)
        print("💾 Top 10 operations by memory:")
        print("=" * 60)
        print(prof.key_averages().table(
            sort_by="cuda_memory_usage",
            row_limit=10
        ))

        # Save trace for chrome://tracing
        trace_path = Path("experimental/diffusion/outputs/trace.json")
        prof.export_chrome_trace(str(trace_path))
        print(f"\n💾 Saved trace to: {trace_path}")
        print(f"   Open in Chrome at: chrome://tracing")

    else:
        # Simple timing profiling for MPS/CPU
        print("\n⏱️ Running timing profiler (MPS/CPU)...")

        # Measure overall throughput
        start_time = time.time()
        train_arc_diffusion(config=config)
        total_time = time.time() - start_time

        throughput = num_steps / total_time
        time_per_step = total_time / num_steps

        print("\n" + "=" * 60)
        print("⚡ Performance Metrics:")
        print("=" * 60)
        print(f"Total time: {total_time:.2f}s")
        print(f"Steps/sec: {throughput:.2f}")
        print(f"Time/step: {time_per_step*1000:.1f}ms")
        print(f"Estimated time for 96k steps: {(96000 * time_per_step / 3600):.1f} hours")

        if device.type == 'mps':
            print("\n💡 MPS Profiling Tips:")
            print("   - Monitor GPU usage with: sudo powermetrics --samplers gpu_power -i1000")
            print("   - Check memory: Activity Monitor > Window > GPU History")


def main():
    parser = argparse.ArgumentParser(description="Profile ARC diffusion training")
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument("--steps", type=int, default=100, help="Number of steps to profile (default: 100)")

    args = parser.parse_args()

    profile_training(args.config, args.steps)


if __name__ == "__main__":
    main()
