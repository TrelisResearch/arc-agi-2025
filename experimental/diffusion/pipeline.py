#!/usr/bin/env python3
"""
Full ARC Diffusion Pipeline Runner

Runs the complete training pipeline in sequence:
1. Diffusion backbone training
2. Size head training
3. Model evaluation

Usage:
    uv run experimental/diffusion/pipeline.py --config experimental/diffusion/configs/my_config.json
    uv run experimental/diffusion/pipeline.py --config experimental/diffusion/configs/my_config.json --skip-training
    uv run experimental/diffusion/pipeline.py --config experimental/diffusion/configs/my_config.json --skip-size-head
"""

import argparse
import json
import sys
import subprocess
from pathlib import Path
import time
import datetime


def run_command(command: list, description: str, cwd: str = None) -> bool:
    """Run a command and return True if successful."""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"Command: {' '.join(command)}")
    print(f"{'='*60}")

    start_time = time.time()

    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            check=True,
            capture_output=False,  # Show output in real-time
            text=True
        )

        elapsed = time.time() - start_time
        print(f"\n✅ {description} completed successfully in {elapsed:.1f}s")
        return True

    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n❌ {description} failed after {elapsed:.1f}s")
        print(f"Exit code: {e.returncode}")
        return False
    except KeyboardInterrupt:
        print(f"\n⏹️ {description} interrupted by user")
        return False
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n💥 {description} failed with error after {elapsed:.1f}s: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run full ARC diffusion training pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  uv run experimental/diffusion/pipeline.py --config configs/my_config.json

  # Skip training, only run size head and evaluation
  uv run experimental/diffusion/pipeline.py --config configs/my_config.json --skip-training

  # Skip size head training
  uv run experimental/diffusion/pipeline.py --config configs/my_config.json --skip-size-head

  # Only run evaluation
  uv run experimental/diffusion/pipeline.py --config configs/my_config.json --skip-training --skip-size-head
        """
    )

    parser.add_argument("--config", required=True, help="Path to config JSON file")
    parser.add_argument("--skip-training", action="store_true", help="Skip diffusion backbone training")
    parser.add_argument("--skip-size-head", action="store_true", help="Skip size head training")
    parser.add_argument("--skip-evaluation", action="store_true", help="Skip evaluation")
    parser.add_argument("--eval-limit", type=int, default=5, help="Limit evaluation to N tasks (default: 5)")

    args = parser.parse_args()

    # Validate config file
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        sys.exit(1)

    # Load and validate config
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except Exception as e:
        print(f"❌ Failed to load config file: {e}")
        sys.exit(1)

    # Extract key paths
    output_dir = Path(config.get('output', {}).get('output_dir', 'experimental/diffusion/outputs/default'))

    print(f"🎯 ARC Diffusion Full Pipeline")
    print(f"📅 Started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 Config: {config_path}")
    print(f"📂 Output dir: {output_dir}")
    print(f"⚡ Evaluation limit: {args.eval_limit}")

    # Track which steps to run
    steps_to_run = []
    if not args.skip_training:
        steps_to_run.append("training")
    if not args.skip_size_head:
        steps_to_run.append("size-head")
    if not args.skip_evaluation:
        steps_to_run.append("evaluation")

    print(f"📋 Pipeline steps: {' → '.join(steps_to_run)}")

    # Get project root (assuming we're running from repo root)
    project_root = Path.cwd()

    # Step 1: Diffusion backbone training
    if not args.skip_training:
        if not run_command(
            ["uv", "run", "python", "experimental/diffusion/train_diffusion_backbone.py", "--config", str(config_path)],
            "Diffusion backbone training",
            cwd=str(project_root)
        ):
            print("❌ Training failed, stopping pipeline")
            sys.exit(1)
    else:
        print("\n⏭️ Skipping diffusion backbone training")

    # Step 2: Size head training
    if not args.skip_size_head:
        # Check if backbone model exists
        best_model_path = output_dir / "best_model.pt"
        final_model_path = output_dir / "final_model.pt"

        if best_model_path.exists():
            backbone_path = best_model_path
            print(f"✓ Using best model for size head: {backbone_path}")
        elif final_model_path.exists():
            backbone_path = final_model_path
            print(f"⚠️ Using final model for size head (no best model found): {backbone_path}")
        else:
            print(f"❌ No backbone model found: tried {best_model_path} and {final_model_path}")
            print("   Cannot train size head without backbone model")
            if not args.skip_evaluation:
                print("   Will skip size head but continue to evaluation")
            backbone_path = None

        if backbone_path:
            print("📐 Size head is now integrated into main model training")
            print("   The backbone model already includes size prediction capabilities!")
    else:
        print("\n⏭️ Skipping size head training")

    # Step 3: Evaluation
    if not args.skip_evaluation:
        # Check if backbone model exists
        best_model_path = output_dir / "best_model.pt"
        final_model_path = output_dir / "final_model.pt"

        if best_model_path.exists():
            print(f"✓ Using best model for evaluation: {best_model_path}")
        elif final_model_path.exists():
            print(f"⚠️ Using final model for evaluation (no best model found): {final_model_path}")
        else:
            print(f"❌ No backbone model found: tried {best_model_path} and {final_model_path}")
            print("   Cannot run evaluation without trained model")
            sys.exit(1)

        # Run evaluation
        eval_command = [
            "uv", "run", "python", "experimental/diffusion/evaluate.py",
            "--config", str(config_path),
            "--limit", str(args.eval_limit)
        ]

        if not run_command(
            eval_command,
            f"Model evaluation (limit: {args.eval_limit} tasks)",
            cwd=str(project_root)
        ):
            print("❌ Evaluation failed")
            sys.exit(1)
    else:
        print("\n⏭️ Skipping evaluation")

    # Pipeline completed
    total_time = time.time()
    print(f"\n{'='*60}")
    print(f"🎉 Pipeline completed successfully!")
    print(f"📂 Results saved in: {output_dir}")
    print(f"📅 Finished at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Show key output files
    print(f"\n📋 Key output files:")

    backbone_path = output_dir / "best_model.pt"
    if backbone_path.exists():
        print(f"  🧠 Backbone model: {backbone_path}")

    size_head_path = output_dir / "best_size_head.pt"
    if size_head_path.exists():
        print(f"  📐 Size head: {size_head_path}")

    # Find evaluation results
    eval_files = list(output_dir.glob("evaluation_*.json"))
    if eval_files:
        latest_eval = max(eval_files, key=lambda x: x.stat().st_mtime)
        print(f"  📊 Evaluation: {latest_eval}")

    visualization_path = output_dir / "training_noise_visualization.png"
    if visualization_path.exists():
        print(f"  🎨 Visualization: {visualization_path}")

    print(f"{'='*60}")


if __name__ == "__main__":
    main()