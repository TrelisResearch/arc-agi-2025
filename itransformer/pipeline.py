#!/usr/bin/env python3
"""
Full ARC Iterative Refiner Pipeline Runner

Runs the complete training pipeline in sequence:
1. Iterative refinement model training (with integrated size head)
2. Model evaluation

Usage:
    uv run itransformer/pipeline.py --config itransformer/configs/test_config.json
    uv run itransformer/pipeline.py --config itransformer/configs/test_config.json --skip-training
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
        description="Run full ARC iterative refiner training pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline (training + evaluation)
  uv run itransformer/pipeline.py --config itransformer/configs/test_config.json

  # Skip training, only run evaluation
  uv run itransformer/pipeline.py --config itransformer/configs/test_config.json --skip-training

  # Only run training
  uv run itransformer/pipeline.py --config itransformer/configs/test_config.json --skip-evaluation
        """
    )

    parser.add_argument("--config", required=True, help="Path to config JSON file")
    parser.add_argument("--skip-training", action="store_true", help="Skip iterative model training")
    parser.add_argument("--skip-evaluation", action="store_true", help="Skip evaluation")
    parser.add_argument("--eval-limit", type=int, default=0, help="Limit evaluation to N tasks (default: 0 for all)")
    parser.add_argument("--prefer-best", action="store_true", help="Use best_model.pt for evaluation instead of final_model.pt")
    parser.add_argument("--inference-steps", type=int, help="Number of refinement steps at inference (default: K from config)")

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
    output_dir = Path(config.get('output', {}).get('output_dir', 'itransformer/outputs/default'))

    print(f"🎯 ARC Iterative Refiner Full Pipeline")
    print(f"📅 Started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 Config: {config_path}")
    print(f"📂 Output dir: {output_dir}")
    print(f"⚡ Refinement steps (K): {config.get('training', {}).get('K', 8)}")
    print(f"⚡ Evaluation limit: {args.eval_limit}")

    # Track which steps to run
    steps_to_run = []
    if not args.skip_training:
        steps_to_run.append("training (with integrated size head)")
    if not args.skip_evaluation:
        steps_to_run.append("evaluation")

    print(f"📋 Pipeline steps: {' → '.join(steps_to_run)}")

    # Get project root (assuming we're running from repo root)
    project_root = Path.cwd()

    # Step 1: Iterative model training (with integrated size head)
    if not args.skip_training:
        if not run_command(
            ["uv", "run", "python", "itransformer/train.py", "--config", str(config_path), "--no-wandb"],
            "Iterative refiner training (with integrated size head)",
            cwd=str(project_root)
        ):
            print("❌ Training failed, stopping pipeline")
            sys.exit(1)
    else:
        print("\n⏭️ Skipping iterative model training")

    # Step 2: Evaluation
    if not args.skip_evaluation:
        # Check if model exists
        best_model_path = output_dir / "best_model.pt"
        final_model_path = output_dir / "final_model.pt"

        # Determine which model will be used based on --prefer-best flag
        if args.prefer_best:
            if best_model_path.exists():
                print(f"✓ Using best model for evaluation: {best_model_path}")
            elif final_model_path.exists():
                print(f"⚠️ Using final model for evaluation (best model not found): {final_model_path}")
            else:
                print(f"❌ No model found: tried {best_model_path} and {final_model_path}")
                print("   Cannot run evaluation without trained model")
                sys.exit(1)
        else:
            if final_model_path.exists():
                print(f"✓ Using final model for evaluation: {final_model_path}")
            elif best_model_path.exists():
                print(f"⚠️ Using best model for evaluation (final model not found): {best_model_path}")
            else:
                print(f"❌ No model found: tried {final_model_path} and {best_model_path}")
                print("   Cannot run evaluation without trained model")
                sys.exit(1)

        # Run evaluation
        eval_command = [
            "uv", "run", "python", "itransformer/evaluate.py",
            "--config", str(config_path),
            "--limit", str(args.eval_limit)
        ]

        # Add --prefer-best flag if specified
        if args.prefer_best:
            eval_command.append("--prefer-best")

        # Add --inference-steps flag if specified
        if args.inference_steps is not None:
            eval_command.extend(["--inference-steps", str(args.inference_steps)])

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
    print(f"\n{'='*60}")
    print(f"🎉 Pipeline completed successfully!")
    print(f"📂 Results saved in: {output_dir}")
    print(f"📅 Finished at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Show key output files
    print(f"\n📋 Key output files:")

    model_path = output_dir / "best_model.pt"
    if model_path.exists():
        print(f"  🧠 Model (with integrated size head): {model_path}")

    # Find evaluation results
    eval_files = list(output_dir.glob("evaluation_*.json"))
    if eval_files:
        latest_eval = max(eval_files, key=lambda x: x.stat().st_mtime)
        print(f"  📊 Evaluation: {latest_eval}")

    print(f"{'='*60}")


if __name__ == "__main__":
    main()