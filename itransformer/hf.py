#!/usr/bin/env python3
"""
Hugging Face Hub integration for ARC Diffusion models.

Push and pull model artifacts (checkpoints, configs, outputs) to/from HF Hub.

Usage:
    # Push outputs to HF
    uv run python experimental/diffusion/hf.py --push --config experimental/diffusion/configs/smol_config.json

    # Pull from HF
    uv run python experimental/diffusion/hf.py --pull --repo-name v4_diff_w_head_smol_baseline
"""
import argparse
import json
import os
import sys
from pathlib import Path
from huggingface_hub import HfApi, create_repo, hf_hub_download, snapshot_download
from huggingface_hub.utils import HfHubHTTPError

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def construct_repo_name(config: dict) -> str:
    """
    Construct HF repo name from config.
    Format: Trelis/{model_version}_{tag}

    Example: Trelis/v1_smol_baseline
    """
    model_version = config.get("model_version", "unknown")
    tag = config.get("tag", "default")

    # Construct full repo name (model_version should include size info)
    repo_name = f"{model_version}_{tag}"

    return repo_name


def push_to_hf(config_path: str):
    """
    Push model outputs to Hugging Face Hub.

    Uploads:
    - All files in the outputs folder
    - The config file used for training
    """
    # Load config
    config_path = Path(config_path)
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path, 'r') as f:
        config = json.load(f)

    # Get output directory from config
    output_dir = Path(config.get("output", {}).get("output_dir", "experimental/diffusion/outputs/default"))

    if not output_dir.exists():
        print(f"❌ Output directory not found: {output_dir}")
        sys.exit(1)

    # Construct repo name
    repo_name = construct_repo_name(config)
    full_repo_name = f"Trelis/{repo_name}"

    print(f"🚀 Pushing to Hugging Face Hub")
    print(f"📦 Repository: {full_repo_name}")
    print(f"📁 Output directory: {output_dir}")

    # Initialize HF API
    api = HfApi()

    # Create repo (or get existing)
    try:
        create_repo(
            repo_id=full_repo_name,
            private=True,
            repo_type="model",
            exist_ok=True
        )
        print(f"✓ Repository created/verified: {full_repo_name}")
    except Exception as e:
        print(f"❌ Failed to create repository: {e}")
        sys.exit(1)

    # Upload all files from output directory
    try:
        print(f"📤 Uploading files from {output_dir}...")
        api.upload_folder(
            folder_path=str(output_dir),
            repo_id=full_repo_name,
            repo_type="model",
        )
        print(f"✓ Uploaded all files from {output_dir}")
    except Exception as e:
        print(f"❌ Failed to upload output files: {e}")
        sys.exit(1)

    # Upload config file to root of repo
    try:
        print(f"📤 Uploading config file...")
        api.upload_file(
            path_or_fileobj=str(config_path),
            path_in_repo="config.json",
            repo_id=full_repo_name,
            repo_type="model",
        )
        print(f"✓ Uploaded config file as config.json")
    except Exception as e:
        print(f"❌ Failed to upload config file: {e}")
        sys.exit(1)

    print(f"✅ Successfully pushed to {full_repo_name}")
    print(f"🔗 View at: https://huggingface.co/{full_repo_name}")


def pull_from_hf(repo_name: str):
    """
    Pull model artifacts from Hugging Face Hub.

    Downloads all contents to experimental/diffusion/outputs/{tag}/
    """
    full_repo_name = f"Trelis/{repo_name}"

    print(f"📥 Pulling from Hugging Face Hub")
    print(f"📦 Repository: {full_repo_name}")

    # Download to outputs folder
    # Extract tag from repo name (last part after last underscore)
    tag = repo_name.split("_")[-1]
    local_dir = Path(f"experimental/diffusion/outputs/{tag}")

    print(f"📁 Downloading to: {local_dir}")

    try:
        snapshot_download(
            repo_id=full_repo_name,
            local_dir=str(local_dir),
            repo_type="model",
        )
        print(f"✅ Successfully pulled from {full_repo_name}")
        print(f"📁 Files saved to: {local_dir}")
    except HfHubHTTPError as e:
        if "404" in str(e):
            print(f"❌ Repository not found: {full_repo_name}")
            print(f"   Make sure the repository exists and you have access")
        else:
            print(f"❌ Failed to pull from HF Hub: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Failed to pull from HF Hub: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Push/pull ARC Diffusion models to/from Hugging Face Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Action flags
    parser.add_argument("--push", action="store_true", help="Push model outputs to HF Hub")
    parser.add_argument("--pull", action="store_true", help="Pull model outputs from HF Hub")

    # Push arguments
    parser.add_argument("--config", help="Path to config file (required for --push)")

    # Pull arguments
    parser.add_argument("--repo-name", help="Repository name to pull from (required for --pull)")

    args = parser.parse_args()

    # Validate arguments
    if not args.push and not args.pull:
        print("❌ Must specify either --push or --pull")
        parser.print_help()
        sys.exit(1)

    if args.push and args.pull:
        print("❌ Cannot specify both --push and --pull")
        sys.exit(1)

    if args.push:
        if not args.config:
            print("❌ --config is required for --push")
            sys.exit(1)
        push_to_hf(args.config)

    elif args.pull:
        if not args.repo_name:
            print("❌ --repo-name is required for --pull")
            sys.exit(1)
        pull_from_hf(args.repo_name)


if __name__ == "__main__":
    main()
