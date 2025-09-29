#!/usr/bin/env python3
"""
Download a Hugging Face repo and zip it.

Example:
  python download_and_zip_hf_repo.py \
    --repo-id Trelis/arc-1-fake-ttt-blended-c802-FP8-Dynamic \
    --out-dir ./arc-1-fake-ttt-blended-c802-FP8-Dynamic \
    --zip-path ./arc-1-fake-ttt-blended-c802-FP8-Dynamic.zip
"""

import argparse
import os
import shutil
from pathlib import Path
from huggingface_hub import snapshot_download, login
from huggingface_hub.utils import HfHubHTTPError

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

def main():
    parser = argparse.ArgumentParser(description="Download a HF repo and zip it.")
    parser.add_argument(
        "--repo-id",
        default="Trelis/arc-1-fake-ttt-blended-c802-FP8-Dynamic",
        help="Hub repo_id (namespace/name)"
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional revision/branch/tag/commit to pin (e.g., 'main' or a commit hash)."
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Where to place the downloaded files (defaults to fine-tuning/downloads/<repo-name>)."
    )
    parser.add_argument(
        "--zip-path",
        default=None,
        help="Output .zip path (defaults to <out-dir>.zip)."
    )
    parser.add_argument(
        "--patterns",
        nargs="*",
        default=None,
        help="Optional allow_patterns (glob patterns) to restrict which files are downloaded."
    )
    parser.add_argument(
        "--token-env",
        default="HF_TOKEN",
        help="Env var name for the HF token (defaults to HF_TOKEN)."
    )
    args = parser.parse_args()

    # Optional: authenticate via env var if provided
    token = os.environ.get(args.token_env)
    if token:
        try:
            login(token=token, add_to_git_credential=True)
        except Exception:
            # If cached login already exists, this may fail harmlessly.
            pass

    repo_name = args.repo_id.split("/")[-1]
    # Default to downloads subfolder in the fine-tuning directory
    script_dir = Path(__file__).parent  # Get the fine-tuning directory
    out_dir = Path(args.out_dir or script_dir / "downloads" / repo_name).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"📥 Downloading '{args.repo_id}' to: {out_dir}")

    try:
        # Download directly into out_dir with real files (no symlinks).
        # If you want to restrict content, set --patterns like:
        # --patterns "*.safetensors" "tokenizer.*" "config.json" "generation_config.json"
        local_dir = snapshot_download(
            repo_id=args.repo_id,
            revision=args.revision,
            local_dir=str(out_dir),
            local_dir_use_symlinks=False,
            allow_patterns=args.patterns,  # None means "everything"
            # You can also add ignore_patterns=... if needed.
        )
    except HfHubHTTPError as e:
        raise SystemExit(f"❌ Hub error: {e}")
    except Exception as e:
        raise SystemExit(f"❌ Unexpected error: {e}")

    # Decide zip path
    zip_path = Path(args.zip_path or f"{out_dir}.zip").resolve()

    # Create the zip
    print(f"🗜️  Zipping to: {zip_path}")
    # shutil.make_archive requires a base name without extension
    base_name = str(zip_path.with_suffix(""))
    root_dir = str(out_dir)
    shutil.make_archive(base_name=base_name, format="zip", root_dir=root_dir)

    print("✅ Done.")
    print(f"   Files directory: {out_dir}")
    print(f"   Zip archive:     {zip_path}")

if __name__ == "__main__":
    main()
