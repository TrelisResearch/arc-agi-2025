#!/usr/bin/env python3
"""
Move downloaded ARC dataset files from Downloads to data/manual/

Usage:
    uv run experimental/move_downloads.py

This script:
1. Looks for arc-agi_*_challenges.json and arc-agi_*_solutions.json in Downloads
2. Moves them to data/manual/
3. Merges with existing files if they exist
"""

import json
import os
import shutil
from pathlib import Path


def find_downloads_folder():
    """Find the user's Downloads folder."""
    home = Path.home()
    downloads = home / "Downloads"
    if downloads.exists():
        return downloads
    return None


def merge_json_files(existing_file, new_file):
    """Merge two JSON files, combining their contents."""
    existing_data = {}
    if existing_file.exists():
        with open(existing_file, 'r') as f:
            existing_data = json.load(f)

    with open(new_file, 'r') as f:
        new_data = json.load(f)

    # Merge the data
    existing_data.update(new_data)

    # Write back to the target location
    with open(existing_file, 'w') as f:
        json.dump(existing_data, f, indent=2)

    return len(new_data)


def main():
    print("🔄 Moving ARC dataset files from Downloads to data/manual/")

    downloads_folder = find_downloads_folder()
    if not downloads_folder:
        print("❌ Could not find Downloads folder")
        return

    repo_root = Path(__file__).parent.parent
    manual_dir = repo_root / "data" / "manual"
    manual_dir.mkdir(exist_ok=True)

    # Main dataset files
    dataset_files = [
        "arc-agi_training_challenges.json",
        "arc-agi_training_solutions.json",
        "arc-agi_evaluation_challenges.json",
        "arc-agi_evaluation_solutions.json"
    ]

    moved_count = 0

    # Handle main dataset files
    for pattern in dataset_files:
        download_file = downloads_folder / pattern
        target_file = manual_dir / pattern

        if download_file.exists():
            print(f"📁 Found: {pattern}")

            if target_file.exists():
                # Merge with existing file
                task_count = merge_json_files(target_file, download_file)
                print(f"✅ Merged {task_count} tasks into existing {pattern}")
            else:
                # Simple copy
                shutil.copy2(download_file, target_file)
                with open(download_file, 'r') as f:
                    data = json.load(f)
                task_count = len(data)
                print(f"✅ Moved {task_count} tasks to {pattern}")

            # Remove the downloaded file
            download_file.unlink()
            moved_count += 1
        else:
            print(f"⏭️  No {pattern} found in Downloads")

    if moved_count > 0:
        print(f"\n🎉 Successfully moved {moved_count} files!")
        print("📁 Dataset files: data/manual/")
        print("Your manual ARC dataset is ready!")
    else:
        print("\n📭 No ARC dataset files found in Downloads")
        print("Download files from the ARC Task Creator first.")


if __name__ == "__main__":
    main()