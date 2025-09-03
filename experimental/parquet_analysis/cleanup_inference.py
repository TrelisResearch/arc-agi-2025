#!/usr/bin/env python3

import os
import shutil
from pathlib import Path

def main():
    inference_dir = Path("/Users/ronanmcgovern/TR/arc-agi-2025/llm_python/datasets/inference")
    
    print(f"🧹 Cleaning up inference directory: {inference_dir}")
    
    # Find all directories (potential leftover folders)
    directories = [p for p in inference_dir.iterdir() if p.is_dir()]
    print(f"Found {len(directories)} directories to examine")
    
    # Categorize directories
    empty_dirs = []
    dirs_with_only_empty_subdirs = []
    dirs_with_parquet_files = []
    
    for dir_path in directories:
        # Check if directory is completely empty
        dir_contents = list(dir_path.iterdir())
        if not dir_contents:
            empty_dirs.append(dir_path)
            continue
        
        # Check if directory only contains empty subdirectories or small files
        has_meaningful_content = False
        for item in dir_contents:
            if item.is_file():
                # Consider files > 100KB as meaningful
                if item.stat().st_size > 100_000:  # 100KB
                    has_meaningful_content = True
                    dirs_with_parquet_files.append(dir_path)
                    break
            elif item.is_dir():
                # Check if subdirectory has meaningful content
                subdir_contents = list(item.iterdir())
                if subdir_contents:
                    for subitem in subdir_contents:
                        if subitem.is_file() and subitem.stat().st_size > 100_000:
                            has_meaningful_content = True
                            dirs_with_parquet_files.append(dir_path)
                            break
                if has_meaningful_content:
                    break
        
        if not has_meaningful_content:
            dirs_with_only_empty_subdirs.append(dir_path)
    
    print(f"\n📊 Analysis:")
    print(f"  - Empty directories: {len(empty_dirs)}")
    print(f"  - Directories with only empty subdirs/small files: {len(dirs_with_only_empty_subdirs)}")
    print(f"  - Directories with meaningful content: {len(dirs_with_parquet_files)}")
    
    # Show what we found
    if dirs_with_parquet_files:
        print(f"\n📁 Directories with meaningful content (keeping):")
        for d in dirs_with_parquet_files:
            files = [f for f in d.rglob("*.parquet")]
            print(f"  - {d.name}: {len(files)} parquet files")
    
    # Clean up empty directories
    if empty_dirs:
        print(f"\n🗑️ Removing {len(empty_dirs)} empty directories...")
        for d in empty_dirs:
            print(f"  - Removing: {d.name}")
            try:
                d.rmdir()
            except Exception as e:
                print(f"    ❌ Failed: {e}")
    
    # Clean up directories with only empty subdirs
    if dirs_with_only_empty_subdirs:
        print(f"\n🗑️ Removing {len(dirs_with_only_empty_subdirs)} directories with only empty/trivial content...")
        for d in dirs_with_only_empty_subdirs:
            print(f"  - Removing: {d.name}")
            try:
                shutil.rmtree(d)
            except Exception as e:
                print(f"    ❌ Failed: {e}")
    
    # Show final count
    remaining_items = list(inference_dir.iterdir())
    parquet_files = [f for f in remaining_items if f.is_file() and f.suffix == '.parquet']
    remaining_dirs = [d for d in remaining_items if d.is_dir()]
    
    print(f"\n✅ Cleanup complete:")
    print(f"  - Parquet files: {len(parquet_files)}")
    print(f"  - Remaining directories: {len(remaining_dirs)}")
    
    if remaining_dirs:
        print(f"  - Remaining dir names: {[d.name for d in remaining_dirs]}")

if __name__ == "__main__":
    main()