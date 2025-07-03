#!/usr/bin/env python3

import os
import shutil
from pathlib import Path

def cleanup_logs():
    """Clean up old log files"""
    logs_dir = Path("logs")
    
    if not logs_dir.exists():
        print("No logs directory found.")
        return
    
    log_files = list(logs_dir.glob("*.json"))
    print(f"Found {len(log_files)} log files")
    
    if len(log_files) == 0:
        print("No log files to clean up.")
        return
    
    response = input(f"Delete all {len(log_files)} log files? (y/N): ")
    if response.lower() == 'y':
        for log_file in log_files:
            log_file.unlink()
        print(f"Deleted {len(log_files)} log files.")
    else:
        print("Cleanup cancelled.")

if __name__ == "__main__":
    cleanup_logs()