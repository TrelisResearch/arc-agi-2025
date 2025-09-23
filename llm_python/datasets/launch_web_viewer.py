#!/usr/bin/env python3

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
"""
Launch the web viewer with a parquet file.
Converts parquet to JSON and opens the web viewer in browser.
"""

import argparse
import json
import os
import tempfile
import webbrowser
from pathlib import Path
from typing import Dict, Any

from llm_python.datasets.io import read_soar_parquet
from llm_python.utils.task_loader import get_task_loader


def launch_web_viewer(parquet_path: str, include_task_data: bool = True, port: int = 8000) -> None:
    """
    Convert parquet to JSON and launch web viewer.

    Args:
        parquet_path: Path to the input parquet file
        include_task_data: Whether to include ARC task data
        port: Port for local HTTP server
    """
    print(f"Loading parquet file: {parquet_path}")
    df = read_soar_parquet(parquet_path)
    print(f"Loaded {len(df)} records")

    # Convert DataFrame to list of dictionaries
    data = df.to_dict('records')

    # Prepare the output structure
    output_data = {
        "metadata": {
            "total_records": len(data),
            "columns": list(df.columns),
            "created_from": str(parquet_path)
        },
        "data": data,
        "task_data": {}
    }

    # Load task data if requested
    if include_task_data:
        print("Loading ARC task data...")
        task_loader = get_task_loader()
        unique_task_ids = df['task_id'].unique()

        for task_id in unique_task_ids:
            try:
                task_data = task_loader.get_task(task_id)
                output_data["task_data"][task_id] = task_data
                print(f"  Loaded task: {task_id}")
            except Exception as e:
                print(f"  Warning: Failed to load task {task_id}: {e}")
                # Create minimal task data structure
                output_data["task_data"][task_id] = {
                    "train": [],
                    "test": []
                }

    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    viewer_path = script_dir / "web-viewer.html"

    if not viewer_path.exists():
        raise FileNotFoundError(f"Web viewer not found at {viewer_path}")

    # Create a temporary directory for serving files
    temp_dir = tempfile.mkdtemp()
    json_path = Path(temp_dir) / "dataset.json"
    viewer_copy_path = Path(temp_dir) / "index.html"

    # Write JSON data
    print(f"Writing temporary JSON file...")
    with open(json_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    # Copy viewer HTML with embedded data
    print("Creating web viewer with embedded data...")
    with open(viewer_path, 'r') as f:
        html_content = f.read()

    # Inject the data directly into the HTML
    data_script = f"""
    <script>
        // Embedded dataset
        window.EMBEDDED_DATASET = {json.dumps(output_data)};
    </script>
    """

    # Insert the data script before the main script
    html_content = html_content.replace('<script>', data_script + '\n    <script>')

    # Modify the DatasetViewer to load embedded data
    html_content = html_content.replace(
        'this.initializeEventListeners();',
        '''this.initializeEventListeners();
                this.loadEmbeddedData();'''
    )

    # Add loadEmbeddedData method
    load_embedded_method = '''
            loadEmbeddedData() {
                if (window.EMBEDDED_DATASET) {
                    console.log('Loading embedded dataset...');
                    this.data = window.EMBEDDED_DATASET.data;
                    if (window.EMBEDDED_DATASET.task_data) {
                        this.taskData = new Map(Object.entries(window.EMBEDDED_DATASET.task_data));
                    }
                    this.applyFilters();
                    this.hideLoading();
                    document.getElementById('content').classList.remove('hidden');
                }
            }
'''

    html_content = html_content.replace(
        'initializeEventListeners() {',
        load_embedded_method + '\n            initializeEventListeners() {'
    )

    with open(viewer_copy_path, 'w') as f:
        f.write(html_content)

    print(f"Starting local HTTP server on port {port}...")

    # Start a simple HTTP server
    import http.server
    import socketserver
    import threading

    os.chdir(temp_dir)

    class QuietHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, format, *args):
            pass  # Suppress HTTP request logs

    with socketserver.TCPServer(("", port), QuietHTTPRequestHandler) as httpd:
        # Start server in background thread
        server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        server_thread.start()

        url = f"http://localhost:{port}"
        print(f"Web viewer available at: {url}")
        print("Opening in browser...")

        # Open browser
        webbrowser.open(url)

        print("\nWeb viewer is running! Press Ctrl+C to stop.")
        print(f"Dataset: {len(data)} records")
        if include_task_data:
            print(f"Task data: {len(output_data['task_data'])} tasks")

        try:
            # Keep the server running
            while True:
                import time
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down web viewer...")
            httpd.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch web viewer for SOAR parquet files")
    parser.add_argument("parquet_file", help="Path to parquet file")
    parser.add_argument(
        "--no-task-data",
        action="store_true",
        help="Don't include ARC task data (faster loading, but no input/expected grids)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for local HTTP server (default: 8000)"
    )

    args = parser.parse_args()

    try:
        launch_web_viewer(
            args.parquet_file,
            include_task_data=not args.no_task_data,
            port=args.port
        )
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Error: {e}")
        exit(1)