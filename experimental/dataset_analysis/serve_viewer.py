#!/usr/bin/env python3

import http.server
import socketserver
import webbrowser
import os
from pathlib import Path

def serve_viewer(port=8000):
    """Serve the HTML viewer on a local web server."""

    # Change to the directory containing the viewer files
    viewer_dir = Path(__file__).parent
    os.chdir(viewer_dir)

    # Check if required files exist
    required_files = ['viewer.html', 'viewer_data.json']
    missing_files = [f for f in required_files if not Path(f).exists()]

    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        print("Make sure to run create_viewer_data.py first!")
        return

    Handler = http.server.SimpleHTTPRequestHandler

    print(f"🌐 Starting web server on port {port}...")
    print(f"📁 Serving files from: {viewer_dir}")

    try:
        with socketserver.TCPServer(("", port), Handler) as httpd:
            url = f"http://localhost:{port}/viewer.html"
            print(f"🚀 Web viewer available at: {url}")
            print("📊 The viewer shows:")
            print("   • Grid visualizations with ARC color palette")
            print("   • Pixel match scores and gzip ratios")
            print("   • Input, expected output, and predicted output grids")
            print("   • Row-by-row navigation with pagination")
            print("   • Generated code for each program")
            print("\n⌨️  Use arrow keys or buttons to navigate between rows")
            print("🛑 Press Ctrl+C to stop the server")

            # Try to open in browser
            try:
                webbrowser.open(url)
                print("🌍 Opening in your default web browser...")
            except Exception:
                print("💡 Manually open the URL above in your browser")

            httpd.serve_forever()

    except KeyboardInterrupt:
        print("\n🛑 Server stopped")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        print(f"💡 Try a different port: python serve_viewer.py --port 8001")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Serve the ARC Grid Viewer')
    parser.add_argument('--port', type=int, default=8000,
                       help='Port to serve on (default: 8000)')

    args = parser.parse_args()
    serve_viewer(args.port)