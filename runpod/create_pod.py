#!/usr/bin/env python3
import requests
import os
import argparse
import signal
import sys
import time
import json
import yaml
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Sensible default data centers
DEFAULT_DATA_CENTERS = [
    "US-IL-1", "US-TX-1", "US-TX-3", "US-KS-2", 
    "US-GA-1", "US-CA-2", "EU-RO-1", "EU-SE-1"
]

# Global variable to store pod ID for cleanup
pod_id = None
cleanup_in_progress = False

def signal_handler(signum, frame):
    """Handle interrupt signals for graceful shutdown"""
    global cleanup_in_progress
    
    if cleanup_in_progress:
        print("\n⚠️  WARNING: Pod deletion already in progress. Please wait...")
        print("⚠️  Do NOT press Ctrl+C again while the pod is being terminated!")
        return
    
    print(f"\n🛑 Received signal {signum}. Initiating graceful shutdown...")
    cleanup_pod()
    sys.exit(0)

def cleanup_pod():
    """Delete the pod and perform cleanup"""
    global pod_id, cleanup_in_progress
    
    if not pod_id:
        print("No pod to cleanup.")
        return
    
    if cleanup_in_progress:
        return
    
    cleanup_in_progress = True
    print(f"\n🗑️  Deleting pod {pod_id}...")
    
    try:
        headers = {
            "Authorization": f"Bearer {os.environ['RUNPOD_API_KEY']}",
            "Content-Type": "application/json"
        }
        
        delete_url = f"https://rest.runpod.io/v1/pods/{pod_id}"
        response = requests.delete(delete_url, headers=headers)
        
        if response.status_code in [200, 204]:
            print(f"✅ Pod {pod_id} successfully deleted.")
        else:
            print(f"❌ Failed to delete pod {pod_id}. Status: {response.status_code}")
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"❌ Error deleting pod: {e}")

def get_pod_status(pod_id):
    """Get the current status of a pod"""
    try:
        headers = {
            "Authorization": f"Bearer {os.environ['RUNPOD_API_KEY']}",
            "Content-Type": "application/json"
        }
        
        status_url = f"https://rest.runpod.io/v1/pods/{pod_id}"
        response = requests.get(status_url, headers=headers)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ Failed to get pod status. Status: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Error getting pod status: {e}")
        return None

def check_http_endpoint_health(pod_id, port, health_path="/health"):
    """Check if the endpoint is responding"""
    try:
        health_url = f"https://{pod_id}-{port}.proxy.runpod.net{health_path}"
        response = requests.get(health_url, timeout=5)
        return response.status_code < 400
    except:
        return False

def load_template(template_name):
    """Load pod configuration from YAML or JSON file in templates/ folder"""
    try:
        # Get the directory of the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        templates_dir = os.path.join(script_dir, 'templates')
        
        # If template_name doesn't have an extension, try common ones
        if not os.path.splitext(template_name)[1]:
            # Try different extensions
            for ext in ['.yaml', '.yml', '.json']:
                template_path = os.path.join(templates_dir, template_name + ext)
                if os.path.exists(template_path):
                    break
            else:
                print(f"❌ Template '{template_name}' not found in {templates_dir}")
                print(f"Available templates: {', '.join(os.listdir(templates_dir)) if os.path.exists(templates_dir) else 'None'}")
                return None
        else:
            template_path = os.path.join(templates_dir, template_name)
            if not os.path.exists(template_path):
                print(f"❌ Template file '{template_path}' not found")
                return None
        
        with open(template_path, 'r') as f:
            if template_path.endswith('.yaml') or template_path.endswith('.yml'):
                config = yaml.safe_load(f)
            else:
                config = json.load(f)
        
        print(f"📁 Loaded template: {template_path}")
        
        # Set default data centers if not specified
        if 'dataCenterIds' not in config:
            config['dataCenterIds'] = DEFAULT_DATA_CENTERS
            
        return config
    except Exception as e:
        print(f"❌ Error loading template {template_name}: {e}")
        return None

def create_pod(config, extra_args):
    """Create a RunPod pod with specified configuration"""
    global pod_id
    
    payload = config.copy()
    
    # Add extra arguments to docker command if provided
    if extra_args and 'dockerStartCmd' in payload:
        if isinstance(payload['dockerStartCmd'], list):
            payload['dockerStartCmd'].extend(extra_args)
        else:
            # Convert string to list and add args
            cmd_list = payload['dockerStartCmd'].split()
            cmd_list.extend(extra_args)
            payload['dockerStartCmd'] = cmd_list
    
    headers = {
        "Authorization": f"Bearer {os.environ['RUNPOD_API_KEY']}",
        "Content-Type": "application/json"
    }
    
    pod_name = payload.get('name', f"pod-{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    print(f"🚀 Creating pod '{pod_name}'")
    
    # Print docker command if available
    if 'dockerStartCmd' in payload:
        cmd = payload['dockerStartCmd']
        if isinstance(cmd, list):
            print(f"📝 Docker command: {' '.join(cmd)}")
        else:
            print(f"📝 Docker command: {cmd}")
    
    try:
        url = "https://rest.runpod.io/v1/pods"
        response = requests.post(url, json=payload, headers=headers)
        
        if response.status_code == 201:
            pod_info = response.json()
            pod_id = pod_info['id']
            print(f"✅ Pod created successfully! ID: {pod_id}")
            print(f"💰 Cost per hour: ${pod_info.get('costPerHr', 'Unknown')}")
            print(f"🖥️  Console: https://console.runpod.io/pods/{pod_id}")
            return pod_id
        else:
            print(f"❌ Failed to create pod. Status: {response.status_code}")
            print(f"Response: {response.text}")
            return None
    except Exception as e:
        print(f"❌ Error creating pod: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(
        description='Create and manage a RunPod pod using configuration templates',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s sglang -- --model-path Qwen/Qwen3-4B
  %(prog)s sglang -- --model-path Qwen/Qwen3-4B --reasoning-parser qwen3

Template names are loaded from the templates/ folder. File extensions are optional.
All arguments after '--' are passed directly to the docker command.
        """
    )
    
    parser.add_argument('template', help='Template name (from templates/ folder, extension optional)')
    parser.add_argument('--no-health-check', action='store_true',
                       help='Skip health check after pod creation')
    
    # Parse arguments, splitting at '--'
    if '--' in sys.argv:
        script_args = sys.argv[1:sys.argv.index('--')]
        extra_args = sys.argv[sys.argv.index('--') + 1:]
    else:
        script_args = sys.argv[1:]
        extra_args = []
    
    # Parse script-specific arguments
    args = parser.parse_args(script_args)
    
    # Check for required environment variable
    if 'RUNPOD_API_KEY' not in os.environ:
        print("❌ Error: RUNPOD_API_KEY environment variable not set!")
        sys.exit(1)
    
    # Load template
    config = load_template(args.template)
    if not config:
        sys.exit(1)
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create the pod
    pod_id = create_pod(config, extra_args)
    if not pod_id:
        sys.exit(1)
    
    # Wait for pod to be ready
    print(f"\n⏳ Pod has started. Waiting for it to be ready...")
    pod_info = get_pod_status(pod_id)
    if not pod_info:
        cleanup_pod()
        sys.exit(1)
        
    ports = pod_info.get('ports', [])
    
    for port in ports:
        port_num = port.split('/')[0]
        print(f"🌐 Exposed port: https://{pod_id}-{port_num}.proxy.runpod.net/")
    
    # Health check
    if not args.no_health_check and ports:
        port_num = ports[0].split('/')[0]
        print(f"\n⏳ Waiting for health checks", end="", flush=True)
        
        start_time = time.time()
        while not check_http_endpoint_health(pod_id, port_num):
            if time.time() - start_time > 300:  # 5 minute timeout
                print(f"\n⚠️  Health check timed out after 300s")
                break
            print(".", end="", flush=True)
            time.sleep(10)
        else:
            print(" ✅ Health check passed!")
    
    print(f"\n💡 The pod is now running. Press Ctrl+C to stop and delete the pod.")
    print(f"⚠️  After pressing Ctrl+C, DO NOT press it again while the pod is being deleted!")
    
    # Keep the script running until interrupted
    try:
        while True:
            time.sleep(60)
            # Check pod status periodically
            pod_info = get_pod_status(pod_id)
            if pod_info and pod_info.get('desiredStatus') not in ['RUNNING']:
                print(f"⚠️  Pod status changed to: {pod_info.get('desiredStatus')}")
                break
    except KeyboardInterrupt:
        pass  # Will be handled by signal_handler
    
    # Cleanup on exit
    cleanup_pod()

if __name__ == "__main__":
    main()
