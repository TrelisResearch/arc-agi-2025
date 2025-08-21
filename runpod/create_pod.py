#!/usr/bin/env python3
import requests
import os
import argparse
import signal
import sys
import time
import json
import yaml
import socket
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Import shared configuration
from config import DEFAULT_DATA_CENTERS, get_regions

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

def get_machine_info(machine_id):
    """Get machine information which might contain the public IP"""
    try:
        headers = {
            "Authorization": f"Bearer {os.environ['RUNPOD_API_KEY']}",
            "Content-Type": "application/json"
        }
        
        machine_url = f"https://rest.runpod.io/v1/machines/{machine_id}"
        response = requests.get(machine_url, headers=headers)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ Failed to get machine info. Status: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Error getting machine info: {e}")
        return None

def check_tcp_endpoint_health(host, port, timeout=5):
    """Check if the TCP endpoint is responding"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, int(port)))
        sock.close()
        return result == 0
    except Exception:
        return False

def test_openai_endpoint(base_url, model_name, timeout=10):
    """Test if the OpenAI-compatible endpoint is responding"""
    try:
        import openai
        client = openai.OpenAI(
            api_key="dummy",  # RunPod doesn't need real key
            base_url=base_url
        )
        
        client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10,
            timeout=timeout
        )
        return True
    except Exception:
        return False

def test_custom_endpoint(url, timeout=10):
    """Test a custom endpoint with HTTP GET"""
    try:
        response = requests.get(url, timeout=timeout)
        return response.status_code < 400
    except Exception:
        return False

def perform_health_check(health_check_config, pod_id, ports, direct_ip, port_mappings, full_model_path):
    """Perform health check based on configuration"""
    if not health_check_config:
        # Default behavior: OpenAI check for TCP ports, /health for HTTP
        has_tcp_ports = any('tcp' in port for port in ports)
        if has_tcp_ports:
            return perform_openai_health_check(pod_id, ports, direct_ip, port_mappings, full_model_path)
        else:
            return perform_http_health_check(pod_id, ports[0].split('/')[0])
    
    health_type = health_check_config.get('type', 'http')
    path = health_check_config.get('path', '/health')
    timeout = health_check_config.get('timeout', 300)
    
    if health_type == 'openai':
        return perform_openai_health_check(pod_id, ports, direct_ip, port_mappings, full_model_path, timeout)
    elif health_type == 'http':
        return perform_http_health_check(pod_id, ports[0].split('/')[0], path, timeout)
    elif health_type == 'tcp':
        return perform_tcp_health_check(pod_id, ports, direct_ip, port_mappings, timeout)
    elif health_type == 'none':
        print("Health check disabled by configuration")
        return True
    else:
        print(f"Unknown health check type: {health_type}, falling back to default")
        return perform_http_health_check(pod_id, ports[0].split('/')[0])

def perform_openai_health_check(pod_id, ports, direct_ip, port_mappings, full_model_path, timeout=600):
    """Perform OpenAI-style health check"""
    print("\n🧠 STAGE 3: Testing OpenAI-compatible endpoint...")
    
    start_time = time.time()
    connection_working = False
    
    while time.time() - start_time < timeout and not connection_working:
        # Test direct connection first
        if direct_ip and port_mappings:
            for port in ports:
                port_num = port.split('/')[0]
                if port_num in port_mappings:
                    external_port = port_mappings[port_num]
                    direct_url = f"http://{direct_ip}:{external_port}/v1"
                    print(f"⏳ Testing direct connection: {direct_url}")
                    
                    if test_openai_endpoint(direct_url, full_model_path):
                        print("✅ Direct connection working!")
                        connection_working = True
                        break
                    else:
                        print("❌ Direct connection failed, trying proxy...")
        
        # Test proxy connection if direct failed
        if not connection_working:
            for port in ports:
                port_num, protocol = port.split('/')
                if protocol == 'tcp':
                    proxy_url = f"http://{pod_id}-{port_num}.proxy.runpod.net/v1"
                    print(f"⏳ Testing proxy connection: {proxy_url}")
                    
                    if test_openai_endpoint(proxy_url, full_model_path):
                        print("✅ Proxy connection working!")
                        connection_working = True
                        break
                    else:
                        print("❌ Proxy connection failed")
        
        if not connection_working:
            elapsed = int(time.time() - start_time)
            print(f"⏳ Waiting for endpoint to be ready... ({elapsed}s elapsed)")
            time.sleep(10)
    
    if connection_working:
        print("\n🎉 OPENAI ENDPOINT READY!")
        print("   The pod is now ready to handle your requests!")
        print("   You can start making API calls to the endpoint above.")
        return True
    else:
        print("\n⚠️  OPENAI ENDPOINT NOT READY")
        print("   The pod is running but the endpoint may still be loading.")
        print("   You can try making requests, but they may fail until loading completes.")
        return False

def perform_http_health_check(pod_id, port_num, path="/health", timeout=300):
    """Perform HTTP health check"""
    print(f"\n🔍 STAGE 3: Testing HTTP endpoint at {path}...")
    print("\n⏳ Waiting for health check", end="", flush=True)
    
    start_time = time.time()
    while not check_http_endpoint_health(pod_id, port_num, path):
        if time.time() - start_time > timeout:
            print(f"\n⚠️  Health check timed out after {timeout}s")
            return False
        print(".", end="", flush=True)
        time.sleep(10)
    else:
        print(" ✅ Health check passed!")
        return True

def perform_tcp_health_check(pod_id, ports, direct_ip, port_mappings, timeout=300):
    """Perform basic TCP connectivity check"""
    print("\n🔌 STAGE 3: Testing TCP connectivity...")
    
    start_time = time.time()
    connection_working = False
    
    while time.time() - start_time < timeout and not connection_working:
        # Test direct connection first
        if direct_ip and port_mappings:
            for port in ports:
                port_num = port.split('/')[0]
                if port_num in port_mappings:
                    external_port = port_mappings[port_num]
                    print(f"⏳ Testing direct TCP connection: {direct_ip}:{external_port}")
                    
                    if check_tcp_endpoint_health(direct_ip, external_port):
                        print("✅ Direct TCP connection working!")
                        connection_working = True
                        break
                    else:
                        print("❌ Direct TCP connection failed")
        
        if not connection_working:
            elapsed = int(time.time() - start_time)
            print(f"⏳ Waiting for TCP endpoint... ({elapsed}s elapsed)")
            time.sleep(10)
    
    if connection_working:
        print("\n🎉 TCP ENDPOINT READY!")
        return True
    else:
        print(f"\n⚠️  TCP endpoint not ready after {timeout}s")
        return False

def check_http_endpoint_health(pod_id, port, health_path="/health"):
    """Check if the endpoint is responding"""
    try:
        health_url = f"https://{pod_id}-{port}.proxy.runpod.net{health_path}"
        response = requests.get(health_url, timeout=5)
        return response.status_code < 400
    except Exception:
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
    
    # Extract health check configuration before removing it from payload
    health_check_config = payload.pop('healthCheck', None)
    
    # Add extra arguments to docker command if provided
    if extra_args and 'dockerStartCmd' in payload:
        if isinstance(payload['dockerStartCmd'], list):
            payload['dockerStartCmd'].extend(extra_args)
        else:
            # Convert string to list and add args
            cmd_list = payload['dockerStartCmd'].split()
            cmd_list.extend(extra_args)
            payload['dockerStartCmd'] = cmd_list
    
    # Extract model name from arguments for better pod naming
    model_name = "unknown"
    full_model_path = "unknown"
    if extra_args:
        for i, arg in enumerate(extra_args):
            if arg == "--model-path" and i + 1 < len(extra_args):
                full_model_path = extra_args[i + 1]
                model_name = full_model_path.split('/')[-1]  # Get last part of model path
                break
    
    headers = {
        "Authorization": f"Bearer {os.environ['RUNPOD_API_KEY']}",
        "Content-Type": "application/json"
    }
    
    # Create a more descriptive pod name
    base_name = payload.get('name', 'sglang-pod')
    pod_name = f"{base_name}-{model_name}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    payload['name'] = pod_name
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
            return pod_id, health_check_config
        else:
            print(f"❌ Failed to create pod. Status: {response.status_code}")
            print(f"Response: {response.text}")
            return None, None
    except Exception as e:
        print(f"❌ Error creating pod: {e}")
        return None, None

def main():
    parser = argparse.ArgumentParser(
        description='Create and manage a RunPod pod using configuration templates with configurable health checks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s sglang -- --model-path Qwen/Qwen3-4B
  %(prog)s sglang --regions us_only -- --model-path Qwen/Qwen3-4B --reasoning-parser qwen3

Template names are loaded from the templates/ folder. File extensions are optional.
All arguments after '--' are passed directly to the docker command.

Region Configuration:
- Default regions are defined in config.py and used automatically
- Use --regions to override: default, us_only, eu_only, low_latency, cost_optimized
- Templates can specify dataCenterIds to override both defaults and --regions

Health Check Configuration:
Templates can include a 'healthCheck' section to configure health monitoring:
- type: "http" (test HTTP endpoint), "openai" (test OpenAI API), "tcp" (basic connectivity), "none" (skip)
- path: "/health" (for HTTP checks)
- timeout: 300 (timeout in seconds)

The healthCheck section is automatically stripped before sending to RunPod.
        """
    )
    
    parser.add_argument('template', help='Template name (from templates/ folder, extension optional)')
    parser.add_argument('--no-health-check', action='store_true',
                       help='Skip health check after pod creation')
    parser.add_argument('--debug', action='store_true',
                       help='Show debug information about pod status')
    parser.add_argument('--regions', 
                       choices=['default', 'us_only', 'eu_only', 'low_latency', 'cost_optimized'],
                       help='Region set to use (overrides template dataCenterIds)')
    
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
    
    # Override regions if specified
    if args.regions:
        config['dataCenterIds'] = get_regions(args.regions)
        print(f"🌍 Using {args.regions} regions: {config['dataCenterIds']}")
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Extract model path for testing
    full_model_path = "unknown"
    if extra_args:
        for i, arg in enumerate(extra_args):
            if arg == "--model-path" and i + 1 < len(extra_args):
                full_model_path = extra_args[i + 1]
                break
    
    # Create the pod
    pod_id, health_check_config = create_pod(config, extra_args)
    if not pod_id:
        sys.exit(1)
    
    # Stage 1: Wait for container to be running with necessary information
    print(f"\n🚀 STAGE 1: Waiting for container to be ready...")
    
    # Initial wait of 30 seconds
    print("⏳ Initial wait: 30 seconds...")
    time.sleep(30)
    
    # Then check every 10 seconds
    max_wait_time = 600  # 10 minutes total
    start_time = time.time()
    pod_info = None
    
    while time.time() - start_time < max_wait_time:
        pod_info = get_pod_status(pod_id)
        if not pod_info:
            cleanup_pod()
            sys.exit(1)
        
        # Check if pod is running and has the necessary information
        # For TCP ports, we need publicIp and portMappings
        # For HTTP ports, we just need the pod to be running
        ports = pod_info.get('ports', [])
        has_tcp_ports = any('tcp' in port for port in ports)
        
        if pod_info.get('desiredStatus') == 'RUNNING':
            if has_tcp_ports:
                # For TCP ports, wait for public IP and port mappings
                if pod_info.get('publicIp') and pod_info.get('portMappings'):
                    print(f"✅ Container is ready with IP: {pod_info.get('publicIp')}")
                    break
            else:
                # For HTTP-only ports, just need the pod to be running
                print(f"✅ Container is ready")
                break
        
        elapsed = int(time.time() - start_time)
        print(f"⏳ Waiting for container to be ready... ({elapsed}s elapsed)")
        time.sleep(10)
    else:
        print(f"\n⚠️  Container didn't become ready within {max_wait_time}s, continuing anyway...")
        pod_info = get_pod_status(pod_id)
        if not pod_info:
            cleanup_pod()
            sys.exit(1)
    
    # Debug: Show full pod info if requested
    if args.debug:
        print(f"\n🔍 DEBUG: Full pod info:")
        print(json.dumps(pod_info, indent=2))
    
    # Always show the raw pod info structure for debugging
    print(f"\n🔍 Pod info keys: {list(pod_info.keys())}")
    if 'runtime' in pod_info:
        print(f"🔍 Runtime keys: {list(pod_info['runtime'].keys())}")
        if 'network' in pod_info['runtime']:
            print(f"🔍 Network keys: {list(pod_info['runtime']['network'].keys())}")
    
    # Stage 2: Display connection information
    ports = pod_info.get('ports', [])
    direct_ip = pod_info.get('publicIp')
    port_mappings = pod_info.get('portMappings', {})
    
    print(f"\n🔌 STAGE 2: Connection Information")
    for port in ports:
        port_num, protocol = port.split('/')
        if protocol == 'http':
            print(f"🌐 HTTP endpoint: https://{pod_id}-{port_num}.proxy.runpod.net/")
        elif protocol == 'tcp':
            print(f"🔌 TCP proxy: {pod_id}-{port_num}.proxy.runpod.net:{port_num}")
        else:
            print(f"🔌 {protocol.upper()} proxy: {pod_id}-{port_num}.proxy.runpod.net:{port_num}")
    
    if direct_ip and port_mappings and any('tcp' in port for port in ports):
        print(f"\n🌍 DIRECT CONNECTION:")
        print(f"   IP Address: {direct_ip}")
        for port in ports:
            port_num = port.split('/')[0]
            if port_num in port_mappings:
                external_port = port_mappings[port_num]
                print(f"   Direct TCP: {direct_ip}:{external_port}")
            else:
                print(f"   Direct TCP: {direct_ip}:{port_num} (port mapping not found)")
    elif any('tcp' in port for port in ports):
        print(f"\n⚠️  No direct IP available for TCP ports - using proxy only")
    
    # Stage 3: Health checks
    if not args.no_health_check and ports:
        health_check_passed = perform_health_check(
            health_check_config, pod_id, ports, direct_ip, port_mappings, full_model_path
        )
        if health_check_passed:
            print("✅ Health check completed successfully!")
        else:
            print("⚠️  Health check did not pass, but continuing anyway.")
            print(f"   Check the RunPod console for more details: https://console.runpod.io/pods/{pod_id}")
    
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
