#!/usr/bin/env python3
import subprocess
import sys
import os
import argparse
import signal
import time
import requests
from dotenv import load_dotenv

load_dotenv()

def signal_handler(signum, frame):
    """Handle interrupt signals for graceful shutdown"""
    print(f"\n🛑 Received signal {signum}. Exiting...")
    sys.exit(0)

def wait_for_openai_endpoint(base_url, model_name, timeout=600, check_interval=10):
    """Wait for the OpenAI-compatible endpoint to be ready"""
    print(f"\n🧠 Waiting for OpenAI endpoint at {base_url} to be ready...")
    
    start_time = time.time()
    while time.time() - start_time < timeout:
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
                timeout=10
            )
            print("✅ OpenAI endpoint is ready!")
            return True
        except Exception as e:
            elapsed = int(time.time() - start_time)
            print(f"⏳ Endpoint not ready yet... ({elapsed}s elapsed)")
            time.sleep(check_interval)
    
    print(f"❌ Endpoint failed to become ready after {timeout}s")
    return False

def run_arc_tasks(dataset, model_path, base_url, subset="all_evaluation"):
    """Run the ARC tasks with the specified configuration"""
    print(f"\n🎯 Running ARC tasks for {dataset} with subset {subset}...")
    
    # Extract model name from path for display
    model_name = model_path.split('/')[-1]
    
    cmd = [
        "uv", "run", "python", "-m", "llm_python.run_arc_tasks_soar",
        "--dataset", dataset,
        "--subset", subset,
        "--max_workers", "32",
        "--max_attempts", "64",
        "--model", model_path,
        "--base-url", base_url,
        "--unsafe-executor",
        "--max-tokens", "2000",
        "--qwen-no-think"
    ]
    
    print(f"📝 Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True)
        print(f"✅ ARC tasks completed successfully for {dataset}!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running ARC tasks: {e}")
        return False
    except KeyboardInterrupt:
        print("\n⚠️  Task execution interrupted by user")
        return False

def main():
    parser = argparse.ArgumentParser(
        description='Create a RunPod pod and automatically run ARC tasks once ready',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s arc-agi-1 Trelis/Qwen3-4B_dsarc-agi-1-train-programs-best-length-filtered-250_20250811-155856-c904
  %(prog)s arc-agi-2 Trelis/Qwen3-4B_dsarc-agi-2-custom-model --subset shortest_1
  %(prog)s arc-agi-1 Trelis/Qwen3-4B_model --subset all_evaluation
  %(prog)s arc-agi-2 Trelis/arc-1-fake-ttt-blended-c802-FP8-Dynamic --subset all_evaluation --kv-cache-dtype fp8_e5m2

This script will:
1. Create a RunPod pod with the specified model
2. Wait for the OpenAI endpoint to be ready
3. Run ARC tasks with 8 attempts and 3 runs on specified subset (default: all_evaluation)
4. Keep the pod running until you press Ctrl+C
        """
    )
    
    parser.add_argument('dataset', 
                       choices=['arc-agi-1', 'arc-agi-2'],
                       help='Dataset to run (arc-agi-1 or arc-agi-2)')
    parser.add_argument('model_path', 
                       help='Full model path (e.g., Trelis/Qwen3-4B_...)')
    parser.add_argument('--template', 
                       default='sglang',
                       help='RunPod template to use (default: sglang)')
    parser.add_argument('--skip-tasks',
                       action='store_true',
                       help='Skip running ARC tasks (only create pod)')
    parser.add_argument('--no-health-check',
                       action='store_true',
                       help='Skip health check after pod creation')
    parser.add_argument('--subset',
                       default='all_evaluation',
                       help='Dataset subset to run (default: all_evaluation)')
    parser.add_argument('--kv-cache-dtype',
                       type=str,
                       help='KV cache data type for sglang servers (e.g., fp8_e5m2)')
    
    args = parser.parse_args()
    
    # Check for required environment variable
    if 'RUNPOD_API_KEY' not in os.environ:
        print("❌ Error: RUNPOD_API_KEY environment variable not set!")
        sys.exit(1)
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Step 1: Create the pod
    print(f"🚀 Step 1: Creating RunPod pod with model {args.model_path}")
    
    create_cmd = [
        "uv", "run", "python", "runpod/create_pod.py", 
        args.template,
        "--",
        "--model-path", args.model_path
    ]
    
    # Add kv-cache-dtype parameter if provided
    if args.kv_cache_dtype:
        create_cmd.extend(["--kv-cache-dtype", args.kv_cache_dtype])
    
    if args.no_health_check:
        create_cmd.insert(5, "--no-health-check")  # Insert after "create_pod.py"
    
    print(f"📝 Running: {' '.join(create_cmd)}")
    
    # Start the pod creation process with unbuffered output
    # Set environment to ensure unbuffered output
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    
    pod_process = subprocess.Popen(
        create_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=0,  # Use unbuffered mode
        env=env
    )
    
    # Variables to track pod status
    pod_id = None
    direct_ip = None
    external_port = None
    proxy_url = None
    endpoint_ready = False
    
    # Read output from the pod creation process
    print("\n--- Pod Creation Output ---")
    print("", flush=True)
    
    try:
        while True:
            line = pod_process.stdout.readline()
            if not line and pod_process.poll() is not None:
                break
            if line:
                # Print immediately with flush
                sys.stdout.write(line)
                sys.stdout.flush()
                
                # Parse pod ID
                if "Pod created successfully! ID:" in line:
                    pod_id = line.split("ID:")[1].strip()
                
                # Parse direct connection info
                if "Direct TCP:" in line and direct_ip is None:
                    parts = line.split("Direct TCP:")[1].strip()
                    if ":" in parts:
                        direct_ip, external_port = parts.split(":")
                        direct_ip = direct_ip.strip()
                        external_port = external_port.strip()
                
                # Parse proxy connection
                if "TCP proxy:" in line and proxy_url is None:
                    proxy_url = line.split("TCP proxy:")[1].strip()
                
                # Check if endpoint is ready
                if "OPENAI ENDPOINT READY!" in line:
                    endpoint_ready = True
                    # Don't break, keep reading to get all output
                
                # Check for health check completion
                if "Health check completed successfully!" in line:
                    endpoint_ready = True
                
                # Check if health check was skipped
                if args.no_health_check and "The pod is now running" in line:
                    endpoint_ready = True
                
                # Check if waiting message appears (pod is running, keep alive)
                if "The pod is now running. Press Ctrl+C" in line:
                    endpoint_ready = True
                    break  # This is where create_pod.py enters its wait loop
                    
    except Exception as e:
        print(f"\n❌ Error reading pod creation output: {e}")
        pass  # Error details already printed
    
    # Check if process ended with error
    if not endpoint_ready and pod_process.poll() is not None:
        print(f"\n❌ Pod creation process ended unexpectedly with code: {pod_process.returncode}")
        sys.exit(1)
    
    # Determine the base URL to use
    base_url = None
    if direct_ip and external_port:
        base_url = f"http://{direct_ip}:{external_port}/v1"
        print(f"\n🌍 Using direct connection: {base_url}")
    elif pod_id:
        # Use proxy if no direct connection available
        base_url = f"http://{pod_id}-8080.proxy.runpod.net/v1"
        print(f"\n🔌 Using proxy connection: {base_url}")
    else:
        print("\n❌ Failed to determine pod endpoint URL")
        if pod_process.poll() is None:
            pod_process.terminate()
        sys.exit(1)
    
    # Additional wait to ensure the endpoint is ready
    if not args.skip_tasks:
        # Double-check the endpoint is ready
        if not wait_for_openai_endpoint(base_url, args.model_path, timeout=300):
            print("❌ OpenAI endpoint failed to become ready")
            pod_process.terminate()
            sys.exit(1)
        
        # Step 2: Run ARC tasks
        print(f"\n🎯 Step 2: Running ARC tasks for {args.dataset}")
        
        if run_arc_tasks(args.dataset, args.model_path, base_url, args.subset):
            print(f"\n🎉 All tasks completed successfully!")
        else:
            print(f"\n⚠️  Some tasks may have failed. Check the output above for details.")
    else:
        print("\n⏭️  Skipping ARC tasks as requested")
    
    # Step 3: Keep the pod running
    print(f"\n💡 The pod is still running. Press Ctrl+C to stop and delete the pod.")
    print(f"   You can continue to use the endpoint at: {base_url}")
    
    try:
        # Wait for the pod process to complete (it will run until Ctrl+C)
        pod_process.wait()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        pod_process.terminate()
        pod_process.wait()

if __name__ == "__main__":
    main()