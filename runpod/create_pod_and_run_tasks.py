#!/usr/bin/env python3
import subprocess
import sys
import os
import argparse
import signal
import time
import requests
import select
from dotenv import load_dotenv
import sys
sys.path.insert(0, os.path.dirname(__file__))
from runpod_utils import delete_pod

load_dotenv()


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

def prompt_keep_pod(timeout=10):
    """Prompt user whether to keep pod running, with timeout"""
    print(f"\n💰 Keep pod running for manual use? [y/N] (timeout in {timeout}s): ", end="", flush=True)
    
    # Use select for timeout on Unix systems
    try:
        if hasattr(select, 'select'):
            ready, _, _ = select.select([sys.stdin], [], [], timeout)
            if ready:
                response = sys.stdin.readline().strip().lower()
                return response.startswith('y')
            else:
                print("\n⏰ Timeout - defaulting to No (will delete pod)")
                return False
        else:
            # Fallback for Windows - no timeout
            response = input().strip().lower()
            return response.startswith('y')
    except Exception:
        return False

def run_arc_tasks_with_graceful_handling(dataset, model_path, base_url, subset="all_evaluation", max_attempts=64, no_transductive_penalty=False, max_workers=32):
    """Run ARC tasks - task runner handles its own graceful shutdown"""
    print(f"\n🎯 Running ARC tasks for {dataset} with subset {subset}...")
    
    cmd = [
        "uv", "run", "python", "-m", "llm_python.run_arc_tasks_soar",
        "--dataset", dataset,
        "--subset", subset,
        "--max_workers", str(max_workers),
        "--max_attempts", str(max_attempts),
        "--model", model_path,
        "--base-url", base_url,
        "--unsafe-executor",
        "--max-tokens", "2000",
        "--qwen-no-think"
    ]
    
    if no_transductive_penalty:
        cmd.append("--no-transductive-penalty")
    
    print(f"📝 Running command: {' '.join(cmd)}")
    
    # Start subprocess
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                              universal_newlines=True, bufsize=1)
    
    try:
        # Stream output in real-time
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                print(line.rstrip())
        
        # Wait for process to complete
        return_code = process.wait()
        
        if return_code == 0:
            print(f"✅ ARC tasks completed successfully for {dataset}!")
            return True
        else:
            print(f"❌ ARC tasks failed with return code {return_code}")
            return False
    
    except KeyboardInterrupt:
        print(f"\n🛑 Ctrl+C received - forwarding to ARC tasks for graceful shutdown...")
        
        # Forward SIGINT to subprocess
        process.send_signal(signal.SIGINT)
        
        # Wait for graceful shutdown (task runner handles this well)
        print(f"⏳ Waiting for ARC tasks to complete gracefully...")
        try:
            return_code = process.wait(timeout=301)  # Just over API timeout for safety
            print(f"✅ ARC tasks shut down gracefully")
            return False  # Interrupted = not successful
        except subprocess.TimeoutExpired:
            print(f"⚠️  ARC tasks didn't shut down gracefully, terminating...")
            process.terminate()
            process.wait()
            return False

def terminate_pod_process(pod_process):
    """Terminate the pod creation subprocess gracefully"""
    if pod_process and pod_process.poll() is None:
        print(f"\n🛑 Terminating pod creation...")
        pod_process.send_signal(signal.SIGINT)
        try:
            pod_process.wait(timeout=30)
            print(f"✅ Pod creation shut down gracefully")
        except subprocess.TimeoutExpired:
            print(f"⚠️  Pod creation didn't respond, force terminating...")
            pod_process.terminate()
            pod_process.wait()

def handle_cleanup_decision(pod_id, base_url=None):
    """Handle the decision to keep or delete the pod"""
    if not pod_id:
        return
    
    print(f"\n💰 Pod {pod_id} is still running and incurring costs.")
    keep_pod = prompt_keep_pod()
    
    if keep_pod:
        print(f"\n🚀 Keeping pod running for manual use:")
        if base_url:
            print(f"   Endpoint: {base_url}")
        print(f"   Console: https://console.runpod.io/pods/{pod_id}")
        print(f"   ⚠️  Remember to delete the pod when done to stop charges!")
    else:
        delete_pod(pod_id)

def terminate_and_cleanup(pod_process, pod_id, base_url=None):
    """Terminate pod process and handle cleanup decision"""
    terminate_pod_process(pod_process)
    handle_cleanup_decision(pod_id, base_url)

def main():
    parser = argparse.ArgumentParser(
        description='Create a RunPod pod and automatically run ARC tasks once ready',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s arc-agi-1 Trelis/Qwen3-4B_dsarc-agi-1-train-programs-best-length-filtered-250_20250811-155856-c904
  %(prog)s arc-agi-2 Trelis/Qwen3-4B_dsarc-agi-2-custom-model --subset shortest_1
  %(prog)s arc-agi-1 Trelis/Qwen3-4B_model --subset all_evaluation --max_attempts 32
  %(prog)s arc-agi-2 Trelis/arc-1-fake-ttt-blended-c802-FP8-Dynamic --subset all_evaluation --kv-cache-dtype fp8_e5m2

This script will:
1. Create a RunPod pod with the specified model
2. Wait for the OpenAI endpoint to be ready
3. Run ARC tasks on specified subset (default: all_evaluation) with configurable max attempts (default: 64)
4. Keep the pod running until you press Ctrl+C
        """
    )
    
    parser.add_argument('dataset', 
                       default='arc-prize-2025',
                       help='Dataset to run (e.g., arc-prize-2025, arc-agi-1, arc-agi-2)')
    parser.add_argument('model_path', 
                       help='Full model path (e.g., Trelis/Qwen3-4B_...)')
    parser.add_argument('--template', 
                       default='sglang',
                       help='RunPod template to use (default: sglang)')
    parser.add_argument('--subset',
                       default='evaluation',
                       help='Dataset subset to run (default: evaluation)')
    parser.add_argument('--skip-tasks',
                       action='store_true',
                       help='Skip running ARC tasks (only create pod)')
    parser.add_argument('--no-health-check',
                       action='store_true',
                       help='Skip health check after pod creation')
    parser.add_argument('--kv-cache-dtype',
                       type=str,
                       help='KV cache data type for sglang servers (e.g., fp8_e5m2)')
    parser.add_argument('--max-attempts', '--max_attempts',
                       type=int,
                       default=64,
                       dest='max_attempts',
                       help='Maximum number of attempts for ARC tasks (default: 64)')
    parser.add_argument('--no-transductive-penalty',
                       action='store_true',
                       help='Disable transductive penalty in voting (passed to run_arc_tasks_soar.py)')
    parser.add_argument('--max-workers',
                       type=int,
                       default=32,
                       help='Maximum number of parallel workers (default: 32)')
    
    args = parser.parse_args()
    
    # Check for required environment variable
    if 'RUNPOD_API_KEY' not in os.environ:
        print("❌ Error: RUNPOD_API_KEY environment variable not set!")
        sys.exit(1)
    
    
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
                    
    except KeyboardInterrupt:
        print(f"\n🛑 Ctrl+C during pod creation - cleaning up...")
        terminate_pod_process(pod_process)
        # create_pod.py handles its own cleanup when terminated
        print(f"✅ Pod cleanup handled by create_pod.py")
        sys.exit(0)
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
        terminate_pod_process(pod_process)
        sys.exit(1)
    
    # Additional wait to ensure the endpoint is ready
    if not args.skip_tasks:
        # Double-check the endpoint is ready
        if not wait_for_openai_endpoint(base_url, args.model_path, timeout=300):
            print("❌ OpenAI endpoint failed to become ready")
            terminate_pod_process(pod_process)
            sys.exit(1)
        
        # Step 2: Run ARC tasks
        print(f"\n🎯 Step 2: Running ARC tasks for {args.dataset}")
        
        task_success = run_arc_tasks_with_graceful_handling(
            args.dataset, args.model_path, base_url, args.subset, args.max_attempts, args.no_transductive_penalty, args.max_workers
        )
        
        if task_success:
            print(f"\n🎉 All tasks completed successfully!")
        else:
            print(f"\n⚠️  Tasks completed with issues or were cancelled.")
        
        # Always just prompt for cleanup (keep pod process running)
        handle_cleanup_decision(pod_id, base_url)
    else:
        print("\n⏭️  Skipping ARC tasks as requested")
        
        # Step 3: Keep the create_pod.py process running until user decides
        print(f"\n💡 Pod is running. Press Ctrl+C when ready to decide about pod.")
        
        try:
            # Wait for the pod process to complete (it will run until Ctrl+C)
            pod_process.wait()
        except KeyboardInterrupt:
            print(f"\n🛑 Ctrl+C received")
        
        # Always terminate and prompt in skip-tasks mode
        terminate_and_cleanup(pod_process, pod_id, base_url)

if __name__ == "__main__":
    main()