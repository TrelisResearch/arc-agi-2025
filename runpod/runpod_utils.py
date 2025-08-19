#!/usr/bin/env python3
"""Utilities for RunPod operations"""
import os
import requests


def delete_pod(pod_id):
    """Delete a RunPod pod by ID"""
    if not pod_id:
        return False
        
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
            return True
        elif response.status_code == 404:
            print(f"ℹ️  Pod {pod_id} already deleted or does not exist.")
            return True  # Consider it a success if pod is already gone
        else:
            print(f"❌ Failed to delete pod {pod_id}. Status: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error deleting pod: {e}")
        return False