#!/usr/bin/env python3
"""
Monitor the progress of the SOAR dataset analysis.
"""

import os
import json
from pathlib import Path

def monitor_soar_analysis():
    """Check the progress of the SOAR analysis if results file exists."""
    
    results_file = "classification/data/soar_classification_results.json"
    
    if not os.path.exists(results_file):
        print("📊 SOAR Analysis Status: RUNNING")
        print("⏳ No results file found yet - analysis still in progress...")
        print("💡 The analysis is classifying 620 programs with 8 parallel workers")
        print("🕒 Expected completion time: ~20-40 minutes")
        return False
    
    try:
        with open(results_file, 'r') as f:
            data = json.load(f)
        
        overall = data['overall']
        print("🎉 SOAR Analysis Status: COMPLETE!")
        print("=" * 50)
        print(f"✅ Total programs analyzed: {overall['total_programs']}")
        print(f"📊 Overall overfitting rate: {overall['overfitting_percentage']:.1f}%")
        print(f"🔥 Overfitting programs: {overall['overfitting_count']}")
        print(f"🧠 General programs: {overall['general_count']}")
        
        if overall['error_count'] > 0:
            print(f"⚠️  Programs with errors: {overall['error_count']}")
        
        print(f"\n📋 Analysis by task: {len(data['by_task'])} tasks analyzed")
        print(f"🤖 Analysis by model: {len(data['by_model'])} models analyzed")
        
        # Show top overfitting tasks
        top_overfitting = sorted(data['by_task'], key=lambda x: x['overfitting_percentage'], reverse=True)[:5]
        print(f"\n🔝 Top 5 tasks by overfitting rate:")
        for i, task in enumerate(top_overfitting):
            print(f"   {i+1}. {task['task_id']}: {task['overfitting_percentage']:.1f}% ({task['overfitting_count']}/{task['total_programs']})")
        
        return True
        
    except Exception as e:
        print(f"❌ Error reading results: {e}")
        return False

if __name__ == "__main__":
    monitor_soar_analysis()