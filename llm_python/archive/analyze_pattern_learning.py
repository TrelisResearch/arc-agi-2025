#!/usr/bin/env python3

import json
import subprocess
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os

def run_tasks_and_collect_data():
    """Run 10 middle tasks and collect results"""
    print("Running 10 middle tasks on gpt-4.1-mini with tools...")
    
    # Run the tasks
    cmd = [
        "uv", "run", "python", "run_arc_tasks.py",
        "--dataset", "arc-agi-1",
        "--subset", "middle_10", 
        "--model", "gpt-4.1-mini",
        "--tools"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    print("Tasks completed!")
    
    if result.returncode != 0:
        print(f"Error running tasks: {result.stderr}")
        return None
    
    # Find the most recent summary file
    log_files = [f for f in os.listdir("logs/") if f.endswith("_summary_arc-agi-1_middle_10.json")]
    if not log_files:
        print("No summary files found!")
        return None
    
    # Get the most recent summary file
    latest_file = max(log_files, key=lambda f: os.path.getctime(os.path.join("logs/", f)))
    summary_path = os.path.join("logs/", latest_file)
    
    # Load the results
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    
    print(f"Loaded results from: {summary_path}")
    return summary

def extract_metrics(summary):
    """Extract pixel accuracy and basic correctness metrics"""
    data = []
    
    for task_result in summary.get('results', []):
        # Get pixel accuracy
        score = task_result.get('score', {})
        if score.get('total_pixels', 0) > 0:
            pixel_accuracy = score.get('correct_pixels', 0) / score.get('total_pixels', 1)
        else:
            pixel_accuracy = 0.0
            
        # Get basic task info
        task_id = task_result.get('task_id', 'unknown')
        correct = score.get('correct', False)
        task_failure_reason = task_result.get('task_failure_reason', '')
        program_length = len(task_result.get('program', ''))
        
        # Check if task completed successfully
        has_task_failure = bool(task_failure_reason)
        
        data.append({
            'task_id': task_id,
            'pixel_accuracy': pixel_accuracy * 100,  # Convert to percentage
            'correct': correct,
            'has_task_failure': has_task_failure,
            'program_length': program_length,
            'task_failure_reason': task_failure_reason
        })
    
    return data

def create_plot(data):
    """Create scatter plot of pixel accuracy vs program length"""
    if not data:
        print("No data to plot!")
        return
    
    # Extract data for plotting
    pixel_accuracies = [d['pixel_accuracy'] for d in data]
    program_lengths = [d['program_length'] for d in data]
    correct_tasks = [d['correct'] for d in data]
    task_failures = [d['has_task_failure'] for d in data]
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    
    # Plot different categories with different colors and shapes
    correct_x = []
    correct_y = []
    incorrect_x = []
    incorrect_y = []
    error_x = []
    error_y = []
    
    for i, d in enumerate(data):
        if d['has_task_failure']:
            error_x.append(d['program_length'])
            error_y.append(d['pixel_accuracy'])
        elif d['correct']:
            correct_x.append(d['program_length'])
            correct_y.append(d['pixel_accuracy'])
        else:
            incorrect_x.append(d['program_length'])
            incorrect_y.append(d['pixel_accuracy'])
    
    # Plot points
    if error_x:
        plt.scatter(error_x, error_y, c='orange', alpha=0.7, s=100, label='Execution Error', edgecolors='black', marker='x')
    if incorrect_x:
        plt.scatter(incorrect_x, incorrect_y, c='red', alpha=0.7, s=100, label='Incorrect', edgecolors='black')
    if correct_x:
        plt.scatter(correct_x, correct_y, c='green', alpha=0.7, s=100, label='Correct', edgecolors='black')
    
    # Add labels and title
    plt.xlabel('Program Length (characters)', fontsize=12)
    plt.ylabel('Pixel Accuracy (%)', fontsize=12)
    plt.title('Pixel Accuracy vs Program Length', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Set axis limits
    plt.xlim(0, max(program_lengths) * 1.1 if program_lengths else 100)
    plt.ylim(-5, 105)
    
    # Add text annotations for interesting points
    for i, d in enumerate(data):
        if d['pixel_accuracy'] > 80 or d['program_length'] > 500:
            plt.annotate(d['task_id'][:8], 
                        (d['program_length'], d['pixel_accuracy']),
                        xytext=(5, 5), textcoords='offset points', 
                        fontsize=8, alpha=0.7)
    
    # Save the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"task_analysis_{timestamp}.png"
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved as: {filename}")
    
    # Also show summary stats
    print("\n" + "="*50)
    print("TASK ANALYSIS SUMMARY")
    print("="*50)
    print(f"Tasks analyzed: {len(data)}")
    print(f"Average pixel accuracy: {np.mean(pixel_accuracies):.1f}%")
    print(f"Average program length: {np.mean(program_lengths):.0f} characters")
    print(f"Tasks with >50% pixel accuracy: {sum(1 for p in pixel_accuracies if p > 50)}")
    print(f"Tasks with failures: {sum(1 for d in data if d['has_task_failure'])}")
    print(f"Perfect solutions: {sum(correct_tasks)}")
    
    print("\nDetailed results:")
    for d in data:
        status = "✓" if d['correct'] else "✗"
        error_info = " (FAILED)" if d['has_task_failure'] else ""
        print(f"  {d['task_id']}: {d['pixel_accuracy']:.1f}% pixels, "
              f"{d['program_length']} chars {status}{error_info}")

def main():
    print("Task Analysis")
    print("=" * 50)
    
    # Run tasks and collect data
    summary = run_tasks_and_collect_data()
    if summary is None:
        return
    
    # Extract metrics
    data = extract_metrics(summary)
    if not data:
        print("No data extracted from results!")
        return
    
    # Create and save plot
    create_plot(data)

if __name__ == "__main__":
    main() 