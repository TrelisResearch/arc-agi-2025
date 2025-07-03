#!/usr/bin/env python3

import json
import subprocess
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os

def run_tasks_and_collect_data():
    """Run 10 shortest tasks and collect results"""
    print("Running 10 shortest tasks on gpt-4o-mini with tools...")
    
    # Run the tasks
    cmd = [
        "uv", "run", "python", "run_arc_tasks.py",
        "--dataset", "arc-agi-1",
        "--subset", "shortest_10", 
        "--model", "gpt-4o-mini",
        "--tools"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    print("Tasks completed!")
    
    if result.returncode != 0:
        print(f"Error running tasks: {result.stderr}")
        return None
    
    # Find the most recent summary file
    log_files = [f for f in os.listdir("logs/") if f.endswith("_summary_arc-agi-1_shortest_10.json")]
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
    """Extract pixel accuracy and pattern learning metrics"""
    data = []
    
    for task_result in summary.get('results', []):
        # Get pixel accuracy
        score = task_result.get('score', {})
        if score.get('total_pixels', 0) > 0:
            pixel_accuracy = score.get('correct_pixels', 0) / score.get('total_pixels', 1)
        else:
            pixel_accuracy = 0.0
            
        # Get pattern learning metrics
        reduction = task_result.get('residual_reduction', {})
        program_residual = reduction.get('program_residual_bytes', 0)
        null_residual = reduction.get('null_residual_bytes', 1)  # Avoid division by zero
        
        # Calculate normalized pattern learning (program efficiency vs null baseline)
        if null_residual > 0:
            pattern_learning_ratio = program_residual / null_residual
        else:
            pattern_learning_ratio = 1.0  # No improvement over null
            
        # Get task info
        task_id = task_result.get('task_id', 'unknown')
        correct = score.get('correct', False)
        
        data.append({
            'task_id': task_id,
            'pixel_accuracy': pixel_accuracy * 100,  # Convert to percentage
            'pattern_learning_ratio': pattern_learning_ratio,
            'pattern_learning_percent': reduction.get('pattern_learning_score', 0),
            'correct': correct,
            'program_residual': program_residual,
            'null_residual': null_residual
        })
    
    return data

def create_plot(data):
    """Create scatter plot of pixel accuracy vs pattern learning ratio"""
    if not data:
        print("No data to plot!")
        return
    
    # Extract data for plotting
    pixel_accuracies = [d['pixel_accuracy'] for d in data]
    pattern_ratios = [d['pattern_learning_ratio'] for d in data]
    correct_tasks = [d['correct'] for d in data]
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    
    # Plot correct vs incorrect tasks with different colors
    correct_x = [pixel_accuracies[i] for i, correct in enumerate(correct_tasks) if correct]
    correct_y = [pattern_ratios[i] for i, correct in enumerate(correct_tasks) if correct]
    
    incorrect_x = [pixel_accuracies[i] for i, correct in enumerate(correct_tasks) if not correct]
    incorrect_y = [pattern_ratios[i] for i, correct in enumerate(correct_tasks) if not correct]
    
    # Add jitter to separate overlapping points at origin
    np.random.seed(42)  # For reproducible jitter
    jitter_x = np.random.normal(0, 1, len(incorrect_x))
    jitter_y = np.random.normal(0, 0.02, len(incorrect_y))
    
    # Only apply jitter to points at (0,0)
    jittered_incorrect_x = [x + (jx if x == 0 else 0) for x, jx in zip(incorrect_x, jitter_x)]
    jittered_incorrect_y = [y + (jy if y == 0 else 0) for y, jy in zip(incorrect_y, jitter_y)]
    
    plt.scatter(jittered_incorrect_x, jittered_incorrect_y, c='red', alpha=0.7, s=100, label='Incorrect', edgecolors='black')
    plt.scatter(correct_x, correct_y, c='green', alpha=0.7, s=100, label='Correct', edgecolors='black')
    
    # Count and annotate points at origin
    zero_points = sum(1 for x, y in zip(pixel_accuracies, pattern_ratios) if x == 0 and y == 0)
    if zero_points > 1:
        plt.annotate(f'{zero_points} tasks\nat origin', 
                    (2, 0.05), fontsize=10, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    # Add horizontal line at y=1.0 (null baseline)
    plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Null baseline (no learning)')
    
    # Add labels and title
    plt.xlabel('Pixel Accuracy (%)', fontsize=12)
    plt.ylabel('Pattern Learning Ratio (Program/Null Residual)', fontsize=12)
    plt.title('Pixel Accuracy vs Pattern Learning Efficiency\n(Lower ratio = better pattern learning)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Set axis limits
    plt.xlim(-5, 105)
    plt.ylim(0, max(pattern_ratios) * 1.1 if pattern_ratios else 2.0)
    
    # Add text annotations for interesting points
    for i, d in enumerate(data):
        if d['pixel_accuracy'] > 80 or d['pattern_learning_ratio'] < 0.5:
            plt.annotate(d['task_id'][:8], 
                        (d['pixel_accuracy'], d['pattern_learning_ratio']),
                        xytext=(5, 5), textcoords='offset points', 
                        fontsize=8, alpha=0.7)
    
    # Save the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"pattern_learning_analysis_{timestamp}.png"
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved as: {filename}")
    
    # Also show summary stats
    print("\n" + "="*50)
    print("ANALYSIS SUMMARY")
    print("="*50)
    print(f"Tasks analyzed: {len(data)}")
    print(f"Average pixel accuracy: {np.mean(pixel_accuracies):.1f}%")
    print(f"Average pattern learning ratio: {np.mean(pattern_ratios):.3f}")
    print(f"Tasks with >50% pixel accuracy: {sum(1 for p in pixel_accuracies if p > 50)}")
    print(f"Tasks with pattern learning ratio <1.0: {sum(1 for r in pattern_ratios if r < 1.0)}")
    print(f"Perfect solutions: {sum(correct_tasks)}")
    
    print("\nDetailed results:")
    for d in data:
        print(f"  {d['task_id']}: {d['pixel_accuracy']:.1f}% pixels, "
              f"ratio={d['pattern_learning_ratio']:.3f} "
              f"({d['program_residual']}/{d['null_residual']} bytes) "
              f"{'✓' if d['correct'] else '✗'}")

def main():
    print("Pattern Learning Analysis")
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