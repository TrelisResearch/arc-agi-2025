#!/usr/bin/env python3

import os
import json
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

from task_loader import TaskLoader
from scoring import ProgramExecutor

# ARC color palette - standard colors used in ARC tasks
ARC_COLORS = [
    '#000000',  # 0: black
    '#0074D9',  # 1: blue  
    '#FF4136',  # 2: red
    '#2ECC40',  # 3: green
    '#FFDC00',  # 4: yellow
    '#AAAAAA',  # 5: grey
    '#F012BE',  # 6: magenta
    '#FF851B',  # 7: orange
    '#7FDBFF',  # 8: sky blue
    '#870C25'   # 9: brown
]

class TaskEvolutionVisualizer:
    """Visualize the evolution of ARC task predictions across turns"""
    
    def __init__(self):
        self.task_loader = TaskLoader()
        self.executor = ProgramExecutor(timeout=2.0)
    
    def load_log_file(self, log_path: str) -> Dict:
        """Load and parse a log file"""
        log_file = Path(log_path)
        
        if not log_file.exists():
            # Try looking in the logs directory
            logs_dir = Path("logs")
            log_file = logs_dir / log_path
            
        if not log_file.exists():
            raise FileNotFoundError(f"Log file not found: {log_path}")
        
        with open(log_file, 'r') as f:
            return json.load(f)
    
    def get_ground_truth(self, task_id: str, dataset: str = "arc-agi-1") -> Dict:
        """Get the ground truth test data for a task"""
        # Try both datasets
        for ds in [dataset, "arc-agi-1", "arc-agi-2"]:
            try:
                task_data = self.task_loader.load_task(task_id, ds)
                if task_data and 'test' in task_data and len(task_data['test']) > 0:
                    return task_data
            except:
                continue
        
        raise ValueError(f"Could not load task {task_id} from any dataset")
    
    def execute_program_on_test(self, program: str, test_input: List[List[int]]) -> Optional[List[List[int]]]:
        """Execute a program on test input and return the result"""
        try:
            result, error, timed_out = self.executor.execute_program(program, test_input)
            if error or timed_out or result is None:
                return None
            return result
        except:
            return None
    
    def plot_grid(self, ax, grid: List[List[int]], title: str):
        """Plot a single grid with ARC colors"""
        if grid is None:
            ax.text(0.5, 0.5, 'No Output\n(Program Failed)', 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12, color='red')
            ax.set_title(title, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
            return
        
        grid_array = np.array(grid)
        height, width = grid_array.shape
        
        # Create color map
        cmap = plt.matplotlib.colors.ListedColormap(ARC_COLORS)
        
        # Plot the grid
        ax.imshow(grid_array, cmap=cmap, vmin=0, vmax=9)
        
        # Add grid lines
        for i in range(height + 1):
            ax.axhline(i - 0.5, color='white', linewidth=1)
        for j in range(width + 1):
            ax.axvline(j - 0.5, color='white', linewidth=1)
        
        # Set title and remove ticks
        ax.set_title(title, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add grid dimensions
        ax.text(0.02, 0.98, f'{height}×{width}', transform=ax.transAxes, 
               fontsize=9, verticalalignment='top', 
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    def calculate_accuracy(self, predicted: List[List[int]], target: List[List[int]]) -> tuple[float, int, int]:
        """Calculate pixel accuracy between predicted and target grids"""
        if predicted is None:
            target_pixels = len(target) * len(target[0]) if target else 0
            return 0.0, 0, target_pixels
        
        # Handle dimension mismatches
        min_rows = min(len(predicted), len(target))
        correct_pixels = 0
        total_pixels = 0
        
        for r in range(min_rows):
            min_cols = min(len(predicted[r]), len(target[r]))
            for c in range(min_cols):
                total_pixels += 1
                if predicted[r][c] == target[r][c]:
                    correct_pixels += 1
        
        # Account for dimension mismatches as errors
        target_size = len(target) * len(target[0]) if target else 0
        predicted_size = len(predicted) * len(predicted[0]) if predicted else 0
        total_pixels = max(target_size, predicted_size, total_pixels)
        
        accuracy = correct_pixels / total_pixels if total_pixels > 0 else 0.0
        return accuracy, correct_pixels, total_pixels
    
    def visualize_evolution(self, log_path: str, dataset: str = "arc-agi-1"):
        """Create visualization of task evolution across turns"""
        # Load log file
        log_data = self.load_log_file(log_path)
        task_id = log_data.get('task_id', 'unknown')
        
        print(f"Visualizing task evolution for: {task_id}")
        
        # Get ground truth data
        try:
            task_data = self.get_ground_truth(task_id, dataset)
            test_input = task_data['test'][0]['input']
            ground_truth = task_data['test'][0]['output']
        except Exception as e:
            print(f"Error loading ground truth: {e}")
            return
        
        # Extract turn details
        multiturn_data = log_data.get('multiturn_data', {})
        turn_details = multiturn_data.get('turn_details', [])
        
        if not turn_details:
            print("No turn details found in log file")
            return
        
        # Filter turns with valid programs
        valid_turns = []
        for turn in turn_details:
            if turn.get('program_extracted', False) and turn.get('program'):
                valid_turns.append(turn)
        
        if not valid_turns:
            print("No valid programs found in any turn")
            return
        
        print(f"Found {len(valid_turns)} turns with valid programs")
        
        # Execute programs and collect results
        turn_results = []
        for turn in valid_turns:
            program = turn['program']
            predicted = self.execute_program_on_test(program, test_input)
            accuracy, correct_pixels, total_pixels = self.calculate_accuracy(predicted, ground_truth)
            
            turn_results.append({
                'turn_number': turn['turn_number'],
                'predicted': predicted,
                'accuracy': accuracy,
                'correct_pixels': correct_pixels,
                'total_pixels': total_pixels,
                'program': program[:100] + "..." if len(program) > 100 else program
            })
        
        # Create visualization
        n_turns = len(turn_results)
        n_cols = min(4, n_turns + 1)  # +1 for ground truth, max 4 columns
        n_rows = (n_turns + n_cols) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        # Plot ground truth first
        self.plot_grid(axes[0, 0], ground_truth, "Ground Truth\n(Target Output)")
        
        # Plot predictions for each turn
        for i, result in enumerate(turn_results):
            row = (i + 1) // n_cols
            col = (i + 1) % n_cols
            
            turn_num = result['turn_number']
            accuracy = result['accuracy']
            correct_px = result['correct_pixels']
            total_px = result['total_pixels']
            
            title = f"Turn {turn_num}\n{accuracy:.1%} accuracy ({correct_px}/{total_px})"
            self.plot_grid(axes[row, col], result['predicted'], title)
        
        # Hide unused subplots
        for i in range(n_turns + 1, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].set_visible(False)
        
        # Add overall title
        final_accuracy = turn_results[-1]['accuracy'] if turn_results else 0.0
        final_turn = turn_results[-1]['turn_number'] if turn_results else 0
        model = log_data.get('model', 'unknown')
        
        fig.suptitle(f'Task Evolution: {task_id}\n'
                    f'Model: {model} | Final: Turn {final_turn} ({final_accuracy:.1%} accuracy)',
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        # Save the plot
        output_path = f"task_evolution_{task_id}_{Path(log_path).stem}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {output_path}")
        
        # Show summary
        print("\nEvolution Summary:")
        print("-" * 50)
        for result in turn_results:
            turn_num = result['turn_number']
            accuracy = result['accuracy']
            status = "✅ SOLVED" if accuracy == 1.0 else f"❌ {accuracy:.1%}"
            print(f"Turn {turn_num}: {status}")
        
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Visualize ARC task evolution across turns")
    parser.add_argument("log_file", help="Path to the log file (can be just filename if in logs/ directory)")
    parser.add_argument("--dataset", default="arc-agi-1", choices=["arc-agi-1", "arc-agi-2"],
                       help="Dataset to use for ground truth lookup (default: arc-agi-1)")
    
    args = parser.parse_args()
    
    visualizer = TaskEvolutionVisualizer()
    visualizer.visualize_evolution(args.log_file, args.dataset)

if __name__ == "__main__":
    main() 