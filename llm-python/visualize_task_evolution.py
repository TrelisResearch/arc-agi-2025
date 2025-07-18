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
        
        # Create plots directory
        self.plots_dir = Path("plots")
        self.plots_dir.mkdir(exist_ok=True)
    
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
    
    def plot_grid(self, ax, grid: List[List[int]], title: str, title_color: str = 'black'):
        """Plot a single grid with ARC colors"""
        if grid is None:
            ax.text(0.5, 0.5, 'No Output\n(Program Failed)', 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12, color='red')
            ax.set_title(title, fontweight='bold', color=title_color)
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
        ax.set_title(title, fontweight='bold', color=title_color)
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
    
    def create_turn_visualization(self, turn_number: int, program: str, task_data: Dict, 
                                task_id: str, model: str, log_stem: str):
        """Create a visualization for a single turn"""
        training_examples = task_data['train']
        test_input = task_data['test'][0]['input']
        test_output = task_data['test'][0]['output']
        
        n_training = len(training_examples)
        n_cols = 3  # Input, Expected Output, Predicted Output
        n_rows = n_training + 1  # Training examples + test example
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        # Training examples
        train_solved = 0
        train_total_accuracy = 0.0
        
        for i, example in enumerate(training_examples):
            # Input
            self.plot_grid(axes[i, 0], example['input'], f'Train {i+1} Input')
            
            # Expected output
            self.plot_grid(axes[i, 1], example['output'], f'Train {i+1} Expected')
            
            # Predicted output
            predicted = self.execute_program_on_test(program, example['input'])
            accuracy, correct_px, total_px = self.calculate_accuracy(predicted, example['output'])
            train_total_accuracy += accuracy
            
            if accuracy == 1.0:
                train_solved += 1
                title_color = 'green'
                title = f'Train {i+1} Predicted\n✅ SOLVED'
            else:
                title_color = 'red'
                title = f'Train {i+1} Predicted\n❌ {accuracy:.1%} ({correct_px}/{total_px})'
                
            self.plot_grid(axes[i, 2], predicted, title, title_color)
        
        # Test example
        test_row = n_training
        
        # Test input
        self.plot_grid(axes[test_row, 0], test_input, 'Test Input')
        
        # Test expected output
        self.plot_grid(axes[test_row, 1], test_output, 'Test Expected')
        
        # Test predicted output
        test_predicted = self.execute_program_on_test(program, test_input)
        test_accuracy, test_correct_px, test_total_px = self.calculate_accuracy(test_predicted, test_output)
        
        if test_accuracy == 1.0:
            test_title_color = 'green'
            test_title = f'Test Predicted\n✅ SOLVED'
        else:
            test_title_color = 'red'
            test_title = f'Test Predicted\n❌ {test_accuracy:.1%} ({test_correct_px}/{test_total_px})'
            
        self.plot_grid(axes[test_row, 2], test_predicted, test_title, test_title_color)
        
        # Calculate overall training accuracy
        avg_train_accuracy = train_total_accuracy / n_training if n_training > 0 else 0.0
        
        # Add title
        fig.suptitle(f'Task {task_id} - Turn {turn_number}\n'
                    f'Model: {model}\n'
                    f'Training: {train_solved}/{n_training} solved ({avg_train_accuracy:.1%} avg) | '
                    f'Test: {test_accuracy:.1%}',
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        # Create subdirectory for this log file
        log_plots_dir = self.plots_dir / log_stem
        log_plots_dir.mkdir(exist_ok=True)
        
        # Save the plot in log-specific subdirectory
        output_path = log_plots_dir / f"turn_{turn_number}_{task_id}_{log_stem}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()  # Close to free memory
        
        return {
            'turn_number': turn_number,
            'train_solved': train_solved,
            'train_total': n_training,
            'train_avg_accuracy': avg_train_accuracy,
            'test_accuracy': test_accuracy,
            'output_path': str(output_path)
        }
    
    def visualize_evolution(self, log_path: str, dataset: str = "arc-agi-1"):
        """Create individual visualization for each turn"""
        # Load log file
        log_data = self.load_log_file(log_path)
        task_id = log_data.get('task_id', 'unknown')
        model = log_data.get('model', 'unknown')
        log_stem = Path(log_path).stem
        
        print(f"Visualizing task evolution for: {task_id}")
        
        # Get ground truth data
        try:
            task_data = self.get_ground_truth(task_id, dataset)
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
        print(f"Creating individual visualizations in {self.plots_dir}/{log_stem}/...")
        
        # Create visualization for each turn
        results = []
        for turn in valid_turns:
            turn_number = turn['turn_number']
            program = turn['program']
            
            print(f"  Creating visualization for Turn {turn_number}...")
            result = self.create_turn_visualization(
                turn_number, program, task_data, task_id, model, log_stem
            )
            results.append(result)
            print(f"    Saved: {result['output_path']}")
        
        # Show summary
        print("\nEvolution Summary:")
        print("-" * 70)
        print(f"{'Turn':<6} {'Test Acc':<10} {'Train Solved':<12} {'Train Avg Acc':<12} {'File':<20}")
        print("-" * 70)
        
        for result in results:
            turn_num = result['turn_number']
            test_acc = result['test_accuracy']
            train_solved = result['train_solved']
            train_total = result['train_total']
            train_avg_acc = result['train_avg_accuracy']
            file_name = Path(result['output_path']).name
            
            test_status = "✅" if test_acc == 1.0 else f"{test_acc:.1%}"
            train_status = f"{train_solved}/{train_total}"
            
            print(f"{turn_num:<6} {test_status:<10} {train_status:<12} {train_avg_acc:.1%}        {file_name}")
        
        print(f"\nCreated {len(results)} turn visualizations for task {task_id}")
        print(f"All plots saved in: {self.plots_dir}/{log_stem}/")

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