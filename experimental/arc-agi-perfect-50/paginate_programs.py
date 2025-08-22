#!/usr/bin/env python3

import pandas as pd
import json
import sys
import os
from typing import Optional
import argparse

class ProgramPaginator:
    def __init__(self, dataset_path='dataset.parquet'):
        self.df = pd.read_parquet(dataset_path)
        self.current_task = None
        self.current_programs = None
        self.current_index = 0
        
    def load_task(self, task_id: str):
        """Load all programs for a specific task"""
        task_data = self.df[self.df['task_id'] == task_id].reset_index(drop=True)
        if len(task_data) == 0:
            print(f"Task {task_id} not found!")
            return False
        
        self.current_task = task_id
        self.current_programs = task_data
        self.current_index = 0
        return True
    
    def display_program(self, index: int):
        """Display a single program with stats"""
        if index < 0 or index >= len(self.current_programs):
            print(f"Invalid index: {index}")
            return
        
        row = self.current_programs.iloc[index]
        
        # Clear screen for better visibility
        os.system('clear' if os.name == 'posix' else 'cls')
        
        print("="*80)
        print(f"TASK: {row['task_id']} | Program {index + 1}/{len(self.current_programs)}")
        print("="*80)
        
        # Calculate stats
        train_correct = sum(row['correct_train_input'])
        train_total = len(row['correct_train_input'])
        test_correct = sum(row['correct_test_input'])
        test_total = len(row['correct_test_input'])
        
        print(f"\n📊 STATISTICS:")
        print(f"  Train predictions: {train_correct}/{train_total} correct")
        print(f"  Test predictions:  {test_correct}/{test_total} correct")
        print(f"  Model: {row['model']}")
        
        if row['reasoning'] and row['reasoning'].strip():
            print(f"\n💭 REASONING:")
            print("-"*40)
            # Limit reasoning display to first 500 chars if too long
            reasoning = row['reasoning']
            if len(reasoning) > 500:
                print(reasoning[:500] + "...[truncated]")
            else:
                print(reasoning)
        
        print(f"\n💻 CODE:")
        print("-"*40)
        print(row['code'])
        
        print(f"\n✅ CORRECTNESS DETAILS:")
        print(f"  Train inputs correct: {row['correct_train_input']}")
        print(f"  Test inputs correct:  {row['correct_test_input']}")
        
        # Show a sample of predictions if they're not too large
        if row['predicted_train_output'] is not None and len(row['predicted_train_output']) > 0:
            print(f"\n🎯 SAMPLE TRAIN OUTPUT (first prediction):")
            first_train = row['predicted_train_output'][0]
            if isinstance(first_train, list) and len(first_train) <= 10:
                for line in first_train[:5]:  # Show first 5 lines
                    if isinstance(line, list) and len(line) <= 20:
                        print(f"  {line}")
                    else:
                        print(f"  {str(line)[:50]}...")
                if len(first_train) > 5:
                    print(f"  ... ({len(first_train) - 5} more rows)")
        
        print("\n" + "="*80)
        print("Commands: [n]ext, [p]revious, [j]ump to index, [q]uit, [t]ask list")
        print("="*80)
    
    def run(self, initial_task: Optional[str] = None):
        """Main interaction loop"""
        if initial_task:
            if not self.load_task(initial_task):
                return
            self.display_program(0)
        else:
            self.show_task_list()
        
        while True:
            try:
                command = input("\n> ").strip().lower()
                
                if command == 'q' or command == 'quit':
                    print("Exiting...")
                    break
                    
                elif command == 'n' or command == 'next':
                    if self.current_programs is not None:
                        if self.current_index < len(self.current_programs) - 1:
                            self.current_index += 1
                            self.display_program(self.current_index)
                        else:
                            print("Already at last program!")
                    else:
                        print("No task loaded. Use 't' to see task list.")
                        
                elif command == 'p' or command == 'prev' or command == 'previous':
                    if self.current_programs is not None:
                        if self.current_index > 0:
                            self.current_index -= 1
                            self.display_program(self.current_index)
                        else:
                            print("Already at first program!")
                    else:
                        print("No task loaded. Use 't' to see task list.")
                        
                elif command.startswith('j ') or command.startswith('jump '):
                    if self.current_programs is not None:
                        try:
                            idx = int(command.split()[1]) - 1  # Convert to 0-based
                            if 0 <= idx < len(self.current_programs):
                                self.current_index = idx
                                self.display_program(self.current_index)
                            else:
                                print(f"Invalid index. Must be between 1 and {len(self.current_programs)}")
                        except (ValueError, IndexError):
                            print("Usage: j <number> or jump <number>")
                    else:
                        print("No task loaded. Use 't' to see task list.")
                        
                elif command == 't' or command == 'task' or command == 'tasks':
                    self.show_task_list()
                    
                elif command.startswith('load '):
                    task_id = command.split()[1]
                    if self.load_task(task_id):
                        self.display_program(0)
                        
                elif command == 'h' or command == 'help':
                    self.show_help()
                    
                else:
                    print("Unknown command. Type 'h' for help.")
                    
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def show_task_list(self):
        """Show available tasks with statistics"""
        print("\n" + "="*80)
        print("AVAILABLE TASKS")
        print("="*80)
        
        task_stats = []
        for task_id in self.df['task_id'].unique():
            task_data = self.df[self.df['task_id'] == task_id]
            num_programs = len(task_data)
            
            # Count perfect programs
            perfect_count = 0
            for _, row in task_data.iterrows():
                if all(row['correct_train_input']) and all(row['correct_test_input']):
                    perfect_count += 1
            
            task_stats.append({
                'task_id': task_id,
                'programs': num_programs,
                'perfect': perfect_count
            })
        
        # Sort by number of programs
        task_stats.sort(key=lambda x: x['programs'], reverse=True)
        
        # Show top tasks
        print(f"\nTop tasks by program count:")
        for i, stat in enumerate(task_stats[:20]):  # Show top 20
            print(f"  {stat['task_id']}: {stat['programs']} programs ({stat['perfect']} perfect)")
        
        if len(task_stats) > 20:
            print(f"\n... and {len(task_stats) - 20} more tasks")
        
        # Load selected tasks if available
        if os.path.exists('selected_tasks.json'):
            with open('selected_tasks.json', 'r') as f:
                selected = json.load(f)
            
            print(f"\n📌 Recommended tasks:")
            if selected.get('perfect_task'):
                print(f"  Perfect task (10+ all correct): {selected['perfect_task']}")
            if selected.get('fewest_task'):
                print(f"  Fewest programs: {selected['fewest_task']}")
        
        print(f"\nUse 'load <task_id>' to load a task")
    
    def show_help(self):
        """Show help message"""
        print("\n" + "="*40)
        print("HELP")
        print("="*40)
        print("Commands:")
        print("  n/next     - Show next program")
        print("  p/prev     - Show previous program")
        print("  j <num>    - Jump to program number")
        print("  t/tasks    - Show task list")
        print("  load <id>  - Load a specific task")
        print("  h/help     - Show this help")
        print("  q/quit     - Exit the program")


def main():
    parser = argparse.ArgumentParser(description='Browse ARC-AGI programs interactively')
    parser.add_argument('--task', '-t', help='Initial task ID to load')
    parser.add_argument('--dataset', '-d', default='dataset.parquet', help='Path to dataset file')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dataset):
        print(f"Error: Dataset file '{args.dataset}' not found!")
        print("Please run download_dataset.py first.")
        sys.exit(1)
    
    paginator = ProgramPaginator(args.dataset)
    
    print("\n🔍 ARC-AGI Program Browser")
    print("="*40)
    
    paginator.run(initial_task=args.task)


if __name__ == '__main__':
    main()