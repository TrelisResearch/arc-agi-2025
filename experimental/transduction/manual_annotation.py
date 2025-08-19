#!/usr/bin/env python3
"""
Manual Annotation Tool for ARC Programs
=======================================

This script allows manual annotation of ARC programs as transductive or inductive.
It loads programs from a parquet file, displays them one by one, and saves
annotations continuously to avoid data loss.

Usage:
    python manual_annotation.py input.parquet output.parquet

Features:
- Loads programs from input parquet file
- Displays programs with syntax highlighting and line numbers
- Skip already annotated programs
- Save continuously to output file
- Show progress and statistics
- Quit anytime with Ctrl+C or 'q'
"""

import argparse
import random
import pandas as pd
import sys
from pathlib import Path
import signal
from typing import Optional

# Add project root to path
project_root = next((parent for parent in [Path.cwd()] + list(Path.cwd().parents) if (parent / "pyproject.toml").exists()), Path.cwd())
sys.path.append(str(project_root))

def display_program(program: str, task_id: str, index: int, total: int, 
                   train_input: Optional[str] = None, test_input: Optional[str] = None) -> None:
    """Display a program with formatting and context"""
    print("\n" + "="*80)
    print(f"PROGRAM {index+1}/{total}")
    print(f"Task ID: {task_id}")
    print("="*80)
    
    # Display task context if available
    if train_input is not None or test_input is not None:
        print("TASK CONTEXT:")
        print("-" * 40)
        
        if train_input is not None:
            print("Train Input:")
            # Display train input with some formatting
            train_lines = str(train_input).split('\n')
            for line in train_lines[:10]:  # Limit to first 10 lines to avoid clutter
                print(f"  {line}")
            if len(train_lines) > 10:
                print(f"  ... ({len(train_lines) - 10} more lines)")
        
        if test_input is not None:
            print("\nTest Input:")
            # Display test input with some formatting  
            test_lines = str(test_input).split('\n')
            for line in test_lines[:10]:  # Limit to first 10 lines to avoid clutter
                print(f"  {line}")
            if len(test_lines) > 10:
                print(f"  ... ({len(test_lines) - 10} more lines)")
        
        print("-" * 40)
    
    print("PROGRAM CODE:")
    # Display program with line numbers
    lines = program.split('\n')
    for i, line in enumerate(lines, 1):
        print(f"{i:3d}: {line}")
    
    print("="*80)

def get_annotation() -> Optional[bool]:
    """Get user annotation for transductive/inductive classification"""
    while True:
        try:
            response = input("\nIs this program TRANSDUCTIVE? (y/n/s/q): ").lower().strip()
            
            if response in ['y', 'yes', '1', 't', 'true']:
                return True
            elif response in ['n', 'no', '0', 'f', 'false']:
                return False
            elif response in ['s', 'skip']:
                return None  # Skip this program
            elif response in ['q', 'quit', 'exit']:
                print("\nQuitting annotation session...")
                sys.exit(0)
            else:
                print("Please enter 'y' for transductive, 'n' for inductive, 's' to skip, or 'q' to quit")
                
        except (KeyboardInterrupt, EOFError):
            print("\n\nQuitting annotation session...")
            sys.exit(0)

def save_annotation(output_file: Path, program: str, task_id: str, is_transductive: bool) -> None:
    """Save a single annotation to the output file"""
    new_row = pd.DataFrame({
        'program': [program],
        'task_id': [task_id], 
        'transductive': [is_transductive]
    })
    
    # Append to existing file or create new one
    if output_file.exists():
        existing_df = pd.read_parquet(output_file)
        combined_df = pd.concat([existing_df, new_row], ignore_index=True)
    else:
        combined_df = new_row
    
    # Save to parquet
    combined_df.to_parquet(output_file, index=False)

def load_existing_annotations(output_file: Path) -> set:
    """Load already annotated task IDs to skip them"""
    if output_file.exists():
        existing_df = pd.read_parquet(output_file)
        return set(existing_df['task_id'].tolist())
    return set()

def show_progress(annotated_count: int, total_count: int, skipped_count: int) -> None:
    """Display current progress"""
    processed = annotated_count + skipped_count
    remaining = total_count - processed
    
    # Calculate breakdown from existing annotations
    print(f"\n📊 PROGRESS: {processed}/{total_count} programs processed")
    print(f"   • Annotated: {annotated_count}")
    print(f"   • Skipped: {skipped_count}")
    print(f"   • Remaining: {remaining}")
    if annotated_count > 0:
        completion_pct = 100 * processed / total_count
        print(f"   • Completion: {completion_pct:.1f}%")

def main():
    parser = argparse.ArgumentParser(description="Manual annotation tool for ARC programs")
    parser.add_argument("input_file", help="Input parquet file with programs")
    parser.add_argument("output_file", help="Output parquet file for annotations")
    parser.add_argument("--sample", type=int, default=None, 
                       help="Sample N random programs instead of processing all")
    parser.add_argument("--shuffle", action="store_true", default=True,
                       help="Shuffle programs before annotation")
    args = parser.parse_args()
    
    input_file = Path(args.input_file)
    output_file = Path(args.output_file)
    
    # Validate input file
    if not input_file.exists():
        print(f"❌ Input file not found: {input_file}")
        sys.exit(1)
    
    # Load input programs
    print(f"📂 Loading programs from: {input_file}")
    try:
        df = pd.read_parquet(input_file)
        print(f"✓ Loaded {len(df):,} programs")
    except Exception as e:
        print(f"❌ Error loading input file: {e}")
        sys.exit(1)
    
    # Validate required columns
    required_cols = ['code', 'task_id'] if 'code' in df.columns else ['program', 'task_id']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ Missing required columns: {missing_cols}")
        print(f"   Available columns: {list(df.columns)}")
        sys.exit(1)
    
    # Check for optional task context columns
    has_train_input = 'correct_train_input' in df.columns
    has_test_input = 'correct_test_input' in df.columns
    
    if has_train_input or has_test_input:
        context_cols = []
        if has_train_input:
            context_cols.append('correct_train_input')
        if has_test_input:
            context_cols.append('correct_test_input')
        print(f"✓ Found task context columns: {context_cols}")
    else:
        print("ℹ️  No task context columns found (correct_train_input, correct_test_input)")
    
    # Standardize column names
    if 'code' in df.columns:
        df = df.rename(columns={'code': 'program'})
    
    # Sample if requested
    if args.sample:
        if args.sample < len(df):
            df = df.sample(n=args.sample, random_state=42)
            print(f"🎯 Sampled {args.sample} programs for annotation")
    
    # Shuffle if requested
    if args.shuffle:
        df = df.sample(frac=1, random_state=random.randint(0, 10000)).reset_index(drop=True)
        print("🔀 Shuffled programs")
    
    # Load existing annotations
    already_annotated = load_existing_annotations(output_file)
    if already_annotated:
        print(f"📋 Found {len(already_annotated)} existing annotations")
    
    # Filter out already annotated programs
    df_remaining = df[~df['task_id'].isin(already_annotated)].copy()
    print(f"📝 {len(df_remaining)} programs remaining to annotate")
    
    if len(df_remaining) == 0:
        print("✅ All programs already annotated!")
        sys.exit(0)
    
    print("\n🎯 Starting annotation session...")
    print("Instructions:")
    print("  • 'y' or 'yes' = Transductive (hardcoded, specific to training data)")
    print("  • 'n' or 'no' = Inductive (generalizable, works on new data)")
    print("  • 's' or 'skip' = Skip this program")
    print("  • 'q' or 'quit' = Quit session")
    print("  • Ctrl+C = Quit session")
    
    # Set up signal handler for graceful exit
    def signal_handler(sig, frame):
        print("\n\n💾 Saving progress and quitting...")
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)
    
    # Annotation loop
    annotated_count = len(already_annotated)
    skipped_count = 0
    
    for index, (_, row) in enumerate(df_remaining.iterrows()):
        program = row['program']
        task_id = row['task_id']
        
        # Extract task context if available
        train_input = row.get('correct_train_input', None)
        test_input = row.get('correct_test_input', None)
        
        # Show progress every 10 programs
        if index % 10 == 0:
            show_progress(annotated_count, len(df), skipped_count)
        
        # Display program with context
        display_program(program, task_id, index, len(df_remaining), train_input, test_input)
        
        # Get annotation
        annotation = get_annotation()
        
        if annotation is None:
            # Skip this program
            skipped_count += 1
            print("⏭️  Skipped")
            continue
        
        # Save annotation
        try:
            save_annotation(output_file, program, task_id, annotation)
            annotated_count += 1
            
            label = "🔴 TRANSDUCTIVE" if annotation else "🟢 INDUCTIVE"
            print(f"✅ Saved: {label}")
            
        except Exception as e:
            print(f"❌ Error saving annotation: {e}")
            continue
    
    # Final summary
    print("\n" + "="*80)
    print("🎉 ANNOTATION SESSION COMPLETE!")
    print("="*80)
    show_progress(annotated_count, len(df), skipped_count)
    
    if output_file.exists():
        final_df = pd.read_parquet(output_file)
        transductive_count = final_df['transductive'].sum()
        inductive_count = len(final_df) - transductive_count
        transductive_pct = 100 * transductive_count / len(final_df)
        
        print("\n📊 FINAL ANNOTATIONS:")
        print(f"   • Total annotated: {len(final_df)}")
        print(f"   • Transductive: {transductive_count} ({transductive_pct:.1f}%)")
        print(f"   • Inductive: {inductive_count} ({100-transductive_pct:.1f}%)")
        print(f"💾 Saved to: {output_file}")

if __name__ == "__main__":
    main()
