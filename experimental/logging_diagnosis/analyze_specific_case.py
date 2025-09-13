#!/usr/bin/env python3
"""
Deep analysis of the specific case from the log:
"10 train-partial (of which 3 trans), 47 train-incorrect (of which 4 trans) (best: 0.0% train)"

This investigates how exactly this combination could occur.
"""

import sys
from typing import List, Dict, Any
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from test_logging_inconsistency import simulate_logging_logic


def analyze_observed_log_case():
    """
    Analyze the exact case: 10 train-partial, 47 train-incorrect, best: 0.0% train

    From log: "✅ f9012d9b: 64 attempts | 57 valid outputs, 4 execution errors, 3 invalid outputs |
               2 test-perfect, 55 test-incorrect (of which 7 trans) |
               10 train-partial (of which 3 trans), 47 train-incorrect (of which 4 trans) (best: 0.0% train)"
    """
    print("=== ANALYZING OBSERVED LOG CASE ===")
    print("Constraints from log:")
    print("- 64 total attempts")
    print("- 57 valid outputs, 4 execution errors, 3 invalid outputs")
    print("- 2 test-perfect, 55 test-incorrect")
    print("- 10 train-partial (3 transductive), 47 train-incorrect (4 transductive)")
    print("- best: 0.0% train")
    print()

    # The mystery: How can we have 10 train-partial but best is 0.0%?
    # Let's construct scenarios that match these exact numbers

    print("🔍 SCENARIO ANALYSIS:")
    print()

    # Scenario 1: All train-partial programs have invalid outputs
    print("Scenario 1: All 10 train-partial programs have invalid outputs")
    attempts_scenario_1 = []

    # Add 10 train-partial programs with INVALID outputs (won't be counted in categorization)
    for i in range(10):
        attempts_scenario_1.append({
            "train_accuracy": 0.2 + i * 0.05,  # 20% to 65% train accuracy
            "test_correct": False,
            "outputs_valid": False,  # This is KEY - invalid outputs!
            "is_transductive": i < 3,  # First 3 are transductive
        })

    # Add 47 train-incorrect programs with valid outputs
    for i in range(47):
        attempts_scenario_1.append({
            "train_accuracy": 0.0,  # 0% train accuracy
            "test_correct": False,
            "outputs_valid": True,
            "is_transductive": i < 4,  # First 4 are transductive
        })

    # Add 2 test-perfect programs
    for i in range(2):
        attempts_scenario_1.append({
            "train_accuracy": 0.0,  # Could be 0% train but still test-perfect (lucky guess)
            "test_correct": True,
            "outputs_valid": True,
            "is_transductive": False,
        })

    # Add remaining attempts to reach 64 total
    remaining = 64 - len(attempts_scenario_1)
    for i in range(remaining):
        attempts_scenario_1.append({
            "train_accuracy": 0.0,
            "test_correct": False,
            "outputs_valid": True,
            "is_transductive": False,
        })

    result1 = simulate_logging_logic(attempts_scenario_1)

    print(f"  Train partial count: {result1['train_partial_total']} (expected: 0 due to invalid outputs)")
    print(f"  Train incorrect count: {result1['train_incorrect_total']} (expected: ≥47)")
    print(f"  Best train accuracy: {result1['best_train_accuracy']:.1%} (expected: >0% from invalid partials)")
    print(f"  Best test correct: {result1['best_test_correct']}")
    print()

    if result1['train_partial_total'] == 0 and result1['best_train_accuracy'] > 0:
        print("✅ Scenario 1 EXPLAINS the discrepancy!")
        print("   Train-partial programs exist but have invalid outputs,")
        print("   so they're excluded from categorization but included in 'best' selection.")
    print()

    # Scenario 2: Test-perfect programs with 0% train beat the partials
    print("Scenario 2: Train-partial programs are valid, but test-perfect programs with 0% train win")
    attempts_scenario_2 = []

    # Add 10 train-partial programs with VALID outputs
    for i in range(10):
        attempts_scenario_2.append({
            "train_accuracy": 0.2 + i * 0.05,  # 20% to 65% train accuracy
            "test_correct": False,
            "outputs_valid": True,  # Valid outputs this time
            "is_transductive": i < 3,  # First 3 are transductive
        })

    # Add 47 train-incorrect programs
    for i in range(47):
        attempts_scenario_2.append({
            "train_accuracy": 0.0,
            "test_correct": False,
            "outputs_valid": True,
            "is_transductive": i < 4,
        })

    # Add 2 test-perfect programs with 0% train (these will win due to sort key)
    for i in range(2):
        attempts_scenario_2.append({
            "train_accuracy": 0.0,  # 0% train but test-perfect
            "test_correct": True,   # This beats everything else in sort
            "outputs_valid": True,
            "is_transductive": False,
        })

    # Add remaining attempts
    remaining = 64 - len(attempts_scenario_2)
    for i in range(remaining):
        attempts_scenario_2.append({
            "train_accuracy": 0.0,
            "test_correct": False,
            "outputs_valid": True,
            "is_transductive": False,
        })

    result2 = simulate_logging_logic(attempts_scenario_2)

    print(f"  Train partial count: {result2['train_partial_total']} (expected: 10)")
    print(f"  Train incorrect count: {result2['train_incorrect_total']} (expected: ≥47)")
    print(f"  Best train accuracy: {result2['best_train_accuracy']:.1%} (expected: 0% due to test-correct priority)")
    print(f"  Best test correct: {result2['best_test_correct']} (expected: True)")
    print()

    if result2['train_partial_total'] == 10 and result2['best_train_accuracy'] == 0.0:
        print("✅ Scenario 2 ALSO EXPLAINS the discrepancy!")
        print("   Train-partial programs are counted, but test-perfect with 0% train wins 'best'")
        print("   due to sort key prioritizing test_correct over train_accuracy.")
    print()


def create_exact_reproduction():
    """
    Try to create the EXACT numbers from the log
    """
    print("=== EXACT REPRODUCTION ATTEMPT ===")
    print("Target: 10 train-partial (3 trans), 47 train-incorrect (4 trans), best: 0.0% train")
    print()

    attempts = []

    # We need the train categorization to show 0 partials despite having partial programs
    # This means all partial programs must have outputs_valid=False

    # Method: Mix of invalid partials + valid incorrects + test-perfect with 0% train

    # 10 programs that WOULD BE train-partial if they had valid outputs (but they don't)
    partial_programs_invalid = []
    for i in range(10):
        partial_programs_invalid.append({
            "train_accuracy": 0.1 + i * 0.07,  # 10% to 73% (partial range)
            "test_correct": False,
            "outputs_valid": False,  # KEY: Invalid outputs!
            "is_transductive": i < 3,  # 3 transductive
        })

    # 47 programs with 0% train accuracy and valid outputs
    incorrect_programs_valid = []
    for i in range(47):
        incorrect_programs_valid.append({
            "train_accuracy": 0.0,
            "test_correct": False,
            "outputs_valid": True,
            "is_transductive": i < 4,  # 4 transductive
        })

    # 2 test-perfect programs (these will be "best" due to test_correct=True)
    test_perfect_programs = []
    for i in range(2):
        test_perfect_programs.append({
            "train_accuracy": 0.0,  # 0% train but test perfect!
            "test_correct": True,
            "outputs_valid": True,
            "is_transductive": False,
        })

    # Remaining 5 programs (64 - 10 - 47 - 2 = 5)
    remaining_programs = []
    for i in range(5):
        remaining_programs.append({
            "train_accuracy": 0.0,
            "test_correct": False,
            "outputs_valid": True,  # Could be invalid outputs (execution errors)
            "is_transductive": False,
        })

    # Combine all
    attempts = partial_programs_invalid + incorrect_programs_valid + test_perfect_programs + remaining_programs

    result = simulate_logging_logic(attempts)

    print("REPRODUCTION RESULTS:")
    print(f"  Total attempts: {len(attempts)} (target: 64)")
    print(f"  Train partial count: {result['train_partial_total']} (target: appears as 10 in log)")
    print(f"  Train incorrect count: {result['train_incorrect_total']} (target: 47)")
    print(f"  Best train accuracy: {result['best_train_accuracy']:.1%} (target: 0.0%)")
    print(f"  Best test correct: {result['best_test_correct']} (should be True)")
    print()

    # Verify transductive counts
    valid_attempts = [att for att in attempts if att.get("outputs_valid", False)]
    trans_partial = sum(1 for att in valid_attempts
                       if att.get("train_accuracy", 0.0) > 0 and att.get("train_accuracy", 0.0) < 1.0
                       and att.get("is_transductive", False))
    trans_incorrect = sum(1 for att in valid_attempts
                         if att.get("train_accuracy", 0.0) == 0.0
                         and att.get("is_transductive", False))

    print("TRANSDUCTIVE COUNTS:")
    print(f"  Transductive partials: {trans_partial} (target: 3)")
    print(f"  Transductive incorrects: {trans_incorrect} (target: 4)")
    print()

    print("🎯 CONCLUSION:")
    if (result['train_partial_total'] == 0 and  # Partials don't show due to invalid outputs
        result['train_incorrect_total'] >= 47 and  # Incorrects show up
        result['best_train_accuracy'] > 0):  # Best comes from invalid partials

        print("✅ MYSTERY SOLVED!")
        print("The '10 train-partial' in the log refers to programs that WOULD be partial")
        print("if they had valid outputs, but they have outputs_valid=False.")
        print("These programs are:")
        print("- EXCLUDED from train categorization (hence 0 train-partial reported)")
        print("- INCLUDED in best selection (hence best > 0% train accuracy)")
        print("This creates the apparent contradiction.")
        print()
        print("The logging is technically correct but misleading!")
    else:
        print("❌ Could not reproduce the exact scenario")
        print("Need to investigate further...")


if __name__ == "__main__":
    analyze_observed_log_case()
    create_exact_reproduction()