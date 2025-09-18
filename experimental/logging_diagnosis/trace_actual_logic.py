#!/usr/bin/env python3
"""
Trace the exact logging logic from the actual code to understand where the "10 train-partial" comes from
"""

import sys
from typing import List, Dict, Any
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def trace_logging_with_debug(attempts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Simulate the exact logging logic with detailed debug tracing
    """
    print("📊 TRACING CATEGORIZATION LOGIC:")
    print("=" * 50)

    # Initialize counters
    train_perfect_total = train_perfect_trans = 0
    train_partial_total = train_partial_trans = 0
    train_incorrect_total = train_incorrect_trans = 0

    valid_attempts = []
    invalid_attempts = []

    # Categorization logic with detailed tracing
    for i, att in enumerate(attempts):
        outputs_valid = att.get("outputs_valid", False)
        is_trans = att.get("is_transductive", False)
        train_acc = att.get("train_accuracy", 0.0)
        test_correct = att.get("test_correct", False)

        print(f"Attempt {i+1:2d}: train={train_acc:4.1%}, test_correct={test_correct}, outputs_valid={outputs_valid}, trans={is_trans}")

        if not outputs_valid:
            invalid_attempts.append(att)
            print(f"            -> SKIPPED (invalid outputs)")
            continue

        valid_attempts.append(att)

        if train_acc == 1.0:
            train_perfect_total += 1
            if is_trans:
                train_perfect_trans += 1
            print(f"            -> TRAIN-PERFECT")
        elif train_acc > 0:
            train_partial_total += 1
            if is_trans:
                train_partial_trans += 1
            print(f"            -> TRAIN-PARTIAL")
        else:
            train_incorrect_total += 1
            if is_trans:
                train_incorrect_trans += 1
            print(f"            -> TRAIN-INCORRECT")

    print("=" * 50)
    print(f"VALID ATTEMPTS: {len(valid_attempts)}")
    print(f"INVALID ATTEMPTS: {len(invalid_attempts)}")
    print()

    # Best attempt selection logic
    print("🏆 BEST ATTEMPT SELECTION:")
    print("Considering ALL attempts (including invalid outputs)")
    print("Sort key: (test_correct, train_accuracy)")
    print()

    # Show all attempts with sort keys
    attempts_with_keys = []
    for i, att in enumerate(attempts):
        test_correct = att.get("test_correct", False)
        train_acc = att.get("train_accuracy", 0.0)
        sort_key = (test_correct, train_acc)
        attempts_with_keys.append((i+1, att, sort_key))
        print(f"Attempt {i+1:2d}: sort_key={sort_key} (test={test_correct}, train={train_acc:.1%})")

    # Find best
    best_attempt = max(attempts, key=lambda x: (x.get("test_correct", False), x.get("train_accuracy", 0.0)))
    best_index = attempts.index(best_attempt) + 1

    print(f"\n🎯 BEST: Attempt {best_index} with train={best_attempt.get('train_accuracy', 0.0):.1%}")
    print()

    return {
        "train_perfect_total": train_perfect_total,
        "train_partial_total": train_partial_total,
        "train_incorrect_total": train_incorrect_total,
        "train_partial_trans": train_partial_trans,
        "train_incorrect_trans": train_incorrect_trans,
        "best_attempt": best_attempt,
        "best_train_accuracy": best_attempt.get("train_accuracy", 0.0),
        "best_test_correct": best_attempt.get("test_correct", False),
        "valid_attempts": len(valid_attempts),
        "invalid_attempts": len(invalid_attempts),
    }


def mystery_case_interpretation():
    """
    Maybe the "10 train-partial" in the log is NOT from the categorization logic?
    Maybe it's from somewhere else entirely?
    """
    print("🕵️ ALTERNATIVE INTERPRETATION:")
    print("What if '10 train-partial' doesn't come from the categorization logic?")
    print()

    # Let's check: is there another place that counts partials?
    # Maybe it's counting from a different source entirely?

    print("Possible sources of '10 train-partial' count:")
    print("1. From categorization logic (what we've been testing)")
    print("2. From refinement dataset metadata")
    print("3. From a different counting mechanism")
    print("4. From pre-computed statistics")
    print()

    # Let's simulate what the actual observed case MIGHT look like
    print("Testing: What if the log message is misleading?")
    print("What if 'train-partial' refers to something else entirely?")
    print()


def test_most_likely_scenario():
    """
    Based on the analysis, test the most likely scenario
    """
    print("🎯 MOST LIKELY SCENARIO:")
    print("Test-correct programs with 0% train beat partial programs in 'best' selection")
    print()

    attempts = []

    # 10 partial programs with valid outputs
    for i in range(10):
        attempts.append({
            "train_accuracy": 0.1 + i * 0.05,  # 10% to 55%
            "test_correct": False,
            "outputs_valid": True,
            "is_transductive": i < 3,  # 3 transductive
        })

    # 47 incorrect programs with valid outputs
    for i in range(47):
        attempts.append({
            "train_accuracy": 0.0,
            "test_correct": False,
            "outputs_valid": True,
            "is_transductive": i < 4,  # 4 transductive
        })

    # 2 test-perfect programs with 0% train (lucky guesses)
    for i in range(2):
        attempts.append({
            "train_accuracy": 0.0,  # 0% train!
            "test_correct": True,   # But test correct!
            "outputs_valid": True,
            "is_transductive": False,
        })

    # 5 remaining attempts to reach 64
    for i in range(5):
        attempts.append({
            "train_accuracy": 0.0,
            "test_correct": False,
            "outputs_valid": True,
            "is_transductive": False,
        })

    result = trace_logging_with_debug(attempts)

    print("FINAL RESULTS:")
    print(f"Train partial: {result['train_partial_total']} (3 trans)")
    print(f"Train incorrect: {result['train_incorrect_total']} (4 trans)")
    print(f"Best train accuracy: {result['best_train_accuracy']:.1%}")
    print(f"Best test correct: {result['best_test_correct']}")
    print()

    if (result['train_partial_total'] == 10 and
        result['train_incorrect_total'] >= 47 and
        result['best_train_accuracy'] == 0.0):
        print("✅ THIS EXPLAINS IT!")
        print("10 train-partial programs exist and are counted,")
        print("but test-perfect programs with 0% train win the 'best' selection")
        print("because the sort key prioritizes test_correct over train_accuracy!")
    else:
        print("❌ Still doesn't match exactly...")


if __name__ == "__main__":
    print("🔬 DETAILED TRACING OF LOGGING LOGIC")
    print("=" * 60)
    print()

    mystery_case_interpretation()
    test_most_likely_scenario()