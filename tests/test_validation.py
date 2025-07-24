#!/usr/bin/env python3

import json
import sys
sys.path.append('/Users/ronanmcgovern/TR/arc-agi-2025/tests')
from validate_training_data import validate_training_example

# Load line 771
with open('/Users/ronanmcgovern/TR/arc-agi-2025/llm-python/training_data/gemini_synth_50_random_split_1_training.jsonl', 'r') as f:
    for i, line in enumerate(f):
        if i == 770:  # Line 771 (0-indexed)
            example = json.loads(line)
            break

print("Validating line 771:")
print("="*50)
result = validate_training_example(example)
print(f"Status: {result['status']}")
print(f"Examples tested: {result['examples_tested']}")
print(f"Examples correct: {result['examples_correct']}")
print(f"Success rate: {result.get('success_rate', 0):.1%}")

if 'results' in result:
    print("\nDetailed results:")
    for res in result['results']:
        print(f"\nExample {res['example_num']}:")
        print(f"  Correct: {res['correct']}")
        print(f"  Expected shape: {res['expected_shape']}")
        print(f"  Predicted shape: {res['predicted_shape']}")
        if res['error']:
            print(f"  Error: {res['error']}")