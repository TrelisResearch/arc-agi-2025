#!/usr/bin/env python3

import json
import re
import sys
import subprocess
import tempfile
import os

sys.path.append('/Users/ronanmcgovern/TR/arc-agi-2025/tests')
from validate_training_data import extract_examples_from_user_message, extract_program_from_assistant_message, execute_program_on_input

# Test lines 771 and 775
test_lines = [770, 774]  # 0-indexed

for line_idx in test_lines:
    print(f"\n{'='*60}")
    print(f"Testing line {line_idx + 1}:")
    print('='*60)
    
    # Load the example
    with open('/Users/ronanmcgovern/TR/arc-agi-2025/llm_python/training_data/gemini_synth_50_random_split_1_training.jsonl', 'r') as f:
        for i, line in enumerate(f):
            if i == line_idx:
                example = json.loads(line)
                break
    
    # Extract program and examples
    user_msg = example['messages'][1]['content']
    assistant_msg = example['messages'][2]['content']
    
    examples = extract_examples_from_user_message(user_msg)
    program = extract_program_from_assistant_message(assistant_msg)
    
    print(f"\nProgram:")
    print(program)
    
    print(f"\n\nTesting on examples:")
    for ex in examples[:3]:  # Test first 3 examples
        print(f"\nExample {ex['example_num']}:")
        print(f"  Input shape: {len(ex['input'])}x{len(ex['input'][0]) if ex['input'] else 0}")
        print(f"  Expected output shape: {len(ex['output'])}x{len(ex['output'][0]) if ex['output'] else 0}")
        
        # Execute in both contexts
        # 1. Direct execution
        namespace = {}
        exec(program, namespace)
        transform = namespace['transform']
        try:
            direct_result = transform(ex['input'])
            direct_shape = (len(direct_result), len(direct_result[0]) if direct_result and direct_result[0] else 0)
            print(f"  Direct execution shape: {direct_shape}")
        except Exception as e:
            print(f"  Direct execution error: {e}")
            direct_result = None
        
        # 2. Subprocess execution (like validation)
        predicted_output, error, timed_out = execute_program_on_input(program, ex['input'])
        if predicted_output is not None:
            subprocess_shape = (len(predicted_output), len(predicted_output[0]) if predicted_output and predicted_output[0] else 0)
            print(f"  Subprocess execution shape: {subprocess_shape}")
        else:
            print(f"  Subprocess execution error: {error}")
        
        # Compare
        if direct_result is not None and predicted_output is not None:
            if direct_result == predicted_output:
                print(f"  ✓ Results match")
            else:
                print(f"  ✗ Results differ!")
                print(f"    Direct: {direct_result}")
                print(f"    Subprocess: {predicted_output}")