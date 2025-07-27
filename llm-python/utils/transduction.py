import re
from typing import Dict, Tuple

def is_transduction_cheating(program: str, task_data: Dict, debug: bool = False) -> Tuple[bool, str]:
    """
    Detect if a program is cheating by hardcoding outputs (transduction).
    Returns (is_cheating, reason).
    """
    # Check 1: Very long lines (likely hardcoded values)
    lines = program.split('\n')
    for line_num, line in enumerate(lines, 1):
        if len(line) > 200:
            reason = f"Line {line_num} exceeds 200 characters (likely hardcoded)"
            if debug:
                print(f"    🚫 Transduction detected: {reason}")
                print(f"       Line: {line[:100]}...")
            return True, reason
    # Check 2: Hardcoded output values in code
    # Determine if task has 1x1 outputs (special case)
    flag_one = any((1, 1) == (len(example["output"]), len(example["output"][0]) if example["output"] else 0) 
                   for example in task_data.get("train", []))
    # Collect all outputs (training + test)
    all_outputs = []
    for example in task_data.get("train", []):
        if example.get("output"):
            all_outputs.append(example["output"])
    for test_example in task_data.get("test", []):
        if test_example.get("output"):
            all_outputs.append(test_example["output"])
    if not all_outputs:
        return False, ""
    # Create string representations of outputs
    if flag_one:
        def clean_string(s):
            return str(s).replace(' ', '')
    else:
        def clean_string(s):
            return str(s).replace(' ', '').replace('[', '').replace(']', '')
    output_strings = [clean_string(output) for output in all_outputs]
    cleaned_code = clean_string(program)
    for i, output_str in enumerate(output_strings):
        if len(output_str) > 2 and output_str in cleaned_code:
            reason = f"Output {i+1} hardcoded in program: {output_str[:50]}..."
            if debug:
                print(f"    🚫 Transduction detected: {reason}")
                code_idx = cleaned_code.find(output_str)
                context_start = max(0, code_idx - 30)
                context_end = min(len(cleaned_code), code_idx + len(output_str) + 30)
                context = cleaned_code[context_start:context_end]
                print(f"       Code context: ...{context}...")
            return True, reason
    return False, "" 