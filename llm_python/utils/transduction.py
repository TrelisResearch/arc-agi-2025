from typing import Tuple

from llm_python.utils.task_loader import TaskData

def detect_transduction(program: str, task_data: TaskData, debug: bool = False) -> Tuple[bool, str, bool, str]:
    """
    Detect if a program hardcodes training or test outputs (transduction).
    Returns (is_train_transductive, train_reason, is_test_transductive, test_reason).
    
    Train transduction: Program hardcodes training outputs - excluded from voting
    Test transduction: Program hardcodes test outputs - tagged for analysis
    """
    # Check 1: Very long lines (likely hardcoded values) - counts as train transduction
    lines = program.split('\n')
    for line_num, line in enumerate(lines, 1):
        if len(line) > 200:
            reason = f"Line {line_num} exceeds 200 characters (likely hardcoded)"
            if debug:
                print(f"    🚫 Long line detected: {reason}")
                print(f"       Line: {line[:100]}...")
            return True, reason, False, ""
    
    # Check 2: Hardcoded output values
    train_examples = task_data.get("train", [])
    test_examples = task_data.get("test", [])
    
    # Collect outputs
    train_outputs = [ex["output"] for ex in train_examples if ex.get("output")]
    test_outputs = [ex["output"] for ex in test_examples if ex.get("output")]
    
    if not train_outputs and not test_outputs:
        return False, "", False, ""
    
    # String cleaning function
    flag_one = any((1, 1) == (len(output), len(output[0]) if output else 0) for output in train_outputs)
    clean_string = (lambda s: str(s).replace(' ', '')) if flag_one else (lambda s: str(s).replace(' ', '').replace('[', '').replace(']', ''))
    
    cleaned_code = clean_string(program)
    
    # Helper function to check outputs
    def check_outputs(outputs, output_type):
        for i, output in enumerate(outputs):
            output_str = clean_string(output)
            if len(output_str) > 2 and output_str in cleaned_code:
                reason = f"{output_type} output {i+1} hardcoded in program: {output_str[:50]}..."
                if debug:
                    symbol = "🚫" if output_type == "Training" else "⚠️"
                    print(f"    {symbol} {output_type} transduction detected: {reason}")
                    code_idx = cleaned_code.find(output_str)
                    context_start = max(0, code_idx - 30)
                    context_end = min(len(cleaned_code), code_idx + len(output_str) + 30)
                    print(f"       Code context: ...{cleaned_code[context_start:context_end]}...")
                return True, reason
        return False, ""
    
    # Check train and test outputs
    train_transductive, train_reason = check_outputs(train_outputs, "Training")
    test_transductive, test_reason = check_outputs(test_outputs, "Test")
    
    return train_transductive, train_reason, test_transductive, test_reason 