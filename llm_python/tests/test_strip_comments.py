#!/usr/bin/env python3

import os
import json
import glob
import io
import tokenize
import ast
import re
from pathlib import Path

def find_latest_log_file():
    """Find the most recent log file in the logs directory"""
    log_files = glob.glob("../logs/*.json")
    log_files = [f for f in log_files if 'summary' not in f]
    
    if not log_files:
        return None
    
    # Sort by modification time, get the most recent
    log_files.sort(key=os.path.getmtime, reverse=True)
    return log_files[0]

def extract_program_from_log(log_path):
    """Extract Python program from a log file"""
    with open(log_path, 'r') as f:
        log_data = json.load(f)
    
    # Try different log structures
    program = log_data.get('program')
    if program:
        return program
    
    # Check multiturn data
    multiturn_data = log_data.get('multiturn_data', {})
    turn_details = multiturn_data.get('turn_details', [])
    if turn_details:
        return turn_details[0].get('program', '')
    
    # Check independent attempts
    independent_data = log_data.get('independent_attempts_data', {})
    attempt_details = independent_data.get('attempt_details', [])
    if attempt_details:
        return attempt_details[0].get('program', '')
    
    return None

def strip_comments_basic(source_code):
    """
    Basic comment stripping - removes # comments and preserves structure.
    Uses tokenize module for reliable parsing.
    """
    if not source_code.strip():
        return source_code
    
    try:
        source_io = io.StringIO(source_code)
        tokens = []
        prev_toktype = tokenize.INDENT
        last_lineno = -1
        last_col = 0
        
        for tok in tokenize.generate_tokens(source_io.readline):
            token_type = tok[0]
            token_string = tok[1]
            start_line, start_col = tok[2]
            end_line, end_col = tok[3]
            
            # Skip comments
            if token_type == tokenize.COMMENT:
                continue
                
            # Skip docstrings (STRING tokens that are statements, not expressions)
            if token_type == tokenize.STRING:
                if prev_toktype in [tokenize.INDENT, tokenize.DEDENT, tokenize.NEWLINE, tokenize.NL, tokenize.COLON]:
                    continue
            
            # Handle newlines properly - preserve structure
            if start_line > last_lineno:
                last_col = 0
            if start_col > last_col:
                tokens.append(" " * (start_col - last_col))
            
            tokens.append(token_string)
            
            prev_toktype = token_type
            last_col = end_col
            last_lineno = end_line
        
        return ''.join(tokens)
        
    except Exception as e:
        print(f"Error in basic comment stripping: {e}")
        return source_code

def strip_comments_aggressive(source_code):
    """
    Aggressive comment stripping - removes comments and cleans up whitespace.
    This version removes comment lines entirely and normalizes whitespace.
    """
    if not source_code.strip():
        return source_code
    
    try:
        # First pass: identify which lines are comment-only or empty
        lines = source_code.split('\n')
        cleaned_lines = []
        
        for line in lines:
            stripped = line.strip()
            # Skip empty lines and comment-only lines
            if not stripped or stripped.startswith('#'):
                continue
            
            # Remove inline comments but preserve the code part
            if '#' in line:
                # Find the # that's not inside a string literal
                in_string = False
                quote_char = None
                for i, char in enumerate(line):
                    if char in ['"', "'"] and (i == 0 or line[i-1] != '\\'):
                        if not in_string:
                            in_string = True
                            quote_char = char
                        elif char == quote_char:
                            in_string = False
                            quote_char = None
                    elif char == '#' and not in_string:
                        line = line[:i].rstrip()
                        break
            
            cleaned_lines.append(line)
        
        # Join lines and normalize whitespace
        result = '\n'.join(cleaned_lines)
        
        # Remove excessive blank lines (more than 2 consecutive)
        result = re.sub(r'\n\s*\n\s*\n+', '\n\n', result)
        
        return result.strip()
        
    except Exception as e:
        print(f"Error in aggressive comment stripping: {e}")
        return source_code

def strip_comments_minimal(source_code):
    """
    Minimal comment stripping - only removes comments, preserves all formatting.
    Uses AST to ensure we don't break the code structure.
    """
    if not source_code.strip():
        return source_code
    
    try:
        # Parse and recompile using AST to remove comments naturally
        tree = ast.parse(source_code)
        # Convert back to source code - this strips comments automatically
        import astor
        return astor.to_source(tree)
    except ImportError:
        # Fallback to regex-based approach if astor not available
        lines = source_code.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Simple regex to remove comments (not perfect but works for most cases)
            cleaned_line = re.sub(r'#.*$', '', line).rstrip()
            cleaned_lines.append(cleaned_line)
        
        return '\n'.join(cleaned_lines)
    except Exception as e:
        print(f"Error in minimal comment stripping: {e}")
        return source_code

def count_lines(text):
    """Count lines in text"""
    return len(text.strip().split('\n')) if text.strip() else 0

def count_comment_lines(source_code):
    """Count the number of comment lines in source code"""
    if not source_code.strip():
        return 0
    
    lines = source_code.split('\n')
    comment_lines = 0
    
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('#'):
            comment_lines += 1
        elif '#' in line:
            # Inline comment
            comment_lines += 0.5  # Count as half a line
    
    return int(comment_lines)

def analyze_comments(source_code):
    """Provide detailed comment analysis"""
    if not source_code.strip():
        return {}
    
    lines = source_code.split('\n')
    comment_only_lines = 0
    inline_comment_lines = 0
    total_comment_chars = 0
    
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('#'):
            comment_only_lines += 1
            total_comment_chars += len(stripped)
        elif '#' in line:
            inline_comment_lines += 1
            # Count characters after #
            comment_start = line.find('#')
            total_comment_chars += len(line[comment_start:])
    
    return {
        'comment_only_lines': comment_only_lines,
        'inline_comment_lines': inline_comment_lines,
        'total_comment_chars': total_comment_chars,
        'total_lines': len(lines)
    }

def main():
    # Find the latest log file
    latest_log = find_latest_log_file()
    if not latest_log:
        print("No log files found in ../logs/")
        return
    
    print(f"Processing latest log file: {Path(latest_log).name}")
    
    # Extract program from log
    program = extract_program_from_log(latest_log)
    if not program:
        print("No program found in the log file")
        return
    
    task_id = None
    try:
        with open(latest_log, 'r') as f:
            log_data = json.load(f)
            task_id = log_data.get('task_id', 'unknown')
    except:
        pass
    
    print(f"Task ID: {task_id}")
    print(f"Original program: {len(program)} chars, {count_lines(program)} lines")
    
    # Analyze comments
    comment_stats = analyze_comments(program)
    print(f"Comment analysis:")
    print(f"  - Comment-only lines: {comment_stats['comment_only_lines']}")
    print(f"  - Lines with inline comments: {comment_stats['inline_comment_lines']}")
    print(f"  - Total comment characters: {comment_stats['total_comment_chars']}")
    
    # Test different stripping methods
    methods = [
        ("Basic", strip_comments_basic),
        ("Aggressive", strip_comments_aggressive),
        ("Minimal", strip_comments_minimal)
    ]
    
    print("\n" + "="*80)
    print("COMMENT STRIPPING COMPARISON")
    print("="*80)
    
    for method_name, method_func in methods:
        stripped = method_func(program)
        char_reduction = len(program) - len(stripped)
        line_reduction = count_lines(program) - count_lines(stripped)
        
        print(f"\n{method_name} method:")
        print(f"  Result: {len(stripped)} chars ({char_reduction} saved, {char_reduction/len(program)*100:.1f}%), {count_lines(stripped)} lines ({line_reduction} saved)")
        
        # Test compilation
        try:
            compile(stripped, f'<{method_name.lower()}>', 'exec')
            print(f"  ✅ Compiles successfully")
        except SyntaxError as e:
            print(f"  ❌ Syntax error: {e}")
        except Exception as e:
            print(f"  ⚠️  Compilation error: {e}")
    
    # Show the best method's output
    print("\n" + "="*80)
    print("AGGRESSIVE STRIPPING RESULT (recommended):")
    print("="*80)
    aggressive_result = strip_comments_aggressive(program)
    print(aggressive_result[:800] + ("..." if len(aggressive_result) > 800 else ""))

if __name__ == "__main__":
    main() 