import json
import os
from statistics import median
from pathlib import Path

def analyze_jsonl_file(file_path):
    """Analyze character counts for system, user, and assistant messages in a JSONL file."""
    system_chars = []
    user_chars = []
    assistant_chars = []
    total_chars = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                messages = data.get('messages', [])
                
                row_system_chars = 0
                row_user_chars = 0
                row_assistant_chars = 0
                
                for message in messages:
                    role = message.get('role', '')
                    content = message.get('content', '')
                    char_count = len(content)
                    
                    if role == 'system':
                        row_system_chars += char_count
                    elif role == 'user':
                        row_user_chars += char_count
                    elif role == 'assistant':
                        row_assistant_chars += char_count
                
                system_chars.append(row_system_chars)
                user_chars.append(row_user_chars)
                assistant_chars.append(row_assistant_chars)
                total_chars.append(row_system_chars + row_user_chars + row_assistant_chars)
                
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num} in {file_path}: {e}")
                continue
    
    return {
        'system': system_chars,
        'user': user_chars,
        'assistant': assistant_chars,
        'total': total_chars
    }

def calculate_stats(values):
    """Calculate median, min, max for a list of values."""
    if not values:
        return {'median': 0, 'min': 0, 'max': 0}
    
    return {
        'median': median(values),
        'min': min(values),
        'max': max(values),
        'count': len(values)
    }

def main():
    training_data_dir = Path("o3-tools/training_data")
    
    if not training_data_dir.exists():
        print(f"Training data directory not found: {training_data_dir}")
        return
    
    # Find all JSONL files
    jsonl_files = list(training_data_dir.glob("*.jsonl"))
    
    if not jsonl_files:
        print("No JSONL files found in training_data directory")
        return
    
    print("=" * 80)
    print("TRAINING DATA CHARACTER COUNT ANALYSIS")
    print("=" * 80)
    
    for file_path in sorted(jsonl_files):
        print(f"\nFile: {file_path.name}")
        print("-" * 60)
        
        char_data = analyze_jsonl_file(file_path)
        
        for message_type in ['system', 'user', 'assistant', 'total']:
            stats = calculate_stats(char_data[message_type])
            print(f"{message_type.capitalize():>10}: "
                  f"Median={stats['median']:>6.0f}, "
                  f"Min={stats['min']:>6}, "
                  f"Max={stats['max']:>8}, "
                  f"Rows={stats['count']:>4}")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main() 