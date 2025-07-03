import os
import json

# Paths
LOGS_DIR = os.path.join(os.path.dirname(__file__), 'logs')
DATA_DIR = os.path.join(os.path.dirname(__file__), '../data/arc-agi-2')
MIDDLE_10_PATH = os.path.join(os.path.dirname(__file__), '../data/subsets/arc-agi-2/middle_10.txt')

# Read middle_10 task ids
def read_middle_10():
    with open(MIDDLE_10_PATH) as f:
        return [line.strip() for line in f if line.strip()]

# Check if a task is in evaluation or training split
def get_split(task_id):
    eval_path = os.path.join(DATA_DIR, 'evaluation', f'{task_id}.json')
    train_path = os.path.join(DATA_DIR, 'training', f'{task_id}.json')
    if os.path.exists(eval_path):
        return 'evaluation'
    elif os.path.exists(train_path):
        return 'training'
    else:
        return 'unknown'

# Find all o3 logs for a given task_id
def find_o3_logs_for_task(task_id):
    results = []
    for fname in os.listdir(LOGS_DIR):
        if not fname.endswith('.json') or 'summary' in fname:
            continue
        fpath = os.path.join(LOGS_DIR, fname)
        try:
            with open(fpath) as f:
                data = json.load(f)
            if (
                data.get('task_id') == task_id and
                data.get('model', '').startswith('o3')
            ):
                correct = None
                if 'score' in data and 'correct' in data['score']:
                    correct = data['score']['correct']
                results.append({
                    'log_file': fname,
                    'use_tools': data.get('use_tools'),
                    'tool_calls_count': data.get('tool_calls_count', 0),
                    'correct': correct
                })
        except Exception:
            continue
    return results

def main():
    middle_10 = read_middle_10()
    print(f'TaskID,Split,LogFile,UseTools,ToolCalls,Correct')
    for task_id in middle_10:
        split = get_split(task_id)
        logs = find_o3_logs_for_task(task_id)
        if not logs:
            print(f'{task_id},{split},NO_LOG,NA,NA,NA')
        for log in logs:
            print(f"{task_id},{split},{log['log_file']},{log['use_tools']},{log['tool_calls_count']},{log['correct']}")

if __name__ == '__main__':
    main() 