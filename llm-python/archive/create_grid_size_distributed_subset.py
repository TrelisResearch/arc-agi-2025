import os
import json
import shutil
from pathlib import Path

SRC_ROOT = Path(__file__).parent.parent / 'data'
EVAL_SETS = [
    ('arc-agi-1', 'evaluation'),
    ('arc-agi-2', 'evaluation'),
]
SUBSET_NAME = 'grid_size_distributed_30'
SUBSET_ROOT = SRC_ROOT / 'subsets' / SUBSET_NAME
MANIFEST_PATH = SUBSET_ROOT / 'manifest.json'

N_SELECT = 30

os.makedirs(SUBSET_ROOT, exist_ok=True)

selected = {}
manifest = {}

def get_grid_size(task_path):
    with open(task_path, 'r') as f:
        task = json.load(f)
    # Use first train example
    input_grid = task['train'][0]['input']
    output_grid = task['train'][0]['output']
    input_size = len(input_grid) * len(input_grid[0])
    output_size = len(output_grid) * len(output_grid[0])
    return input_size + output_size

def select_evenly_distributed(files_and_sizes, n):
    files_and_sizes = sorted(files_and_sizes, key=lambda x: x[1])
    total = len(files_and_sizes)
    if total <= n:
        return files_and_sizes
    step = total / n
    indices = [int(i * step) for i in range(n)]
    # Ensure unique indices
    indices = sorted(set(indices))
    return [files_and_sizes[i] for i in indices]

for dataset, split in EVAL_SETS:
    eval_dir = SRC_ROOT / dataset / split
    files = [f for f in os.listdir(eval_dir) if f.endswith('.json')]
    files_and_sizes = []
    for fname in files:
        fpath = eval_dir / fname
        try:
            size = get_grid_size(fpath)
            files_and_sizes.append((fname, size))
        except Exception as e:
            print(f"Skipping {fname}: {e}")
    selected_files = select_evenly_distributed(files_and_sizes, N_SELECT)
    selected[dataset] = selected_files
    # Copy files and record manifest
    subset_dir = SUBSET_ROOT / dataset
    os.makedirs(subset_dir, exist_ok=True)
    manifest[dataset] = []
    for fname, size in selected_files:
        src = eval_dir / fname
        dst = subset_dir / fname
        shutil.copy2(src, dst)
        manifest[dataset].append({'filename': fname, 'grid_size': size})

with open(MANIFEST_PATH, 'w') as f:
    json.dump(manifest, f, indent=2)

# Also write .txt files for compatibility with TaskLoader/run_arc_tasks.py
for dataset in selected:
    subset_txt_path = SRC_ROOT / 'subsets' / dataset / f'{SUBSET_NAME}.txt'
    os.makedirs(subset_txt_path.parent, exist_ok=True)
    with open(subset_txt_path, 'w') as f:
        for fname, _ in selected[dataset]:
            f.write(f"{fname.replace('.json', '')}\n")

print(f"Subset created at {SUBSET_ROOT}")
print(f"Manifest written to {MANIFEST_PATH}") 