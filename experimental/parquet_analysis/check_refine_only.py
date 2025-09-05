#!/usr/bin/env python3

import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from llm_python.datasets.io import read_soar_parquet

# Check the refine-only file
parquet_path = Path(__file__).parent.parent.parent / "llm_python/datasets/inference/20250903_094643_openai_gpt-oss-120b_arc-prize-2025_training-hard.parquet"

df = read_soar_parquet(parquet_path)
print(f"File: 20250903_094643_openai_gpt-oss-120b_arc-prize-2025_training-hard.parquet")
print(f"Total rows: {len(df)}")
print(f"Unique tasks: {df['task_id'].nunique()}")
print(f"Task IDs: {df['task_id'].unique()}")
print(f"\nNon-transductive: {len(df[df['is_transductive'] == False])}")
print(f"Transductive: {len(df[df['is_transductive'] == True])}")

# Check if there are any correct programs
def check_all_correct(x):
    if x is None or (hasattr(x, '__len__') and len(x) == 0):
        return False
    if hasattr(x, 'to_pylist'):
        x = x.to_pylist()
    return all(x)

df['all_train'] = df['correct_train_input'].apply(check_all_correct)
df['all_test'] = df['correct_test_input'].apply(check_all_correct)
df['all_correct'] = df['all_train'] & df['all_test']

print(f"\nAll-correct programs: {df['all_correct'].sum()}")
print(f"All-train correct: {df['all_train'].sum()}")
print(f"All-test correct: {df['all_test'].sum()}")