import sys
sys.path.append('/Users/ronanmcgovern/TR/arc-agi-2025')
from llm_python.datasets.io import read_soar_parquet
import pandas as pd

# Read the dataset
df = read_soar_parquet('/Users/ronanmcgovern/TR/arc-agi-2025/20250919_174321_Trelis_Qwen3-4B_ds-arc-agi-2-reasoning-5-c178_arc-prize-2025_evaluation.parquet')

# Pick any row (0-387)
row_idx = 5  # Change this to pick different rows
row = df.iloc[row_idx]

print(f'Row {row_idx}:')
print(f'  Task ID: {row["task_id"]}')
print(f'  Model: {row["model"]}')
print()
print('Reasoning:')
print(row['reasoning'] if pd.notna(row['reasoning']) else 'No reasoning available')