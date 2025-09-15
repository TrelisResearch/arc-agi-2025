import pandas as pd
import json
from collections import defaultdict

# Load the Trelis dataset
print("Loading Trelis dataset...")
try:
    df = pd.read_parquet('superking_aa2.parquet')
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
except Exception as e:
    print(f"Error loading with pandas: {e}")
    # Try with pyarrow directly
    import pyarrow.parquet as pq
    table = pq.read_table('superking_aa2.parquet')
    print(f"PyArrow table loaded. Schema: {table.schema}")

    # Convert to pandas with strings_to_categorical=False
    df = table.to_pandas(strings_to_categorical=False, types_mapper=pd.ArrowDtype)
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

# Load ARC-Prize-2025 training tasks
print("\nLoading ARC-Prize-2025 training tasks...")
with open('../../data/arc-prize-2025/arc-agi_training_challenges.json', 'r') as f:
    arc_training_tasks = json.load(f)

arc_training_task_ids = set(arc_training_tasks.keys())
print(f"Found {len(arc_training_task_ids)} training task IDs")

# Examine the dataset structure first
print("\nDataset sample:")
print(df.head())
print("\nDataset info:")
print(df.info())

# Check unique values in relevant columns
if 'refined_from_id' in df.columns:
    print(f"\nRefined_from_id non-null count: {df['refined_from_id'].notna().sum()}")
else:
    print(f"\nAvailable columns: {list(df.columns)}")
    # Let's see if there's a similar column
    refined_cols = [col for col in df.columns if 'refin' in col.lower()]
    print(f"Columns containing 'refin': {refined_cols}")

# Check if there's a task_id or similar column
task_id_cols = [col for col in df.columns if any(x in col.lower() for x in ['task', 'id'])]
print(f"Columns containing task/id: {task_id_cols}")

# Check for predicted_train_output_original column
if 'predicted_train_output_original' in df.columns:
    print(f"\nPredicted_train_output_original info:")
    print(df['predicted_train_output_original'].dtype)
    print(f"Non-null values: {df['predicted_train_output_original'].notna().sum()}")

# Let's see what the data looks like
print("\nFirst few rows of key columns:")
key_cols = ['refined_from_id'] if 'refined_from_id' in df.columns else []
if 'predicted_train_output_original' in df.columns:
    key_cols.append('predicted_train_output_original')
if task_id_cols:
    key_cols.extend(task_id_cols[:2])  # Add first couple task id columns

if key_cols:
    print(df[key_cols].head(10))