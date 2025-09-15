import pandas as pd
import pyarrow.parquet as pq
import numpy as np

# Load the similarity results
print("Loading similarity results...")
top_similarities_df = pd.read_csv('ff805c23_top_similarities.csv')

# Find the pair with lowest similarity
lowest_similarity = top_similarities_df.iloc[-1]
print(f"Lowest similarity: {lowest_similarity['similarity']:.6f}")
print(f"Between programs with row IDs: {lowest_similarity['program1_row_id']} and {lowest_similarity['program2_row_id']}")

# Load the original dataset to get the code
table = pq.read_table('superking_aa2.parquet')
df = table.to_pandas(strings_to_categorical=False, types_mapper=pd.ArrowDtype)
task_df = df[df['task_id'] == 'ff805c23'].copy()

# Find the two programs with lowest similarity
row_id1 = lowest_similarity['program1_row_id']
row_id2 = lowest_similarity['program2_row_id']

program1 = task_df[task_df['row_id'] == row_id1].iloc[0]
program2 = task_df[task_df['row_id'] == row_id2].iloc[0]

print("\n" + "="*80)
print("TWO LEAST SIMILAR PROGRAMS")
print("="*80)

print(f"\nProgram 1 (Row ID: {row_id1}):")
print("-" * 50)
print(program1['code'])

print(f"\nProgram 2 (Row ID: {row_id2}):")
print("-" * 50)
print(program2['code'])

print(f"\nCosine Similarity: {lowest_similarity['similarity']:.6f}")

# Show metadata
print(f"\nProgram 1 metadata:")
print(f"  - Model: {program1['model']}")
print(f"  - Transductive: {program1['is_transductive']}")
print(f"  - Refined from: {program1['refined_from_id']}")

print(f"\nProgram 2 metadata:")
print(f"  - Model: {program2['model']}")
print(f"  - Transductive: {program2['is_transductive']}")
print(f"  - Refined from: {program2['refined_from_id']}")

# Also show a few more low similarity pairs
print(f"\nBottom 5 similarity pairs:")
bottom_5 = top_similarities_df.tail(5)[['program1_row_id', 'program2_row_id', 'similarity']]
for idx, row in bottom_5.iterrows():
    print(f"  {row['similarity']:.6f}: {row['program1_row_id'][:8]}... vs {row['program2_row_id'][:8]}...")