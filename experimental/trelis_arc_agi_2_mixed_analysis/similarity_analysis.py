import pandas as pd
import pyarrow.parquet as pq
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
print("Loading dataset...")
table = pq.read_table('superking_aa2.parquet')
df = table.to_pandas(strings_to_categorical=False, types_mapper=pd.ArrowDtype)

# Filter for task ff805c23
print("Filtering for task ff805c23...")
task_df = df[df['task_id'] == 'ff805c23'].copy()
print(f"Found {len(task_df)} programs for task ff805c23")

if len(task_df) == 0:
    print("No programs found for task ff805c23")
    exit()

# Get the code from each program
codes = task_df['code'].tolist()

print("Sample codes:")
for i, code in enumerate(codes[:3]):
    print(f"\nProgram {i+1}:")
    print(f"Row ID: {task_df.iloc[i]['row_id']}")
    print("Code:")
    print(code[:200] + "..." if len(code) > 200 else code)

# Load CodeRankEmbed model
print("\nLoading CodeRankEmbed model...")
model = SentenceTransformer("nomic-ai/CodeRankEmbed", trust_remote_code=True)

# Generate embeddings
print(f"Generating embeddings for {len(codes)} programs...")
code_embeddings = model.encode(codes, batch_size=8, show_progress_bar=True)

# Calculate pairwise cosine similarities
print("Calculating pairwise cosine similarities...")
similarity_matrix = cosine_similarity(code_embeddings)

# Find the pair with highest similarity (excluding self-similarity)
print("Finding most similar programs...")
np.fill_diagonal(similarity_matrix, -1)  # Remove self-similarities
max_similarity_idx = np.unravel_index(np.argmax(similarity_matrix), similarity_matrix.shape)
max_similarity = similarity_matrix[max_similarity_idx]

print(f"\nHighest similarity: {max_similarity:.4f}")
print(f"Between programs {max_similarity_idx[0]} and {max_similarity_idx[1]}")

# Display the two most similar programs
print("\n" + "="*80)
print("TWO MOST SIMILAR PROGRAMS")
print("="*80)

idx1, idx2 = max_similarity_idx
row1 = task_df.iloc[idx1]
row2 = task_df.iloc[idx2]

print(f"\nProgram 1 (Row ID: {row1['row_id']}):")
print("-" * 50)
print(row1['code'])

print(f"\nProgram 2 (Row ID: {row2['row_id']}):")
print("-" * 50)
print(row2['code'])

print(f"\nCosine Similarity: {max_similarity:.6f}")

# Show some additional metadata
print(f"\nProgram 1 metadata:")
print(f"  - Model: {row1['model']}")
print(f"  - Transductive: {row1['is_transductive']}")
print(f"  - Refined from: {row1['refined_from_id']}")

print(f"\nProgram 2 metadata:")
print(f"  - Model: {row2['model']}")
print(f"  - Transductive: {row2['is_transductive']}")
print(f"  - Refined from: {row2['refined_from_id']}")

# Save similarity matrix for further analysis
print("\nSaving similarity analysis...")
similarity_df = pd.DataFrame(similarity_matrix)
similarity_df.index = [f"Program_{i}" for i in range(len(codes))]
similarity_df.columns = [f"Program_{i}" for i in range(len(codes))]

# Also create a summary of top similarities
top_similarities = []
for i in range(len(codes)):
    for j in range(i+1, len(codes)):
        top_similarities.append({
            'program1_idx': i,
            'program2_idx': j,
            'program1_row_id': task_df.iloc[i]['row_id'],
            'program2_row_id': task_df.iloc[j]['row_id'],
            'similarity': similarity_matrix[i, j]
        })

top_similarities_df = pd.DataFrame(top_similarities)
top_similarities_df = top_similarities_df.sort_values('similarity', ascending=False)

print(f"\nTop 10 most similar program pairs:")
print(top_similarities_df.head(10)[['program1_row_id', 'program2_row_id', 'similarity']])

# Save results
similarity_df.to_csv('ff805c23_similarity_matrix.csv')
top_similarities_df.to_csv('ff805c23_top_similarities.csv', index=False)

print(f"\nResults saved:")
print(f"- Similarity matrix: ff805c23_similarity_matrix.csv")
print(f"- Top similarities: ff805c23_top_similarities.csv")