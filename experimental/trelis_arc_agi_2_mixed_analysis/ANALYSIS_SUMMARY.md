# Program Similarity Analysis

## Objective
Extract similar program pairs within the same ARC task for three accuracy categories:
1. **All Correct**: Perfect performance on both train and test
2. **Partially Correct**: Exactly one correct on train examples
3. **All Incorrect**: Zero correct on both train and test

## Method
- Used CodeRankEmbed (nomic-ai/CodeRankEmbed) for semantic code similarity
- Found pairs within same task_id with target similarities: 0.9, 0.95, 0.99
- Dataset: 20,540 programs from superking_aa2.parquet

## Results

### Dataset Distribution
- **All Correct**: 11,904 programs (58%)
- **All Incorrect**: 3,384 programs (16%)
- **Partially Correct**: 2,714 programs (13%)
- **Other**: 2,538 programs (12%)

### Selected Program Pairs

| Similarity Target | All Correct | Partially Correct | All Incorrect |
|------------------|-------------|-------------------|---------------|
| **0.9** | 0.899 (ec883f72) | 0.904 (d6542281) | 0.900 (fc4aaf52) |
| **0.95** | 0.950 (ec883f72) | 0.950 (d6542281) | 0.950 (fc4aaf52) |
| **0.99** | 0.974 (ec883f72) | 0.991 (d6542281) | 0.990 (fc4aaf52) |

## Output Files
- `selected_program_pairs_fast_sim0.9.parquet` (6 programs)
- `selected_program_pairs_fast_sim0.95.parquet` (6 programs)
- `selected_program_pairs_fast_sim0.99.parquet` (6 programs)
- Corresponding JSON summaries for each

## Key Findings
- Successfully identified highly similar program pairs within same tasks
- All similarity targets achieved within 0.02 tolerance
- Same tasks (ec883f72, d6542281, fc4aaf52) contained pairs across all similarity levels
- CodeRankEmbed embeddings effectively captured semantic code similarity