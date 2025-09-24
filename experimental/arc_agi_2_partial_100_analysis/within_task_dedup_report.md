# ARC-AGI-2 Partial 100: Within-Task Program Deduplication Analysis

## Overview
- **Analysis Type**: Deduplication within each task (comparing programs for the same task_id)
- **Total Dataset**: 44,880 programs across 863 tasks
- **Sample Analyzed**: 25 tasks with 1302 programs total

## Sample Task Details
- **5582e5ca**: 100 programs
- **ce22a75a**: 100 programs
- **f0afb749**: 100 programs
- **f8ff0b80**: 99 programs
- **7fe24cdd**: 97 programs
- **3bdb4ada**: 95 programs
- **22233c11**: 91 programs
- **f5b8619d**: 88 programs
- **42918530**: 82 programs
- **292dd178**: 75 programs
- **ec883f72**: 70 programs
- **a9f96cdd**: 63 programs
- **e3497940**: 57 programs
- **94414823**: 48 programs
- **f45f5ca7**: 40 programs
- **cdecee7f**: 32 programs
- **0d3d703e**: 22 programs
- **3345333e**: 15 programs
- **72207abc**: 9 programs
- **e6de6e8f**: 7 programs
- **fe9372f3**: 5 programs
- **d06dbe63**: 3 programs
- **1a244afd**: 2 programs
- **22a4bbc2**: 1 programs
- **50aad11f**: 1 programs

## Deduplication Results

### 99% Similarity Threshold

**Sample Results:**
- Original programs: 1,302
- Programs to remove: 280
- Programs remaining: 1,022
- Reduction: 21.5%

**Estimated Full Dataset:**
- Programs to remove: 9,651
- Programs remaining: 35,229
- Reduction: 21.5%

### 95% Similarity Threshold

**Sample Results:**
- Original programs: 1,302
- Programs to remove: 573
- Programs remaining: 729
- Reduction: 44.0%

**Estimated Full Dataset:**
- Programs to remove: 19,751
- Programs remaining: 25,129
- Reduction: 44.0%

### 90% Similarity Threshold

**Sample Results:**
- Original programs: 1,302
- Programs to remove: 885
- Programs remaining: 417
- Reduction: 68.0%

**Estimated Full Dataset:**
- Programs to remove: 30,505
- Programs remaining: 14,375
- Reduction: 68.0%

### 85% Similarity Threshold

**Sample Results:**
- Original programs: 1,302
- Programs to remove: 1,154
- Programs remaining: 148
- Reduction: 88.6%

**Estimated Full Dataset:**
- Programs to remove: 39,778
- Programs remaining: 5,102
- Reduction: 88.6%

### 80% Similarity Threshold

**Sample Results:**
- Original programs: 1,302
- Programs to remove: 1,239
- Programs remaining: 63
- Reduction: 95.2%

**Estimated Full Dataset:**
- Programs to remove: 42,708
- Programs remaining: 2,172
- Reduction: 95.2%

## Key Findings

1. **Within-task similarity**: Programs solving the same task show varying levels of similarity
2. **Deduplication potential**: Significant reduction possible while maintaining solution diversity
3. **Task-specific patterns**: Some tasks have many similar solutions, others have diverse approaches

## Methodology
- **Embedding Model**: nomic-ai/CodeRankEmbed
- **Similarity Metric**: Cosine similarity
- **Deduplication Strategy**: Greedy removal (keep first occurrence)
- **Processing**: 2 CPU cores maximum

## Limitations
- Results extrapolated from sample of 25 tasks
- Actual full dataset results may vary
- Some tasks might have different similarity patterns than sampled tasks
