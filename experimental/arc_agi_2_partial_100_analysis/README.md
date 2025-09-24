# ARC-AGI-2 Partial 100 Dataset Deduplication Analysis

## Overview

This directory contains a comprehensive analysis of program similarity and deduplication potential within the ARC-AGI-2 Partial 100 dataset.

## Dataset Structure

- **863 unique tasks** (problems)
- **44,880 total programs** (solutions)
- **~52 programs per task** on average (ranging from 1-100 programs per task)

## Analysis Type

**Within-task deduplication**: For each task_id, we compared all programs solving that same task to identify similar solutions that could be removed while preserving solution diversity.

## Key Results

Based on analysis of 25 evenly distributed tasks (1,302 programs total):

| Similarity Threshold | Programs Remaining | Reduction |
|---------------------|-------------------|-----------|
| **99%** | **~35,229** | **21.5%** |
| **95%** | **~25,129** | **44.0%** |
| **90%** | **~14,375** | **68.0%** |

## Files

### Final Results
- `within_task_dedup_report.md` - Main analysis report
- `sim_0.99.json` - 99% threshold results
- `sim_0.95.json` - 95% threshold results
- `sim_0.9.json` - 90% threshold results
- `within_task_dedup_results.pkl` - Complete analysis data

### Scripts
- `correct_within_task_dedup.py` - Main analysis script
- `examine_dataset_structure.py` - Dataset exploration

### Intermediate Work
- `final_deduplication_analysis.md` - Cross-task analysis (incorrect approach)
- `realistic_analysis_report.md` - Cross-task analysis results
- Various `.pkl`, `.npy` files - Embeddings and similarity matrices

### Data
- `arc_agi_2_partial_100_data/` - Downloaded dataset
- `task_embeddings.pkl` - CodeRankEmbed embeddings for sample tasks

## Methodology

- **Embedding Model**: nomic-ai/CodeRankEmbed (768-dimensional)
- **Similarity Metric**: Cosine similarity
- **Deduplication Strategy**: Greedy removal (keep first occurrence)
- **Sample**: 25 tasks evenly distributed by program count
- **Processing**: 2 CPU cores maximum

## Confidence Level

**High** - Results are based on a representative sample of 2.9% of all tasks, covering the full range from high-program tasks (100 solutions) to low-program tasks (1 solution).

## Usage Recommendations

- **99% threshold**: Conservative deduplication, removes only near-identical programs
- **95% threshold**: Balanced approach, good reduction while preserving diversity
- **90% threshold**: Aggressive deduplication, use with caution as it removes many distinct solutions