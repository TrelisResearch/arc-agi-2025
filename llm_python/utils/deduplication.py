"""
Program deduplication utilities.

Provides functions to deduplicate programs by normalizing whitespace and doing exact matching.
"""

import pandas as pd
from typing import Dict, List, Tuple
import re
from collections import defaultdict


def normalize_code(code: str) -> str:
    """
    Normalize code by removing newlines, tabs, and extra whitespace for deduplication.
    
    Args:
        code: Raw code string
        
    Returns:
        Normalized code string with consistent whitespace
    """
    if not code or pd.isna(code):
        return ""
    
    # Remove newlines and tabs, collapse multiple spaces to single space
    normalized = re.sub(r'\s+', ' ', code.strip())
    return normalized


def deduplicate_programs(df: pd.DataFrame, code_column: str = 'code') -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Deduplicate programs by normalizing whitespace and doing exact matching.
    Keeps the first occurrence of each unique normalized program.
    
    Args:
        df: DataFrame containing programs
        code_column: Name of the column containing code
        
    Returns:
        Tuple of (deduplicated_df, dedup_stats)
        dedup_stats contains:
            - 'original_count': Original number of programs
            - 'deduplicated_count': Number after deduplication
            - 'duplicates_removed': Number of duplicates removed
            - 'programs_with_duplicates': Number of unique programs that had at least one duplicate
    """
    print(f"🔍 Deduplicating programs...")
    
    original_count = len(df)
    
    # Add normalized code column
    df_with_normalized = df.copy()
    df_with_normalized['_normalized_code'] = df_with_normalized[code_column].apply(normalize_code)
    
    # Track which normalized codes appear multiple times
    normalized_counts = df_with_normalized['_normalized_code'].value_counts()
    programs_with_duplicates = (normalized_counts > 1).sum()
    
    # Keep first occurrence of each normalized code
    deduplicated_df = df_with_normalized.drop_duplicates(subset=['_normalized_code'], keep='first')
    
    # Remove the temporary normalized code column
    deduplicated_df = deduplicated_df.drop('_normalized_code', axis=1)
    
    deduplicated_count = len(deduplicated_df)
    duplicates_removed = original_count - deduplicated_count
    
    dedup_stats = {
        'original_count': original_count,
        'deduplicated_count': deduplicated_count,
        'duplicates_removed': duplicates_removed,
        'programs_with_duplicates': programs_with_duplicates
    }
    
    print(f"📊 Deduplication stats:")
    print(f"  Original programs: {original_count}")
    print(f"  After deduplication: {deduplicated_count}")
    print(f"  Duplicates removed: {duplicates_removed} ({duplicates_removed/original_count*100:.1f}%)")
    print(f"  Programs with at least one duplicate: {programs_with_duplicates}")
    
    return deduplicated_df, dedup_stats


def deduplicate_by_task(df: pd.DataFrame, task_column: str = 'task_id', code_column: str = 'code') -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Deduplicate programs within each task separately.
    
    Args:
        df: DataFrame containing programs
        task_column: Name of the column containing task IDs
        code_column: Name of the column containing code
        
    Returns:
        Tuple of (deduplicated_df, dedup_stats)
    """
    print(f"🔍 Deduplicating programs by task...")
    
    original_count = len(df)
    
    # Add normalized code column for all data
    df_with_normalized = df.copy()
    df_with_normalized['_normalized_code'] = df_with_normalized[code_column].apply(normalize_code)
    
    # Group by task and deduplicate within each group (without individual task stats)
    deduplicated_groups = []
    
    for task_id in df_with_normalized[task_column].unique():
        task_df = df_with_normalized[df_with_normalized[task_column] == task_id].copy()
        # Keep first occurrence of each normalized code within the task
        task_deduplicated = task_df.drop_duplicates(subset=['_normalized_code'], keep='first')
        deduplicated_groups.append(task_deduplicated)
    
    # Combine all deduplicated groups
    deduplicated_df = pd.concat(deduplicated_groups, ignore_index=True)
    
    # Remove the temporary normalized code column
    deduplicated_df = deduplicated_df.drop('_normalized_code', axis=1)
    
    deduplicated_count = len(deduplicated_df)
    total_duplicates_removed = original_count - deduplicated_count
    
    # Count programs with duplicates globally
    normalized_counts = df_with_normalized.groupby([task_column, '_normalized_code']).size()
    programs_with_duplicates = (normalized_counts > 1).sum()
    
    dedup_stats = {
        'original_count': original_count,
        'deduplicated_count': deduplicated_count,
        'duplicates_removed': total_duplicates_removed,
        'programs_with_duplicates': programs_with_duplicates
    }
    
    print(f"📊 Global deduplication stats:")
    print(f"  Original programs: {original_count}")
    print(f"  After deduplication: {deduplicated_count}")
    print(f"  Duplicates removed: {total_duplicates_removed} ({total_duplicates_removed/original_count*100:.1f}%)")
    print(f"  Programs with at least one duplicate: {programs_with_duplicates}")
    
    return deduplicated_df, dedup_stats