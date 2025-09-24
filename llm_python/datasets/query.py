from typing import List, Optional

from llm_python.utils.task_loader import get_task_loader


def filter_soar_df(
    df,
    include_subset=None,
    exclude_subset=None,
    all_train_correct=None,
    all_test_correct=None,
    any_train_correct=None,
    exclude_transductive=True,
    max_rows=None,
):
    """
    Load a SOAR-format parquet file and filter rows based on subset membership and correctness.

    Args:
        parquet_path (str): Path to the parquet file.
        include_subset (str, optional): Subset name to include (from task_loader).
        exclude_subset (str, optional): Subset name to exclude (from task_loader).
        all_train_correct (bool, optional): If True, only rows where all train inputs are correct.
        all_test_correct (bool, optional): If True, only rows where all test inputs are correct.
        any_train_correct (bool, optional): If True, only rows where any train input is correct.
        exclude_transductive (bool, optional): If True, exclude transductive programs.
        max_rows (int, optional): Limit number of rows returned.

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    task_loader = get_task_loader()

    # Subset filtering
    if include_subset:
        allowed_ids = set(
            [id for id, _ in task_loader.get_subset_tasks(include_subset)]
        )
        df = df[df["task_id"].isin(allowed_ids)]
    if exclude_subset:
        excluded_ids = set(
            [id for id, _ in task_loader.get_subset_tasks(exclude_subset)]
        )
        df = df[~df["task_id"].isin(excluded_ids)]

    # Correctness filters
    if all_train_correct is not None:
        df = df[df["correct_train_input"].apply(lambda x: all(x) == all_train_correct)]
    if all_test_correct is not None:
        df = df[df["correct_test_input"].apply(lambda x: all(x) == all_test_correct)]
    if any_train_correct is not None:
        df = df[df["correct_train_input"].apply(lambda x: any(x) == any_train_correct)]

    # Exclude transductive
    if exclude_transductive:
        df = df[~df["is_transductive"]]

    # Limit rows
    if max_rows is not None:
        df = df.head(max_rows)

    return df


def sample_by_task(
    df,
    sort_keys: List[str],
    sort_ascending: Optional[List[bool]] = None,
    task_limit=10,
):
    """
    Sample up to `task_limit` rows per task, sorting by specified keys.
    """
    if not sort_keys:
        raise ValueError("sort_keys must be provided")

    df = df.copy()
    df["program_length"] = df["program"].str.len()
    return (
        df.sort_values(
            by=sort_keys,
            ascending=sort_ascending,
        )
        .groupby("task_id")
        .head(task_limit)
    )
