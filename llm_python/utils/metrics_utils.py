"""
metrics_utils.py   –   July 2025
Completely standalone.  Works with multiple test grids and the new
'all_test_correct' flag coming from compute_arc_metrics.

Simply replace your old file with this one and restart the kernel.
"""

from typing import Dict, List, Optional
from .voting_utils import (
    filter_non_transductive_attempts,
    compute_weighted_majority_voting,
    compute_train_majority_voting,
)

# ---------------------------------------------------------------- helper utils
def _first(x):
    """Return first element if list/tuple, else x."""
    return x[0] if isinstance(x, (list, tuple)) and x else x

def _to_tuple(x):
    """Make prediction hashable (handles multi‑grid)."""
    if x is None:
        return None
    return tuple(x) if isinstance(x, (list, tuple)) else (x,)


# ---------------------------------------------------------------- core metric
def calculate_task_metrics(
    results: List[Dict],
    *,
    upto_attempt: Optional[int] = None,
    max_tokens:   Optional[int] = None,
) -> Dict:
    # task‑level counters
    weighted_pass2 = train_majority_pass2 = 0
    all_test_correct = all_train_correct = 0
    min1_train_correct = min1_transductive = 0
    min1_code_success = 0
    total_tasks = 0

    # response‑level counters
    total_responses = max_length_responses = 0
    timeout_responses = api_failure_responses = 0

    # --------------- iterate through tasks -----------------
    for task in results:
        if not task:
            continue

        attempts = task["attempt_details"]
        if upto_attempt:
            attempts = attempts[:upto_attempt]

        # ---------- response‑level metrics ----------
        for att in attempts:
            total_responses += 1
            if att.get("hit_max_tokens", False):
                max_length_responses += 1
            if att.get("api_timeout", False):
                timeout_responses += 1
            if not att.get("api_success", True) and not att.get("api_timeout", False):
                api_failure_responses += 1

        # ---------- remove transductive ----------
        non_trans = filter_non_transductive_attempts(
            {"attempt_details": attempts, "task_data": task["task_data"]}
        )
        if len(attempts) > len(non_trans):
            min1_transductive += 1

        # ---------- min‑1 code success (extracted and executed without errors) ----------
        code_success = False
        for att in attempts:
            if (att.get("code_ran", False) and 
                not att.get("test_exec_error", False) and 
                not att.get("test_exec_timeout", False) and
                att.get("train_exec_errors", 0) == 0 and
                att.get("train_exec_timeouts", 0) == 0):
                code_success = True
                break
        if code_success:
            min1_code_success += 1

        total_tasks += 1
        if not non_trans:
            continue  # nothing to score

        # ---------- ground truth ----------
        gt = tuple(t["output"] for t in task["task_data"]["test"])

        # ---------- weighted voting pass@2 ----------
        try:
            preds = compute_weighted_majority_voting(non_trans)   # list
            # Handle both single test (raw grid) and multi-test (tuple of grids) cases
            normalized_preds = []
            for p in preds:
                if p is None:
                    continue
                # If it's already a tuple (multi-test), keep as is
                # If it's a single grid, convert to tuple to match gt format
                if len(gt) == 1:
                    # Single test case: p should be raw grid, wrap in tuple
                    normalized_preds.append(_to_tuple([p]))
                else:
                    # Multi test case: p should already be tuple
                    normalized_preds.append(_to_tuple(p))
            if any(p == gt for p in normalized_preds):
                weighted_pass2 += 1
        except Exception:
            pass

        # ---------- train‑majority voting pass@2 ----------
        try:
            preds = compute_train_majority_voting(non_trans)
            # Handle both single test (raw grid) and multi-test (tuple of grids) cases
            normalized_preds = []
            for p in preds:
                if p is None:
                    continue
                # If it's already a tuple (multi-test), keep as is
                # If it's a single grid, convert to tuple to match gt format
                if len(gt) == 1:
                    # Single test case: p should be raw grid, wrap in tuple
                    normalized_preds.append(_to_tuple([p]))
                else:
                    # Multi test case: p should already be tuple
                    normalized_preds.append(_to_tuple(p))
            if any(p == gt for p in normalized_preds):
                train_majority_pass2 += 1
        except Exception:
            pass

        # ---------- oracle all‑test ----------
        if any(att.get("all_test_correct", False) for att in non_trans):
            all_test_correct += 1

        # ---------- oracle train‑set metrics ----------
        any_all = any_min1 = False
        for att in non_trans:
            trs = att["train_results"]
            if not trs:
                continue
            if all(tr["correct"] for tr in trs):
                any_all = True
            if any(tr["correct"] for tr in trs):
                any_min1 = True
        if any_all:
            all_train_correct += 1
        if any_min1:
            min1_train_correct += 1

    # ------------------- assemble ---------------------
    return {
        # task‑level
        "weighted_pass2":         weighted_pass2,
        "train_majority_pass2":   train_majority_pass2,
        "all_test_correct":       all_test_correct,
        "all_train_correct":      all_train_correct,
        "min1_train_correct":     min1_train_correct,
        "min1_transductive":      min1_transductive,
        "min1_code_success":      min1_code_success,
        "total":                  total_tasks,
        # response‑level
        "total_responses":        total_responses,
        "max_length_responses":   max_length_responses,
        "timeout_responses":      timeout_responses,
        "api_failure_responses":  api_failure_responses,
    }


# ---------------------------------------------------------------- nice print
def format_metrics_display(metrics: Dict, layer: Optional[int] = None) -> str:
    """Readable, single‑string summary of raw counts."""
    total = metrics["total"]
    if total == 0:
        return "No valid tasks found."

    head = f"[Layer {layer}] Metrics:" if layer else "Metrics:"
    tr   = metrics["total_responses"]
    ln = [
        f"\n{head}",
        f"  All‑test correct (oracle): {metrics['all_test_correct']}/{total} "
        f"({metrics['all_test_correct']/total:.1%})",
        f"  Test correct (pass@2, weighted voting): "
        f"{metrics['weighted_pass2']}/{total} "
        f"({metrics['weighted_pass2']/total:.1%})",
        f"  Test correct (pass@2, train‑majority): "
        f"{metrics['train_majority_pass2']}/{total} "
        f"({metrics['train_majority_pass2']/total:.1%})",
        f"  All‑train correct (oracle): {metrics['all_train_correct']}/{total} "
        f"({metrics['all_train_correct']/total:.1%})",
        f"  Min‑1‑train correct (oracle): {metrics['min1_train_correct']}/{total} "
        f"({metrics['min1_train_correct']/total:.1%})",
        f"  Min‑1‑code success: {metrics['min1_code_success']}/{total} "
        f"({metrics['min1_code_success']/total:.1%})",
        f"  Min‑1‑transductive: {metrics['min1_transductive']}/{total} "
        f"({metrics['min1_transductive']/total:.1%})",
    ]
    if tr > 0:
        ln.extend([
            f"  Max‑length responses: {metrics['max_length_responses']}/{tr} "
            f"({metrics['max_length_responses']/tr:.1%})",
            f"  Timeout responses: {metrics['timeout_responses']}/{tr} "
            f"({metrics['timeout_responses']/tr:.1%})",
            f"  API failure responses: {metrics['api_failure_responses']}/{tr} "
            f"({metrics['api_failure_responses']/tr:.1%})",
        ])
    return "\n".join(ln)


# ---------------------------------------------------------------- percentages
def metrics_to_percentages(metrics: Dict) -> Dict:
    """Convert raw counts to percentages (except totals)."""
    total = metrics["total"]
    total_responses = metrics["total_responses"]
    if total == 0:
        return {k: 0.0 for k in metrics if k not in ["total", "total_responses"]}

    pct = {
        "weighted_voting_pass2":  metrics["weighted_pass2"] / total,
        "train_majority_pass2":   metrics["train_majority_pass2"] / total,
        "all_test_correct":       metrics["all_test_correct"] / total,
        "all_train_correct":      metrics["all_train_correct"] / total,
        "min1_train_correct":     metrics["min1_train_correct"] / total,
        "min1_transductive":      metrics["min1_transductive"] / total,
        "min1_code_success":      metrics["min1_code_success"] / total,
        "total_tasks":            total,
        "total_responses":        total_responses,
    }
    if total_responses > 0:
        pct.update({
            "max_length_responses": metrics["max_length_responses"] / total_responses,
            "timeout_responses":    metrics["timeout_responses"]    / total_responses,
            "api_failure_responses":metrics["api_failure_responses"]/ total_responses,
        })
    else:
        pct.update({
            "max_length_responses": 0.0,
            "timeout_responses":    0.0,
            "api_failure_responses":0.0,
        })
    return pct
