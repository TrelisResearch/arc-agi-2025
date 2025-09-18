"""
metrics_utils.py   –   July 2025
Completely standalone.  Works with multiple test grids and the new
'all_test_correct' flag coming from compute_arc_metrics.

Simply replace your old file with this one and restart the kernel.
"""

from typing import Dict, List, Optional
from .voting_utils import (
    compute_weighted_majority_voting,
    compute_train_majority_voting,
)

# ---------------------------------------------------------------- helper utils
def _first(x):
    """Return first element if list/tuple, else x."""
    return x[0] if isinstance(x, (list, tuple)) and x else x



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
    # exclusive counters (non-transductive only)
    weighted_pass2_excl_total = train_majority_pass2_excl_total = all_test_correct_excl_total = 0
    # inclusive train counters (for display)
    all_train_correct_incl_total = min1_train_correct_incl_total = 0
    total_tasks = 0

    # response‑level counters
    total_responses = max_length_responses = 0
    timeout_responses = api_failure_responses = execution_timeout_responses = execution_error_responses = 0
    no_program_responses = 0
    transductive_responses = non_transductive_responses = 0

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
            if att.get("is_transductive", False):
                transductive_responses += 1
            else:
                non_transductive_responses += 1
            if att.get("hit_max_tokens", False):
                max_length_responses += 1
            if att.get("api_timeout", False):
                timeout_responses += 1
            if not att.get("api_success", True) and not att.get("api_timeout", False):
                api_failure_responses += 1
            if att.get("train_exec_timeouts", 0) > 0 or att.get("test_exec_timeout", False):
                execution_timeout_responses += 1
                # Debug logging
                if False:  # Set to True to debug
                    print(f"DEBUG: Found timeout - train_exec_timeouts={att.get('train_exec_timeouts', 0)}, test_exec_timeout={att.get('test_exec_timeout', False)}")
            # Count execution errors (excluding timeouts)
            if ((att.get("train_exec_errors", 0) > 0 or att.get("test_exec_error", False)) and
                not (att.get("train_exec_timeouts", 0) > 0 or att.get("test_exec_timeout", False))):
                execution_error_responses += 1
            # Count no program responses (code extraction failed)
            if not att.get("program_extracted", False):
                no_program_responses += 1

        # ---------- track transductive for stats ----------
        trans_count = sum(1 for att in attempts if att.get("is_transductive", False))
        if trans_count > 0:
            min1_transductive += 1
        
        # Include all attempts with valid outputs (both transductive and non-transductive)
        # Only exclude attempts with invalid outputs to ensure consistency with parquet filtering
        valid_attempts = [att for att in attempts if att.get("outputs_valid", False)]
        
        # For train metrics, exclude transductive attempts (they hardcode so train accuracy is meaningless)
        non_trans_valid = [att for att in valid_attempts if not att.get("is_transductive", False)]

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
        if not valid_attempts:
            continue  # nothing to score

        # ---------- ground truth ----------
        gt = [t["output"] for t in task["task_data"]["test"]]

        # ---------- weighted voting pass@2 ----------
        try:
            preds = compute_weighted_majority_voting(valid_attempts)   # Pass all valid attempts
            # Compare predictions as lists of test outputs
            if any(p == gt for p in preds if p is not None):
                weighted_pass2 += 1
        except Exception:
            pass

        # ---------- train‑majority voting pass@2 ----------
        try:
            preds = compute_train_majority_voting(valid_attempts)   # Pass all valid attempts
            # Compare predictions as lists of test outputs
            if any(p == gt for p in preds if p is not None):
                train_majority_pass2 += 1
        except Exception:
            pass

        # ---------- oracle all‑test (test-correct) ----------
        # Oracle should consider ALL valid attempts, including transductive ones
        # A transductive program that gets the right answer still counts as success
        if any(att.get("all_test_correct", False) for att in valid_attempts):
            all_test_correct += 1

        # ---------- exclusive metrics (non-transductive only) ----------
        # Calculate what the metrics would be with only non-transductive programs
        weighted_pass2_excl = train_majority_pass2_excl = all_test_correct_excl = 0
        
        # Weighted voting (non-trans only)
        try:
            preds = compute_weighted_majority_voting(non_trans_valid)
            if any(p == gt for p in preds if p is not None):
                weighted_pass2_excl = 1
        except Exception:
            pass

        # Train-majority voting (non-trans only)  
        try:
            preds = compute_train_majority_voting(non_trans_valid)
            if any(p == gt for p in preds if p is not None):
                train_majority_pass2_excl = 1
        except Exception:
            pass

        # Oracle (non-trans only)
        if any(att.get("all_test_correct", False) for att in non_trans_valid):
            all_test_correct_excl = 1

        # ---------- oracle train‑set metrics ----------
        # Helper function to avoid duplication
        def check_train_metrics(attempts_list):
            any_all = any_min1 = False
            for att in attempts_list:
                trs = att.get("train_results", [])
                if not trs:
                    continue
                if all(tr.get("correct", False) for tr in trs):
                    any_all = True
                if any(tr.get("correct", False) for tr in trs):
                    any_min1 = True
            return any_all, any_min1
        
        # Exclusive (non-transductive only)
        any_all_excl, any_min1_excl = check_train_metrics(non_trans_valid)
        if any_all_excl:
            all_train_correct += 1
        if any_min1_excl:
            min1_train_correct += 1
            
        # Inclusive (all valid attempts)
        any_all_incl, any_min1_incl = check_train_metrics(valid_attempts)
        all_train_correct_incl = 1 if any_all_incl else 0
        min1_train_correct_incl = 1 if any_min1_incl else 0
        
        # Accumulate metrics
        weighted_pass2_excl_total += weighted_pass2_excl
        train_majority_pass2_excl_total += train_majority_pass2_excl
        all_test_correct_excl_total += all_test_correct_excl
        all_train_correct_incl_total += all_train_correct_incl
        min1_train_correct_incl_total += min1_train_correct_incl

    # ------------------- assemble ---------------------
    return {
        # task‑level
        "weighted_pass2":         weighted_pass2,
        "train_majority_pass2":   train_majority_pass2,
        "all_test_correct":       all_test_correct,
        "all_train_correct":      all_train_correct,
        "min1_train_correct":     min1_train_correct,
        "all_train_correct_incl": all_train_correct_incl_total,
        "min1_train_correct_incl": min1_train_correct_incl_total,
        # exclusive metrics (non-transductive only)
        "weighted_pass2_excl":    weighted_pass2_excl_total,
        "train_majority_pass2_excl": train_majority_pass2_excl_total,
        "all_test_correct_excl":  all_test_correct_excl_total,
        "min1_transductive":      min1_transductive,
        "min1_code_success":      min1_code_success,
        "total":                  total_tasks,
        # response‑level
        "total_responses":        total_responses,
        "transductive_responses": transductive_responses,
        "non_transductive_responses": non_transductive_responses,
        "max_length_responses":   max_length_responses,
        "timeout_responses":      timeout_responses,
        "api_failure_responses":  api_failure_responses,
        "execution_timeout_responses": execution_timeout_responses,
        "execution_error_responses": execution_error_responses,
        "no_program_responses": no_program_responses,
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
        f"  All‑test correct (oracle, incl. trans): {metrics['all_test_correct']}/{total} "
        f"({metrics['all_test_correct']/total:.1%})",
        f"  Test correct (pass@2, weighted voting, incl. trans): "
        f"{metrics['weighted_pass2']}/{total} "
        f"({metrics['weighted_pass2']/total:.1%})",
        f"  Test correct (pass@2, train‑majority, incl. trans): "
        f"{metrics['train_majority_pass2']}/{total} "
        f"({metrics['train_majority_pass2']/total:.1%})",
        f"  All‑train correct (oracle, incl. trans): {metrics['all_train_correct_incl']}/{total} "
        f"({metrics['all_train_correct_incl']/total:.1%})",
        f"  All‑train correct (oracle, excl. trans): {metrics['all_train_correct']}/{total} "
        f"({metrics['all_train_correct']/total:.1%})",
        f"  Min‑1‑train correct (oracle, incl. trans): {metrics['min1_train_correct_incl']}/{total} "
        f"({metrics['min1_train_correct_incl']/total:.1%})",
        f"  Min‑1‑train correct (oracle, excl. trans): {metrics['min1_train_correct']}/{total} "
        f"({metrics['min1_train_correct']/total:.1%})",
        f"  Min‑1‑code success (incl. trans): {metrics['min1_code_success']}/{total} "
        f"({metrics['min1_code_success']/total:.1%})",
        f"  Min‑1‑transductive: {metrics['min1_transductive']}/{total} "
        f"({metrics['min1_transductive']/total:.1%})",
    ]
    if tr > 0:
        ln.extend([
            f"  Transductive responses: {metrics['transductive_responses']}/{tr} "
            f"({metrics['transductive_responses']/tr:.1%})",
            f"  Non‑transductive responses: {metrics['non_transductive_responses']}/{tr} "
            f"({metrics['non_transductive_responses']/tr:.1%})",
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
        "all_train_correct_incl": metrics["all_train_correct_incl"] / total,
        "min1_train_correct_incl": metrics["min1_train_correct_incl"] / total,
        # exclusive percentages
        "weighted_voting_pass2_excl": metrics["weighted_pass2_excl"] / total,
        "train_majority_pass2_excl": metrics["train_majority_pass2_excl"] / total,
        "all_test_correct_excl":  metrics["all_test_correct_excl"] / total,
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
            "execution_timeout_responses": metrics["execution_timeout_responses"] / total_responses,
            "execution_error_responses": metrics["execution_error_responses"] / total_responses,
            "no_program_responses": metrics["no_program_responses"] / total_responses,
        })
    else:
        pct.update({
            "max_length_responses": 0.0,
            "timeout_responses":    0.0,
            "api_failure_responses":0.0,
            "execution_timeout_responses": 0.0,
            "execution_error_responses": 0.0,
            "no_program_responses": 0.0,
        })
    return pct
