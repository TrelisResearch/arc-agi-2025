import json
import argparse
import os

def is_rectangular(grid):
    if not isinstance(grid, list) or not grid:
        return False
    row_len = len(grid[0])
    if row_len == 0:
        return False
    if len(grid) > 30 or row_len > 30:
        return False
    for row in grid:
        if not isinstance(row, list) or len(row) != row_len:
            return False
        for val in row:
            if not isinstance(val, int) or val < 0 or val > 9:
                return False
    return True

def validate_submission(submission_path, challenges_path):
    errors = []

    # --- Load files ---
    try:
        with open(submission_path, "r") as f:
            submission = json.load(f)
    except Exception as e:
        errors.append(f"❌ Failed to load submission JSON: {e}")
        return errors

    try:
        with open(challenges_path, "r") as f:
            challenges = json.load(f)
    except Exception as e:
        errors.append(f"❌ Failed to load challenges JSON: {e}")
        return errors

    # --- Validate structure ---
    if not isinstance(submission, dict):
        errors.append("❌ Submission must be a dict at top level {task_id: predictions}.")
        return errors

    for task_id, preds in submission.items():
        if task_id not in challenges:
            errors.append(f"⚠️ Task ID {task_id} not found in challenges file.")

        if not isinstance(preds, list):
            errors.append(f"❌ Task {task_id}: predictions must be a list.")
            continue

        expected_n = len(challenges.get(task_id, {}).get("test", []))
        if len(preds) != expected_n:
            errors.append(f"❌ Task {task_id}: expected {expected_n} predictions, found {len(preds)}.")
            continue

        for i, pred in enumerate(preds):
            if not isinstance(pred, dict):
                errors.append(f"❌ Task {task_id} test[{i}]: must be dict with attempt_1 and attempt_2.")
                continue
            for key in ["attempt_1", "attempt_2"]:
                if key not in pred:
                    errors.append(f"❌ Task {task_id} test[{i}]: missing {key}.")
                    continue
                grid = pred[key]
                if not is_rectangular(grid):
                    errors.append(f"❌ Task {task_id} test[{i}] {key}: invalid grid format.")

    if not errors:
        print("✅ Submission format looks valid!")
    else:
        print("Found issues:")
        for e in errors:
            print(e)

    return errors

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", required=True, help="Path to submission.json")
    parser.add_argument("--challenges", required=True, help="Path to arc-agi_test-challenges.json")
    args = parser.parse_args()

    validate_submission(args.submission, args.challenges)
