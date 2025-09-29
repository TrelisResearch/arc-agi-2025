"""
Submission file validation utility for ARC competitions.
Validates submission.json files against competition guidelines.
"""

import json
from typing import List
from .validator import ARCTaskValidator


def validate_submission_file(
    submission_path: str, expected_task_ids: List[str], challenges_path: str
) -> None:
    """
    Validate submission file format against competition guidelines.

    Args:
        submission_path: Path to submission.json file
        expected_task_ids: List of expected task IDs (optional, derived from challenges_path if not provided)
        challenges_path: Path to challenges file (for validation and task ID extraction)
    """
    print(f"\n🔍 VALIDATING SUBMISSION: {submission_path}")

    errors = []
    warnings = []

    try:
        # Load the submission file
        with open(submission_path, "r") as f:
            submission = json.load(f)

        # Load challenges if provided for prediction count validation and task ID extraction
        challenges = {}
        if challenges_path:
            try:
                with open(challenges_path, "r") as f:
                    challenges = json.load(f)
                print(f"📁 Loaded challenges: {len(challenges)} tasks")

                # If no expected_task_ids provided, derive from challenges (runtime scenario)
                if expected_task_ids is None:
                    expected_task_ids = list(challenges.keys())
                    print(
                        f"📋 Using task IDs from runtime challenges file: {len(expected_task_ids)} tasks"
                    )

            except Exception as e:
                errors.append(f"Could not load challenges file: {e}")
                challenges = {}
                if expected_task_ids is None:
                    errors.append("No task IDs available for validation")
                    return

        # Check 1: File is a dictionary
        if not isinstance(submission, dict):
            errors.append("Submission file must be a JSON dictionary")
            return

        # Check 2: Task ID coverage validation
        if expected_task_ids is not None:
            submission_task_ids = set(submission.keys())
            expected_task_ids_set = set(expected_task_ids)

            missing_tasks = expected_task_ids_set - submission_task_ids
            extra_tasks = submission_task_ids - expected_task_ids_set

            if missing_tasks:
                errors.append(
                    f"Missing required task IDs: {sorted(list(missing_tasks))[:10]}{'...' if len(missing_tasks) > 10 else ''} ({len(missing_tasks)} total)"
                )

            if extra_tasks:
                if challenges_path:
                    # Runtime scenario: extra tasks not in challenges file are serious errors
                    errors.append(
                        f"Submission contains task IDs not in challenges file: {sorted(list(extra_tasks))[:10]}{'...' if len(extra_tasks) > 10 else ''} ({len(extra_tasks)} total)"
                    )
                else:
                    # Development scenario: just warn about extra tasks
                    warnings.append(
                        f"Extra task IDs not in expected set: {sorted(list(extra_tasks))[:10]}{'...' if len(extra_tasks) > 10 else ''} ({len(extra_tasks)} total)"
                    )

            # Perfect match validation for runtime
            if (
                challenges_path
                and len(missing_tasks) == 0
                and len(extra_tasks) == 0
            ):
                print(f"✅ Perfect task ID match: {len(expected_task_ids)} tasks")
        else:
            warnings.append("No expected task IDs provided for validation")

        # Check 3: Validate structure for each task
        empty_predictions_count = 0
        prediction_count_errors = 0

        for task_id, predictions in submission.items():
            # Must be a list
            if not isinstance(predictions, list):
                errors.append(f"Task {task_id}: predictions must be a list")
                continue

            # Must have at least one prediction
            if len(predictions) == 0:
                errors.append(f"Task {task_id}: must have at least one prediction")
                continue

            # Check prediction count matches test inputs (if challenges available)
            if challenges and task_id in challenges:
                expected_count = len(challenges[task_id].get("test", []))
                if len(predictions) != expected_count:
                    errors.append(
                        f"Task {task_id}: has {len(predictions)} predictions but task has {expected_count} test inputs"
                    )
                    prediction_count_errors += 1

            # Check each prediction
            for i, pred in enumerate(predictions):
                if not isinstance(pred, dict):
                    errors.append(
                        f"Task {task_id}[{i}]: prediction must be a dictionary"
                    )
                    continue

                # Must have both attempt_1 and attempt_2
                if "attempt_1" not in pred:
                    errors.append(f"Task {task_id}[{i}]: missing 'attempt_1'")
                if "attempt_2" not in pred:
                    errors.append(f"Task {task_id}[{i}]: missing 'attempt_2'")

                # Validate attempts using proper ARC grid validation
                for attempt_key in ["attempt_1", "attempt_2"]:
                    if attempt_key in pred:
                        attempt = pred[attempt_key]
                        # Use ARCTaskValidator for comprehensive validation
                        if not ARCTaskValidator.validate_prediction(
                            attempt, f"Task {task_id}[{i}].{attempt_key}"
                        ):
                            errors.append(
                                f"Task {task_id}[{i}].{attempt_key}: invalid ARC grid format"
                            )
                        elif attempt == [[0, 0], [0, 0]]:
                            empty_predictions_count += 1

        # Summary statistics
        total_predictions = sum(len(predictions) for predictions in submission.values())
        unique_tasks = len(submission)

        print("📊 Validation Results:")
        print(f"  Total tasks: {unique_tasks}")
        print(f"  Total predictions: {total_predictions}")
        print(f"  Empty predictions ([[0,0],[0,0]]): {empty_predictions_count}")
        if challenges:
            print(f"  Prediction count mismatches: {prediction_count_errors}")

        # Report errors and warnings
        if errors:
            print(f"❌ VALIDATION FAILED - {len(errors)} errors:")
            for error in errors[:10]:  # Show first 10 errors
                print(f"   • {error}")
            if len(errors) > 10:
                print(f"   ... and {len(errors) - 10} more errors")
        else:
            print("✅ VALIDATION PASSED - No structural errors found")

        if warnings:
            print(f"⚠️  {len(warnings)} warnings:")
            for warning in warnings[:5]:  # Show first 5 warnings
                print(f"   • {warning}")
            if len(warnings) > 5:
                print(f"   ... and {len(warnings) - 5} more warnings")

        # Final status
        if not errors:
            print("🎯 Submission file is ready for competition!")
        else:
            print("🚨 Fix errors before submitting!")

    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing error: {e}")
    except Exception as e:
        print(f"❌ Validation error: {e}")
