"""
Validation module for SOAR datasets.

Provides comprehensive validation including schema checks, business logic validation,
and program correctness testing.
"""

from typing import List, Optional
from dataclasses import dataclass
import argparse
import random
import pandas as pd
from llm_python.datasets.io import validate_soar_dataframe_schema, read_soar_parquet
from llm_python.utils.arc_tester import ArcTester
from llm_python.utils.grids import grids_equal
from llm_python.utils.task_loader import get_task_loader
from llm_python.utils.validator import ARCTaskValidator


@dataclass
class ValidateRowResult:
    is_valid: bool
    errors: List[str]


def validate_soar_row(row: dict) -> ValidateRowResult:
    """
    Validate a single SOAR row for schema and business logic.
    Returns a list of error messages (empty if valid).
    """
    errors = []
    required_fields = [
        "task_id",
        "code",
        "correct_train_input",
        "correct_test_input",
        "predicted_train_output",
        "predicted_test_output",
        "model",
        "is_transductive",
    ]
    missing_fields = [field for field in required_fields if field not in row]
    if missing_fields:
        errors.append(f"Missing required fields: {missing_fields}")

    # Field type and value checks
    if "task_id" in row and (
        not isinstance(row["task_id"], str) or not row["task_id"].strip()
    ):
        errors.append("task_id must be a non-empty string")
    if "code" in row and (not isinstance(row["code"], str) or not row["code"].strip()):
        errors.append("code must be a non-empty string")
    if "model" in row and (
        not isinstance(row["model"], str) or not row["model"].strip()
    ):
        errors.append("model must be a non-empty string")
    for field in ["correct_train_input", "correct_test_input"]:
        if field in row:
            if not isinstance(row[field], list):
                errors.append(f"{field} must be a list")
            elif not all(isinstance(x, bool) for x in row[field]):
                errors.append(f"{field} must be a list of booleans")
            elif len(row[field]) == 0:
                errors.append(f"{field} is an empty list")
    for field in ["predicted_train_output", "predicted_test_output"]:
        if field in row:
            if not isinstance(row[field], list):
                errors.append(f"{field} must be a list")
            elif len(row[field]) == 0:
                errors.append(f"{field} is an empty list")
            else:
                # Use common prediction validator
                is_valid, validation_errors = ARCTaskValidator.validate_prediction_list(row[field], field)
                errors.extend(validation_errors)
    if (
        "reasoning" in row
        and row["reasoning"] is not None
        and not isinstance(row["reasoning"], str)
    ):
        errors.append("reasoning must be a string if provided")
    if "is_transductive" in row and not isinstance(row["is_transductive"], bool):
        errors.append("is_transductive must be a boolean")
    return ValidateRowResult(is_valid=not errors, errors=errors)


@dataclass
class ValidationResult:
    total_rows: int
    schema_valid: bool
    schema_error: Optional[str]
    business_logic_valid: bool
    business_logic_issues: List[str]

    def is_valid(self) -> bool:
        return self.schema_valid and self.business_logic_valid

    def summary(self) -> str:
        lines = [
            "Format validation:",
            f"    Total programs: {self.total_rows}",
            f"    Schema valid: {'PASS' if self.schema_valid else 'FAIL'}",
            f"    Business logic valid: {'PASS' if self.business_logic_valid else 'FAIL'}",
            f"    Issues: {len(self.business_logic_issues)}",
        ]
        if self.schema_error:
            lines.append(f"    Schema error: {self.schema_error}")
        if self.business_logic_issues:
            lines.append("    Sample issues:")
            for issue in self.business_logic_issues[:3]:
                lines.append(f"      {issue}")
        return "\n".join(lines)


@dataclass
class CorrectnessValidationResult:
    total_rows: int
    sample_size: int
    correctness_valid: bool
    correctness_errors: List[str]

    def is_valid(self) -> bool:
        return self.correctness_valid

    def summary(self) -> str:
        lines = [
            "Correctness validation:",
            f"    Total programs: {self.total_rows}",
            f"    Sample size: {self.sample_size}",
            f"    Correctness valid: {'PASS' if self.correctness_valid else 'FAIL'}",
            f"    Errors: {len(self.correctness_errors)}",
        ]
        if self.correctness_errors:
            lines.append("    Sample errors:")
            for error in self.correctness_errors[:3]:
                lines.append(f"      {error}")
        return "\n".join(lines)


def validate_soar_dataframe(df: pd.DataFrame) -> ValidationResult:
    """
    Validate SOAR DataFrame format: schema and business logic.
    """
    total_rows = len(df)
    schema_valid = True
    schema_error = None
    try:
        validate_soar_dataframe_schema(df)
    except ValueError as e:
        schema_valid = False
        schema_error = str(e)

    business_logic_valid = True
    business_logic_issues = []
    # Row-level validation
    for idx, row in df.iterrows():
        row_validation_result = validate_soar_row(row.to_dict())
        if not row_validation_result.is_valid:
            for err in row_validation_result.errors:
                business_logic_issues.append(f"Row {idx}: {err}")
    if business_logic_issues:
        business_logic_valid = False
    return ValidationResult(
        total_rows=total_rows,
        schema_valid=schema_valid,
        schema_error=schema_error,
        business_logic_valid=business_logic_valid,
        business_logic_issues=business_logic_issues,
    )


def validate_soar_dataframe_correctness(
    df: pd.DataFrame, correctness_samples: int = 1000, seed: int = 42
) -> CorrectnessValidationResult:
    """
    Validate SOAR DataFrame program correctness.
    """
    total_rows = len(df)
    correctness_valid = True
    correctness_errors = []
    sample_size = min(correctness_samples, total_rows)
    if sample_size > 0:
        random.seed(seed)
        if sample_size == total_rows:
            sample_df = df
        else:
            sample_indices = random.sample(range(total_rows), sample_size)
            sample_df = df.iloc[sample_indices]
        task_loader = get_task_loader()
        arc_tester = ArcTester()
        for idx, row in sample_df.iterrows():
            try:
                result = arc_tester.test_program(
                    row["code"], task_loader.get_task(row["task_id"])
                )
                for i, train_output in enumerate(row["predicted_train_output"]):
                    if not grids_equal(result.train_outputs[i], train_output):
                        correctness_errors.append(
                            f"Row {idx}, Train Output {i}: predicted != actual"
                        )
                for i, test_output in enumerate(row["predicted_test_output"]):
                    if not grids_equal(result.test_outputs[i], test_output):
                        correctness_errors.append(
                            f"Row {idx}, Test Output {i}: predicted != actual"
                        )
            except Exception as e:
                correctness_errors.append(
                    f"Row {idx}: Program execution failed - {str(e)}"
                )
    if correctness_errors:
        correctness_valid = False
    return CorrectnessValidationResult(
        total_rows=total_rows,
        sample_size=sample_size,
        correctness_valid=correctness_valid,
        correctness_errors=correctness_errors,
    )


def main():
    """Main method to validate a parquet file."""
    parser = argparse.ArgumentParser(description="Validate SOAR dataset parquet file")
    parser.add_argument("parquet_file", help="Path to the parquet file to validate")
    parser.add_argument(
        "--correctness-samples",
        type=int,
        default=1000,
        help="Number of random rows to validate for program correctness (default: 1000)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for sampling (default: 42)"
    )

    args = parser.parse_args()

    try:
        df = read_soar_parquet(args.parquet_file)
    except Exception as e:
        print(f"Error loading parquet file: {e}")
        return 1

    # Validate format
    format_result = validate_soar_dataframe(df)
    print(format_result.summary())

    # Validate correctness
    correctness_result = validate_soar_dataframe_correctness(
        df, correctness_samples=args.correctness_samples, seed=args.seed
    )
    print(correctness_result.summary())

    # Return appropriate exit code
    return 0 if (format_result.is_valid() and correctness_result.is_valid()) else 1


if __name__ == "__main__":
    exit(main())
