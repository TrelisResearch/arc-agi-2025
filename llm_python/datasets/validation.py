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
from llm_python.utils.task_loader import TaskLoader


@dataclass
class ValidationResult:
    """Results from validating a SOAR dataset."""

    total_rows: int
    schema_valid: bool
    schema_error: Optional[str]
    business_logic_valid: bool
    business_logic_issues: List[str]
    correctness_sample_size: int
    correctness_valid: bool
    correctness_errors: List[str]

    def is_valid(self) -> bool:
        """Return True if all validations passed."""
        return (
            self.schema_valid and self.business_logic_valid and self.correctness_valid
        )

    def summary(self) -> str:
        """Return a formatted summary of validation results."""
        lines = [
            "Validation results:",
            f"    Total programs: {self.total_rows}",
            f"    Rows failing schema validation: {0 if self.schema_valid else 1} [{'PASS' if self.schema_valid else 'FAIL'}]",
            f"    Rows failing data inspection: {len(self.business_logic_issues)} [{'PASS' if self.business_logic_valid else 'FAIL'}]",
            f"    Rows failing correctness checks: {len(self.correctness_errors)} [{'PASS' if self.correctness_valid else 'FAIL'}]",
        ]

        # Collect sample errors
        sample_errors = []
        
        if not self.schema_valid and self.schema_error:
            sample_errors.append(f"Schema: {self.schema_error}")
        
        if self.business_logic_issues:
            for issue in self.business_logic_issues[:3]:  # Show first 3
                sample_errors.append(f"Data: {issue}")
        
        if self.correctness_errors:
            for error in self.correctness_errors[:3]:  # Show first 3
                sample_errors.append(f"Correctness: {error}")
        
        if sample_errors:
            lines.append("    Sample of errors:")
            for error in sample_errors:
                lines.append(f"      {error}")

        return "\n".join(lines)


def validate_soar_dataframe(
    df: pd.DataFrame, correctness_samples: int = 1000, seed: int = 42
) -> ValidationResult:
    """
    Comprehensive validation of a SOAR DataFrame.

    Args:
        df: DataFrame to validate
        correctness_samples: Maximum number of rows to test for program correctness
        seed: Random seed for sampling

    Returns:
        ValidationResult with detailed validation information
    """
    total_rows = len(df)

    # 1. Schema validation using io.py method
    schema_valid = True
    schema_error = None
    try:
        validate_soar_dataframe_schema(df)
    except ValueError as e:
        schema_valid = False
        schema_error = str(e)

    # 2. Business logic validation
    business_logic_valid = True
    business_logic_issues = []

    # Only do business logic checks if we have the expected columns
    expected_columns = {
        "correct_train_input",
        "correct_test_input",
        "predicted_train_output",
        "predicted_test_output",
        "code",
    }

    if expected_columns.issubset(set(df.columns)):
        # Check for empty lists in list columns
        list_columns = [
            "correct_train_input",
            "correct_test_input",
            "predicted_train_output",
            "predicted_test_output",
        ]

        for col in list_columns:
            if col in df.columns:
                empty_lists = (
                    df[col].apply(lambda x: x is not None and len(x) == 0).sum()
                )
                if empty_lists > 0:
                    business_logic_issues.append(
                        f"Column '{col}' has {empty_lists} empty lists"
                    )

        # Check for rows where code field is empty or just whitespace
        empty_code = df["code"].str.strip().eq("").sum()
        if empty_code > 0:
            business_logic_issues.append(
                f"{empty_code} rows have empty or whitespace-only code"
            )

    # If there are issues, mark business logic as having problems
    # (but don't fail completely - these might be warnings)
    if business_logic_issues:
        business_logic_valid = False

    # 3. Program correctness validation
    correctness_valid = True
    correctness_errors = []

    # Sample rows for correctness testing
    sample_size = min(correctness_samples, total_rows)
    if sample_size > 0:
        random.seed(seed)
        if sample_size == total_rows:
            sample_df = df
        else:
            sample_indices = random.sample(range(total_rows), sample_size)
            sample_df = df.iloc[sample_indices]

        # Test program correctness
        task_loader = TaskLoader()
        arc_tester = ArcTester()

        for idx, row in sample_df.iterrows():
            try:
                result = arc_tester.test_program(
                    row["code"], task_loader.get_task(row["task_id"])
                )

                # Check training outputs
                for i, train_output in enumerate(row["predicted_train_output"]):
                    if not grids_equal(result.train_outputs[i], train_output):
                        correctness_errors.append(
                            f"Row {idx}, Train Output {i}: predicted != actual"
                        )

                # Check test outputs
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

    return ValidationResult(
        total_rows=total_rows,
        schema_valid=schema_valid,
        schema_error=schema_error,
        business_logic_valid=business_logic_valid,
        business_logic_issues=business_logic_issues,
        correctness_sample_size=sample_size,
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

    # Validate the dataframe
    result = validate_soar_dataframe(
        df, correctness_samples=args.correctness_samples, seed=args.seed
    )

    # Print summary
    print(result.summary())

    # Return appropriate exit code
    return 0 if result.is_valid() else 1


if __name__ == "__main__":
    exit(main())
