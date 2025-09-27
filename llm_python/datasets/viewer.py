import argparse
from typing import List, Optional

from llm_python.datasets.io import read_soar_parquet
import numpy as np

from llm_python.utils.task_loader import get_task_loader

# ANSI color codes for numbers 0-9 and PAD token (10)
COLOR_CODES = {
    0: "\033[90m",  # Dark gray (background/empty)
    1: "\033[94m",  # Blue
    2: "\033[91m",  # Red
    3: "\033[92m",  # Green
    4: "\033[93m",  # Yellow
    5: "\033[95m",  # Magenta
    6: "\033[96m",  # Cyan
    7: "\033[97m",  # White
    8: "\033[35m",  # Purple
    9: "\033[33m",  # Orange-ish
    10: "\033[100m\033[37m",  # PAD token: White text on black background (*)
}
RESET_CODE = "\033[0m"


def grid_to_text(grid: List[List[int]], use_colors: bool = True) -> str:
    if not use_colors:
        return "\n".join(" ".join("*" if cell == 10 else str(cell) for cell in row) for row in grid)

    lines = []
    for row in grid:
        colored_cells = []
        for cell in row:
            # Display PAD tokens (10) as '*' for better readability
            display_char = "*" if cell == 10 else str(cell)
            color_code = COLOR_CODES.get(cell, "")
            colored_cells.append(f"{color_code}{display_char}{RESET_CODE}")
        lines.append(" ".join(colored_cells))
    return "\n".join(lines)


def strip_ansi_codes(text: str) -> str:
    """Remove ANSI color codes from text for length calculation"""
    import re

    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    return ansi_escape.sub("", text)


def print_grids_horizontally(
    grids: List[List[List[int]]],
    labels: Optional[List[str]] = None,
    use_colors: bool = True,
) -> str:
    """Print multiple grids side by side horizontally"""
    if len(grids) == 0:
        return ""

    # Convert each grid to lines
    grid_lines = []
    max_height = 0

    for grid in grids:
        lines = grid_to_text(grid, use_colors).split("\n")
        grid_lines.append(lines)
        max_height = max(max_height, len(lines))

    # Pad all grids to the same height
    for lines in grid_lines:
        while len(lines) < max_height:
            # Use the visual width of the first line for padding
            if lines:
                visual_width = len(strip_ansi_codes(lines[0]))
                lines.append(" " * visual_width)
            else:
                lines.append("")

    # Calculate column widths based on visual content (without ANSI codes)
    # Also consider label width to prevent overlap
    col_widths = []
    for i, lines in enumerate(grid_lines):
        content_width = max(len(strip_ansi_codes(line)) for line in lines) if lines else 0
        label_width = len(labels[i]) if labels and i < len(labels) else 0
        # Use the maximum of content width and label width
        width = max(content_width, label_width)
        col_widths.append(width)

    # Use larger separator between columns for better spacing
    separator = "    "  # 4 spaces instead of "  |  "

    # Add labels if provided
    result_lines = []
    if labels:
        label_line = ""
        for i, (label, width) in enumerate(zip(labels, col_widths)):
            padded_label = label.ljust(width)
            label_line += padded_label
            if i < len(labels) - 1:
                label_line += separator
        result_lines.append(label_line)

        # Add separator line
        sep_line = ""
        for i, width in enumerate(col_widths):
            sep_line += "-" * width
            if i < len(col_widths) - 1:
                sep_line += separator
        result_lines.append(sep_line)

    # Combine grid lines horizontally
    for row_idx in range(max_height):
        line = ""
        for col_idx, (lines, width) in enumerate(zip(grid_lines, col_widths)):
            current_line = lines[row_idx] if row_idx < len(lines) else ""
            visual_width = len(strip_ansi_codes(current_line))
            padding_needed = max(0, width - visual_width)
            padded_line = current_line + (" " * padding_needed)
            line += padded_line
            if col_idx < len(grid_lines) - 1:
                line += separator
        result_lines.append(line)

    return "\n".join(result_lines)


def print_soar_dataset(
    file_path, train_filter="any", test_filter="any", task_id_filter=None, use_colors=True
):
    task_loader = get_task_loader()
    df = read_soar_parquet(file_path)

    original_size = len(df)
    print(f"Original dataset size: {original_size} rows")

    # Apply task ID filter
    if task_id_filter:
        df = df[df["task_id"] == task_id_filter]
        print(f"Task ID filter: {task_id_filter}")

    # Apply train filter
    if train_filter == "all-correct":
        df = df[df["correct_train_input"].apply(lambda x: all(val == 1 for val in x))]
        print("Train filter: All training inputs correct")
    elif train_filter == "none-correct":
        df = df[df["correct_train_input"].apply(lambda x: all(val == 0 for val in x))]
        print("Train filter: No training inputs correct")
    elif train_filter == "partial-correct":
        df = df[
            df["correct_train_input"].apply(
                lambda x: any(val == 1 for val in x) and not all(val == 1 for val in x)
            )
        ]
        print("Train filter: Partial training inputs correct (some but not all)")
    else:
        print("Train filter: None (any train correctness)")

    # Apply test filter
    if test_filter == "all-correct":
        df = df[df["correct_test_input"].apply(lambda x: all(val == 1 for val in x))]
        print("Test filter: All test inputs correct")
    elif test_filter == "none-correct":
        df = df[df["correct_test_input"].apply(lambda x: all(val == 0 for val in x))]
        print("Test filter: No test inputs correct")
    elif test_filter == "partial-correct":
        df = df[
            df["correct_test_input"].apply(
                lambda x: any(val == 1 for val in x) and not all(val == 1 for val in x)
            )
        ]
        print("Test filter: Partial test inputs correct (some but not all)")
    else:
        print("Test filter: None (any test correctness)")

    filtered_size = len(df)
    print(f"Dataset size after filtering: {filtered_size} rows")

    if filtered_size == 0:
        print("No rows match the selected filters.")
        return

    print(
        f"Showing {filtered_size} rows ({filtered_size / original_size * 100:.1f}% of original)"
    )
    print()

    np.random.seed(42)
    shuffled_df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    for idx, (_, row) in enumerate(shuffled_df.iterrows()):
        # Task separator
        print("=" * 80)
        print(f"TASK #{idx + 1}")
        print("=" * 80)

        code = row["code"]
        task_data = task_loader.get_task(row["task_id"])

        print(f"Task ID: {row['task_id']}")

        # Show correctness information
        correct_train = row["correct_train_input"]
        correct_test = row["correct_test_input"]

        print(
            f"Train Correctness: {correct_train} ({sum(correct_train)}/{len(correct_train)} correct)"
        )
        print(
            f"Test Correctness:  {correct_test} ({sum(correct_test)}/{len(correct_test)} correct)"
        )
        print()

        # Show at most 3 training pairs
        train_examples = task_data["train"][:3]

        if train_examples:
            print("TRAINING EXAMPLES:")
            print("-" * 50)

            # Collect inputs and outputs for horizontal display
            inputs = [example["input"] for example in train_examples]
            outputs = [example["output"] for example in train_examples]

            # Create labels with correctness indicators
            input_labels = []
            output_labels = []
            for i in range(len(inputs)):
                if i < len(correct_train):
                    status = "✓" if correct_train[i] == 1 else "✗"
                    input_labels.append(f"Input #{i + 1} {status}")
                    output_labels.append(f"Output #{i + 1} {status}")
                else:
                    input_labels.append(f"Input #{i + 1}")
                    output_labels.append(f"Output #{i + 1}")

            # Print inputs horizontally
            print("INPUTS:")
            print(print_grids_horizontally(inputs, input_labels, use_colors))
            print()

            # Print outputs horizontally
            print("EXPECTED:")
            print(print_grids_horizontally(outputs, output_labels, use_colors))
            print()

            # Print outputs horizontally
            print("OUTPUT:")
            print(
                print_grids_horizontally(
                    row["predicted_train_output"][:3], output_labels, use_colors
                )
            )
            print()

        # Show at most 3 test pairs
        test_examples = task_data["test"][:3]

        if test_examples:
            print("TEST EXAMPLES:")
            print("-" * 50)

            # Collect inputs and outputs for horizontal display
            test_inputs = [example["input"] for example in test_examples]
            test_outputs = [example.get("output") for example in test_examples]

            # Create labels with correctness indicators
            test_input_labels = []
            test_output_labels = []
            for i in range(len(test_inputs)):
                if i < len(correct_test):
                    status = "✓" if correct_test[i] == 1 else "✗"
                    test_input_labels.append(f"Test Input #{i + 1} {status}")
                    test_output_labels.append(f"Test Output #{i + 1} {status}")
                else:
                    test_input_labels.append(f"Test Input #{i + 1}")
                    test_output_labels.append(f"Test Output #{i + 1}")

            # Print test inputs horizontally
            print("TEST INPUTS:")
            print(print_grids_horizontally(test_inputs, test_input_labels, use_colors))
            print()

            # Print test expected outputs horizontally (if available)
            if any(output is not None for output in test_outputs):
                print("TEST EXPECTED:")
                # Replace None with empty grids for display
                display_outputs = [output if output is not None else [[]] for output in test_outputs]
                print(print_grids_horizontally(display_outputs, test_output_labels, use_colors))
                print()

            # Print test predicted outputs horizontally
            print("TEST PREDICTED:")
            print(
                print_grids_horizontally(
                    row["predicted_test_output"][:3], test_output_labels, use_colors
                )
            )
            print()

        print("-" * 50)
        print("GENERATED CODE:")
        print("-" * 50)
        print(code)
        print()

        # Wait for user input before showing next task
        user_input = input("Press Enter to continue to next task (or 'q' to quit)...")
        if user_input.lower() == "q":
            break
        print()  # Extra spacing between tasks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print SOAR dataset")
    parser.add_argument(
        "file", type=str, help="Path to the SOAR dataset file"
    )
    parser.add_argument(
        "--train-filter",
        type=str,
        choices=["any", "all-correct", "none-correct", "partial-correct"],
        default="any",
        help="Filter based on training correctness: "
        "'any' (no filter), "
        "'all-correct' (all training inputs correct), "
        "'none-correct' (no training inputs correct), "
        "'partial-correct' (some but not all training inputs correct)",
    )
    parser.add_argument(
        "--test-filter",
        type=str,
        choices=["any", "all-correct", "none-correct", "partial-correct"],
        default="any",
        help="Filter based on test correctness: "
        "'any' (no filter), "
        "'all-correct' (all test inputs correct), "
        "'none-correct' (no test inputs correct), "
        "'partial-correct' (some but not all test inputs correct)",
    )
    parser.add_argument(
        "--no-colors", action="store_true", help="Disable colored output for grids"
    )
    parser.add_argument(
        "--task-id",
        type=str,
        help="Filter to show only the specified task ID"
    )
    args = parser.parse_args()
    print(f"Using file: {args.file}")
    print(f"Train filter: {args.train_filter}")
    print(f"Test filter: {args.test_filter}")
    if args.task_id:
        print(f"Task ID filter: {args.task_id}")
    print(f"Colors: {'disabled' if args.no_colors else 'enabled'}")
    print_soar_dataset(
        args.file, args.train_filter, args.test_filter, args.task_id, not args.no_colors
    )
