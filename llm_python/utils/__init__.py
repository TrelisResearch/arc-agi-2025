"""
Utilities for ARC-AGI task processing.
"""

from .prompt_utils import (
    create_arc_prompt,
    extract_python_code,
    _format_grid_for_prompt as format_grid_for_prompt,
    _get_grid_shape_string as get_grid_shape_string
)

from .voting_utils import (
    filter_non_transductive_attempts,
    compute_weighted_majority_voting,
    compute_train_majority_voting,
    generic_voting,
    serialize_prediction_for_voting,
    deserialize_prediction_from_voting
)

from .metrics_utils import (
    calculate_task_metrics,
    format_metrics_display,
    metrics_to_percentages
)

from .timeout_utils import (
    execute_with_timeout
)

from .submission_validator import (
    validate_submission_file,
)

from .validator import (
    replace_invalid_grid,
)

__all__ = [
    'create_arc_prompt',
    'extract_python_code',
    'format_grid_for_prompt',
    'get_grid_shape_string',
    'filter_non_transductive_attempts',
    'compute_weighted_majority_voting',
    'compute_train_majority_voting',
    'generic_voting',
    'serialize_prediction_for_voting',
    'deserialize_prediction_from_voting',
    'calculate_task_metrics',
    'format_metrics_display',
    'metrics_to_percentages',
    'execute_with_timeout',
    'validate_submission_file',
    'replace_invalid_grid'
]
