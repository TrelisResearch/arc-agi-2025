# Utils Tests

This directory contains comprehensive tests for the `utils` module, following proper pytest conventions.

## Test Structure

- `test_task_loader.py` - Tests for the `TaskLoader` class
- `test_scoring.py` - Tests for the `GridScorer` and `ProgramExecutor` classes
- `test_voting_utils.py` - Tests for voting utility functions
- `test_voting_edge_cases.py` - Edge case tests for voting functionality
- `test_metrics_voting_integration.py` - Integration tests for metrics calculation with voting

## Running Tests

### Run all utils tests:
```bash
uv run python -m pytest utils/tests/ -v
```

### Run specific test file:
```bash
uv run python -m pytest utils/tests/test_task_loader.py -v
uv run python -m pytest utils/tests/test_scoring.py -v
uv run python -m pytest utils/tests/test_voting_utils.py -v
uv run python -m pytest utils/tests/test_voting_edge_cases.py -v
uv run python -m pytest utils/tests/test_metrics_voting_integration.py -v
```

### Run all voting-related tests:
```bash
uv run python -m pytest utils/tests/test_voting* -v
```

### Run only unit tests (excludes integration tests):
```bash
uv run python -m pytest utils/tests/ -v -m "not integration"
```

### Run only integration tests:
```bash
uv run python -m pytest utils/tests/ -v -m integration
```

## Test Types

### Unit Tests
Most tests are unit tests that use mocking and temporary fixtures to test functionality in isolation. These tests:
- Use temporary directories with mock data structures
- Mock external dependencies (like the sandbox executor)
- Test edge cases and error conditions
- Run quickly and don't require external resources

### Integration Tests
Integration tests are marked with `@pytest.mark.integration` and test against real data:
- `test_load_real_shortest_task` - Tests loading actual ARC tasks from the data directory
- `test_get_real_available_subsets` - Tests getting real subset information
- `test_execute_simple_program_real_sandbox` - Tests program execution with the real sandbox

Integration tests will be skipped if the required data or dependencies are not available.

## Test Coverage

### TaskLoader Tests
- Initialization with valid/invalid data roots
- Loading tasks from training and evaluation directories
- Loading subsets and handling missing files
- Formatting tasks for prompts
- Grid formatting utilities
- Error handling for missing files and directories

### GridScorer Tests
- Perfect, partial, and no matches
- Grid dimension mismatches
- Empty and single-cell grids
- Large grid scoring
- Error handling

### ProgramExecutor Tests
- Successful program execution with different function names (`transform`, `solve`, `apply_transform`)
- Timeout handling
- Error handling
- Programs with no output or invalid functions
- Numpy type conversion
- Exception handling during sandbox creation

### Voting and Metrics Tests
- Voting utility serialization/deserialization
- Weighted majority voting with train accuracy preferences
- Train-majority voting with best group filtering
- Metrics calculation integration with voting results
- Single vs multi-test case format handling
- Bug regression tests for the `_first()` function corruption issue

## Key Features

1. **Comprehensive Coverage**: Tests cover happy paths, edge cases, and error conditions
2. **Mocking**: External dependencies are mocked to ensure tests are reliable and fast
3. **Fixtures**: Reusable test data and temporary structures
4. **Integration Support**: Real data tests for end-to-end validation
5. **Clear Documentation**: Each test has descriptive names and docstrings

## Best Practices Demonstrated

- Use of pytest fixtures for reusable test data
- Proper mocking of external dependencies
- Separation of unit and integration tests
- Comprehensive error condition testing
- Clear test organization and naming
- Use of temporary directories for file system tests
