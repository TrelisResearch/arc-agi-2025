# Tests and Debug Scripts

This directory contains test scripts and debugging utilities for the ARC-AGI training and validation system.

## Validation & Training Data

- `validate_training_data.py` - **Main validation script** for verifying training dataset quality
- `check_expected_output.py` - Debug script for investigating specific JSONL line outputs
- `debug_execution.py` - Debug program execution differences
- `investigate_validation_failure.py` - Comprehensive investigation of validation issues
- `test_validation.py` - Simple validation test
- `trace_output_formatting.py` - Debug grid serialization format/parse cycle

## Training Data Generation

- `test_generation_flow.py` - Test training data generation process
- `test_generation_flow_fixed.py` - Test generation with fixed serialization
- `test_execution_diff.py` - Test execution environment differences
- `test_multiple_examples.py` - Test handling of multiple training examples

## API & Integration Tests

- `test_arc_visual_with_api.py` - Test ARC visual integration with API
- `test_image_responses_api.py` - Test image response handling
- `test_visual_integration.py` - Visual integration tests
- `demo_arc_visual_integration.py` - Demo of ARC visual features
- `create_arc_grid_demo.py` - Create ARC grid demonstrations

## Reasoning & Persistence Tests

- `test_reasoning_persistence.py` - Test reasoning state persistence
- `test_hidden_reasoning_persistence.py` - Test hidden reasoning persistence
- `test_multiturn_reasoning.py` - Test multi-turn reasoning capabilities

## Usage

**Validate training data:**
```bash
uv run python tests/validate_training_data.py path/to/training_data.jsonl --verbose
```

**Debug specific issues:**
```bash
uv run python tests/debug_execution.py
uv run python tests/investigate_validation_failure.py
```

**Test generation process:**
```bash
uv run python tests/test_generation_flow_fixed.py
```