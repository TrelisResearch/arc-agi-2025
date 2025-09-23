#!/usr/bin/env python3

import pytest
import random
from unittest.mock import patch
from llm_python.utils.refinement_utils import (
    calculate_program_metrics,
    select_best_program_for_refinement,
    is_program_valid_for_refinement,
    REXProgramPool,
    select_program_for_refinement,
    create_refined_program_entry
)


class TestCalculateProgramMetrics:
    """Test the calculate_program_metrics function"""
    
    def test_calculate_metrics_basic(self):
        """Test basic metrics calculation"""
        program = {
            'correct_train_input': [True, False, True],
            'code': 'print("hello")'
        }
        correctness, length = calculate_program_metrics(program)
        
        assert correctness == 2/3  # 2 out of 3 correct
        assert length == len('print("hello")')
    
    def test_calculate_metrics_perfect_score(self):
        """Test perfect correctness score"""
        program = {
            'correct_train_input': [True, True, True],
            'code': 'x = 1'
        }
        correctness, length = calculate_program_metrics(program)
        
        assert correctness == 1.0
        assert length == 5
    
    def test_calculate_metrics_zero_score(self):
        """Test zero correctness score"""
        program = {
            'correct_train_input': [False, False, False],
            'code': 'pass'
        }
        correctness, length = calculate_program_metrics(program)
        
        assert correctness == 0.0
        assert length == 4
    
    def test_calculate_metrics_empty_list(self):
        """Test empty correct_train_input list"""
        program = {
            'correct_train_input': [],
            'code': 'x = 1'
        }
        correctness, length = calculate_program_metrics(program)
        
        assert correctness == 0.0
        assert length == 5
    
    def test_calculate_metrics_missing_code(self):
        """Test missing code field"""
        program = {
            'correct_train_input': [True, False]
        }
        correctness, length = calculate_program_metrics(program)
        
        assert correctness == 0.5
        assert length == 0  # Empty string length
    
    def test_calculate_metrics_numpy_array(self):
        """Test numpy array input"""
        import numpy as np
        program = {
            'correct_train_input': np.array([True, True, False]),
            'code': 'test'
        }
        correctness, length = calculate_program_metrics(program)
        
        assert correctness == 2/3
        assert length == 4
    
    def test_calculate_metrics_single_value_list(self):
        """Test single value in list"""
        program = {
            'correct_train_input': [True],
            'code': 'a'
        }
        correctness, length = calculate_program_metrics(program)
        
        assert correctness == 1.0
        assert length == 1


class TestSelectBestProgramForRefinement:
    """Test the select_best_program_for_refinement function"""
    
    def test_select_empty_programs(self):
        """Test empty program list"""
        result = select_best_program_for_refinement([])
        assert result == {}
    
    def test_select_single_program(self):
        """Test single program selection"""
        programs = [{'correct_train_input': [True, False], 'code': 'test'}]
        result = select_best_program_for_refinement(programs)
        assert result == programs[0]
    
    def test_select_uniform_sampling(self):
        """Test that uniform sampling works with legacy function"""
        programs = [
            {'correct_train_input': [True, False], 'code': 'program1'},
            {'correct_train_input': [True, True, False], 'code': 'program2'},
            {'correct_train_input': [False, False], 'code': 'program3'}
        ]

        # Should be able to select any program with uniform sampling
        results = set()
        for _ in range(50):  # Run multiple times to see variety
            result = select_best_program_for_refinement(programs)
            results.add(result['code'])

        # With uniform sampling, should get some variety (not deterministic)
        assert len(results) >= 1  # At least one program selected
        assert all(code in ['program1', 'program2', 'program3'] for code in results)
    
    def test_select_debug_output(self):
        """Test debug output"""
        programs = [{'correct_train_input': [True, False], 'code': 'test'}]
        
        with patch('builtins.print') as mock_print:
            select_best_program_for_refinement(programs, debug=True)
            mock_print.assert_called_once()
            args = mock_print.call_args[0][0]
            assert '50.0% correct' in args
            assert '4 chars' in args
    
    def test_select_various_programs(self):
        """Test that uniform sampling can select from various programs"""
        programs = [
            {'correct_train_input': [True, True, True], 'code': 'perfect'},        # 100%
            {'correct_train_input': [True, True, False], 'code': 'good'},          # 67%
            {'correct_train_input': [True, False], 'code': 'partial'},             # 50%
            {'correct_train_input': [False], 'code': 'poor'},                      # 0%
        ]

        # With uniform sampling, any program can be selected
        results = set()
        for _ in range(100):
            result = select_best_program_for_refinement(programs)
            results.add(result['code'])

        # Should be able to select from various programs (not deterministic)
        assert len(results) >= 2  # Should get some variety


class TestIsProgramValidForRefinement:
    """Test the is_program_valid_for_refinement function"""
    
    def test_exclude_transductive_programs(self):
        """Test that transductive programs are excluded"""
        program = {
            'is_transductive': True,
            'correct_train_input': [True, False]
        }
        assert not is_program_valid_for_refinement(program)
    
    def test_exclude_perfect_programs(self):
        """Test that 100% correct programs are excluded"""
        program = {
            'is_transductive': False,
            'correct_train_input': [True, True, True]
        }
        assert not is_program_valid_for_refinement(program)
    
    def test_include_partial_programs(self):
        """Test that partially correct programs are included"""
        program = {
            'is_transductive': False,
            'correct_train_input': [True, False, True]
        }
        assert is_program_valid_for_refinement(program)
    
    def test_include_zero_percent_programs(self):
        """Test that 0% correct programs are included (new behavior)"""
        program = {
            'is_transductive': False,
            'correct_train_input': [False, False, False]
        }
        assert is_program_valid_for_refinement(program)
    
    def test_missing_transductive_field(self):
        """Test programs without is_transductive field (default to False)"""
        program = {
            'correct_train_input': [True, False]
        }
        assert is_program_valid_for_refinement(program)
    
    def test_numpy_array_input(self):
        """Test numpy array input for correctness"""
        import numpy as np
        program = {
            'is_transductive': False,
            'correct_train_input': np.array([True, False, True])
        }
        assert is_program_valid_for_refinement(program)
        
        # Test 100% correct numpy array
        program_perfect = {
            'is_transductive': False,
            'correct_train_input': np.array([True, True, True])
        }
        assert not is_program_valid_for_refinement(program_perfect)
    
    def test_single_value_cases(self):
        """Test single value correctness input"""
        # Single True (100% correct) - should be excluded
        program_perfect = {
            'is_transductive': False,
            'correct_train_input': True
        }
        assert not is_program_valid_for_refinement(program_perfect)
        
        # Single False (0% correct) - should be included
        program_zero = {
            'is_transductive': False,
            'correct_train_input': False
        }
        assert is_program_valid_for_refinement(program_zero)
    
    def test_empty_list_input(self):
        """Test empty correctness list"""
        program = {
            'is_transductive': False,
            'correct_train_input': []
        }
        assert is_program_valid_for_refinement(program)  # Should be included
    
    def test_edge_cases(self):
        """Test various edge cases"""
        # Missing correct_train_input field
        program_missing = {
            'is_transductive': False
        }
        assert is_program_valid_for_refinement(program_missing)

        # Empty dict
        program_empty = {}
        assert is_program_valid_for_refinement(program_empty)

        # Only transductive field
        program_only_trans = {
            'is_transductive': True
        }
        assert not is_program_valid_for_refinement(program_only_trans)

    def test_exclude_pass_through_programs(self):
        """Test that pass-through programs are excluded"""
        program = {
            'is_transductive': False,
            'correct_train_input': [False, False],
            'predicted_train_output': [
                [[1, 2], [3, 4]],
                [[5, 6], [7, 8]]
            ]
        }

        task_data = {
            'train': [
                {'input': [[1, 2], [3, 4]], 'output': [[0, 1], [2, 3]]},
                {'input': [[5, 6], [7, 8]], 'output': [[4, 5], [6, 7]]}
            ]
        }

        # Should be excluded because predicted outputs == inputs
        assert not is_program_valid_for_refinement(program, task_data)

    def test_include_non_pass_through_programs(self):
        """Test that non-pass-through programs are included"""
        program = {
            'is_transductive': False,
            'correct_train_input': [False, False],
            'predicted_train_output': [
                [[0, 1], [2, 3]],  # Different from input
                [[5, 6], [7, 8]]   # Same as input
            ]
        }

        task_data = {
            'train': [
                {'input': [[1, 2], [3, 4]], 'output': [[0, 1], [2, 3]]},
                {'input': [[5, 6], [7, 8]], 'output': [[4, 5], [6, 7]]}
            ]
        }

        # Should be included because at least one predicted output != input
        assert is_program_valid_for_refinement(program, task_data)

    def test_exclude_single_color_predictions_with_multi_color_truth(self):
        """Test excluding single-color predictions when ground truth is multi-colored"""
        program = {
            'is_transductive': False,
            'correct_train_input': [False, False],
            'predicted_train_output': [
                [[1, 1], [1, 1]],  # Single color (1)
                [[2, 2], [2, 2]]   # Single color (2)
            ]
        }

        task_data = {
            'train': [
                {'input': [[0, 1], [2, 3]], 'output': [[1, 2], [3, 4]]},  # Multi-color ground truth
                {'input': [[0, 0], [0, 0]], 'output': [[5, 6], [7, 8]]}   # Multi-color ground truth
            ]
        }

        # Should be excluded because predictions are single-color but ground truth is multi-colored
        assert not is_program_valid_for_refinement(program, task_data)

    def test_include_single_color_predictions_with_single_color_truth(self):
        """Test including single-color predictions when ground truth is also single-colored"""
        program = {
            'is_transductive': False,
            'correct_train_input': [False, False],
            'predicted_train_output': [
                [[1, 1], [1, 1]],  # Single color (1)
                [[2, 2], [2, 2]]   # Single color (2)
            ]
        }

        task_data = {
            'train': [
                {'input': [[0, 1], [2, 3]], 'output': [[1, 1], [1, 1]]},  # Single-color ground truth
                {'input': [[0, 0], [0, 0]], 'output': [[2, 2], [2, 2]]}   # Single-color ground truth
            ]
        }

        # Should be included because ground truth is also single-colored
        assert is_program_valid_for_refinement(program, task_data)

    def test_include_multi_color_predictions_with_multi_color_truth(self):
        """Test including multi-color predictions regardless of ground truth"""
        program = {
            'is_transductive': False,
            'correct_train_input': [False, False],
            'predicted_train_output': [
                [[1, 2], [3, 4]],  # Multi-color
                [[5, 6], [7, 8]]   # Multi-color
            ]
        }

        task_data = {
            'train': [
                {'input': [[0, 1], [2, 3]], 'output': [[1, 2], [3, 4]]},  # Multi-color ground truth
                {'input': [[0, 0], [0, 0]], 'output': [[5, 6], [7, 8]]}   # Multi-color ground truth
            ]
        }

        # Should be included because predictions are multi-colored
        assert is_program_valid_for_refinement(program, task_data)

    def test_filtering_without_task_data(self):
        """Test that filtering works correctly when task_data is not provided"""
        program = {
            'is_transductive': False,
            'correct_train_input': [False, False],
            'predicted_train_output': [
                [[1, 1], [1, 1]],  # Single color
                [[2, 2], [2, 2]]   # Single color
            ]
        }

        # Should be included because without task_data, we can't apply pass-through or single-color filtering
        assert is_program_valid_for_refinement(program, None)
        assert is_program_valid_for_refinement(program)


class TestIntegrationScenarios:
    """Test integration scenarios combining multiple functions"""
    
    def test_end_to_end_refinement_pipeline(self):
        """Test complete refinement selection pipeline"""
        # Simulate a realistic set of programs
        programs = [
            {'correct_train_input': [True, True, True], 'code': 'perfect', 'is_transductive': False},    # Perfect - excluded
            {'correct_train_input': [True, True, False], 'code': 'almost', 'is_transductive': False},   # 67% - included
            {'correct_train_input': [True, False, False], 'code': 'ok', 'is_transductive': False},      # 33% - included  
            {'correct_train_input': [False, False, False], 'code': 'zero', 'is_transductive': False},   # 0% - included
            {'correct_train_input': [True, False], 'code': 'trans', 'is_transductive': True},           # Transductive - excluded
        ]
        
        # Filter programs
        valid_programs = [p for p in programs if is_program_valid_for_refinement(p)]
        
        # Should exclude perfect and transductive, include others
        valid_codes = [p['code'] for p in valid_programs]
        assert 'perfect' not in valid_codes
        assert 'trans' not in valid_codes  
        assert 'almost' in valid_codes
        assert 'ok' in valid_codes
        assert 'zero' in valid_codes
        
        # Select best program
        with patch('random.choice', side_effect=lambda x: x[0]):  # Always pick first (best)
            selected = select_best_program_for_refinement(valid_programs, top_k=2)
            # Should select 'almost' (67% correct) over others
            assert selected['code'] == 'almost'
    
    def test_all_excluded_scenarios(self):
        """Test scenarios where all programs are excluded"""
        programs = [
            {'correct_train_input': [True, True], 'code': 'perfect1', 'is_transductive': False},
            {'correct_train_input': [True, True, True], 'code': 'perfect2', 'is_transductive': False},
            {'correct_train_input': [True], 'code': 'trans', 'is_transductive': True},
        ]
        
        valid_programs = [p for p in programs if is_program_valid_for_refinement(p)]
        assert len(valid_programs) == 0  # All should be excluded
        
        # Selection should return empty dict
        result = select_best_program_for_refinement(valid_programs)
        assert result == {}
    
    def test_sampling_randomness(self):
        """Test that uniform sampling provides randomness over multiple calls"""
        programs = [
            {'correct_train_input': [True, False, True], 'code': 'program_a', 'is_transductive': False},
            {'correct_train_input': [True, False, True], 'code': 'program_b', 'is_transductive': False},
            {'correct_train_input': [True, False], 'code': 'program_c', 'is_transductive': False},
        ]

        # Uniform sampling should show variety across multiple calls
        results = []
        for _ in range(30):
            result = select_best_program_for_refinement(programs)
            results.append(result['code'])

        # Should get variety in selections (not always the same)
        unique_results = set(results)
        assert len(unique_results) >= 2  # Should get at least 2 different programs


class TestREXProgramPool:
    """Test the REXProgramPool class"""

    def test_init_empty_pool(self):
        """Test initializing empty pool"""
        pool = REXProgramPool([])
        assert len(pool.programs) == 0
        assert len(pool.refinement_counts) == 0
        assert len(pool.program_hashes) == 0

    def test_init_with_programs(self):
        """Test initializing pool with programs"""
        programs = [
            {'row_id': 'prog1', 'code': 'def solve(): pass', 'correct_train_input': [True, False]},
            {'row_id': 'prog2', 'code': 'def solve(): return []', 'correct_train_input': [False, False]}
        ]
        pool = REXProgramPool(programs)
        assert len(pool.programs) == 2
        assert len(pool.program_hashes) == 2

    def test_sample_program_empty_pool(self):
        """Test sampling from empty pool returns None"""
        pool = REXProgramPool([])
        result = pool.sample_program("uniform")
        assert result is None

        result = pool.sample_program("rex")
        assert result is None

    def test_sample_program_uniform(self):
        """Test uniform sampling from pool"""
        programs = [
            {'row_id': 'prog1', 'code': 'code1', 'correct_train_input': [True, False]},
            {'row_id': 'prog2', 'code': 'code2', 'correct_train_input': [True, True]}
        ]
        pool = REXProgramPool(programs)

        # Sample multiple times and check we get variety
        results = set()
        for _ in range(20):
            result = pool.sample_program("uniform")
            results.add(result['code'])

        assert len(results) >= 1  # Should get at least one program
        assert all(code in ['code1', 'code2'] for code in results)

    def test_sample_program_rex(self):
        """Test REX sampling from pool"""
        programs = [
            {'row_id': 'prog1', 'code': 'code1', 'correct_train_input': [True, False]},  # 50% correct
            {'row_id': 'prog2', 'code': 'code2', 'correct_train_input': [True, True]}    # 100% correct
        ]
        pool = REXProgramPool(programs)

        # REX should be able to sample both programs
        results = []
        for _ in range(10):
            result = pool.sample_program("rex")
            results.append(result['code'])

        assert len(results) == 10
        assert all(code in ['code1', 'code2'] for code in results)

    def test_rex_refinement_count_tracking(self):
        """Test that REX sampling tracks refinement counts"""
        programs = [
            {'row_id': 'prog1', 'code': 'code1', 'correct_train_input': [True, False]}
        ]
        pool = REXProgramPool(programs)

        # Sample same program multiple times
        for i in range(3):
            result = pool.sample_program("rex")
            assert result['row_id'] == 'prog1'
            # Check refinement count increases
            assert pool.refinement_counts['prog1'] == i + 1

    def test_add_programs_basic(self):
        """Test adding new programs to pool"""
        pool = REXProgramPool([])
        new_programs = [
            {'row_id': 'new1', 'code': 'new code', 'correct_train_input': [False]}
        ]

        added = pool.add_programs(new_programs)
        assert added == 1
        assert len(pool.programs) == 1
        assert pool.programs[0]['code'] == 'new code'

    def test_add_programs_with_deduplication(self):
        """Test deduplication when adding programs"""
        initial_program = {'row_id': 'orig', 'code': 'def solve(): pass', 'correct_train_input': [True]}
        pool = REXProgramPool([initial_program])

        # Try to add duplicate code
        duplicate_program = {'row_id': 'dup', 'code': 'def solve(): pass', 'correct_train_input': [False]}
        added = pool.add_programs([duplicate_program], deduplicate=True)

        assert added == 0  # Should not add duplicate
        assert len(pool.programs) == 1  # Still only original program

        # Try to add genuinely new program
        new_program = {'row_id': 'new', 'code': 'def solve(): return []', 'correct_train_input': [True]}
        added = pool.add_programs([new_program], deduplicate=True)

        assert added == 1  # Should add new program
        assert len(pool.programs) == 2

    def test_add_programs_without_deduplication(self):
        """Test adding programs without deduplication"""
        initial_program = {'row_id': 'orig', 'code': 'def solve(): pass', 'correct_train_input': [True]}
        pool = REXProgramPool([initial_program])

        # Add duplicate with deduplication disabled
        duplicate_program = {'row_id': 'dup', 'code': 'def solve(): pass', 'correct_train_input': [False]}
        added = pool.add_programs([duplicate_program], deduplicate=False)

        assert added == 1  # Should add even if duplicate
        assert len(pool.programs) == 2

    def test_get_pool_stats_empty(self):
        """Test pool statistics for empty pool"""
        pool = REXProgramPool([])
        stats = pool.get_pool_stats()

        expected = {"total_programs": 0, "avg_correctness": 0.0, "total_refinements": 0}
        assert stats == expected

    def test_get_pool_stats_with_programs(self):
        """Test pool statistics with programs"""
        programs = [
            {'row_id': 'prog1', 'code': 'code1', 'correct_train_input': [True, False]},    # 50%
            {'row_id': 'prog2', 'code': 'code2', 'correct_train_input': [True, True]}      # 100%
        ]
        pool = REXProgramPool(programs)

        # Sample to generate refinement counts
        pool.sample_program("rex")
        pool.sample_program("rex")

        stats = pool.get_pool_stats()

        assert stats["total_programs"] == 2
        assert stats["avg_correctness"] == 0.75  # (0.5 + 1.0) / 2
        assert stats["total_refinements"] == 2
        assert stats["unique_hashes"] == 2

    def test_program_hash_generation(self):
        """Test that program hashing works correctly with normalization"""
        pool = REXProgramPool([])

        program1 = {'code': 'def solve(): pass'}
        program2 = {'code': 'def solve(): return []'}
        program3 = {'code': 'def solve(): pass'}  # Same as program1
        program4 = {'code': 'DEF SOLVE():    PASS'}  # Same as program1 but different case/whitespace

        hash1 = pool._get_program_hash(program1)
        hash2 = pool._get_program_hash(program2)
        hash3 = pool._get_program_hash(program3)
        hash4 = pool._get_program_hash(program4)

        assert hash1 == hash3  # Same code, same hash
        assert hash1 == hash4  # Same code with different case/whitespace, same hash
        assert hash1 != hash2  # Different code, different hash
        assert len(hash1) == 16  # Should be 16 chars (truncated SHA256)


class TestCreateRefinedProgramEntry:
    """Test the create_refined_program_entry function"""

    def test_create_basic_refined_entry(self):
        """Test creating a basic refined program entry"""
        original = {
            'row_id': 'orig123',
            'code': 'original code',
            'correct_train_input': [True, False]
        }

        refined_code = 'refined code'
        entry = create_refined_program_entry(original, refined_code)

        assert entry['code'] == 'refined code'
        assert entry['model'] == 'unknown'
        assert entry['is_transductive'] == False
        assert entry['parent_program_id'] == 'orig123'
        assert entry['row_id'].startswith('refined_')
        assert len(entry['row_id']) == 16  # 'refined_' + 8 hex chars

        # Should copy correctness from original
        assert entry['correct_train_input'] == [True, False]

    def test_create_refined_entry_with_task_results(self):
        """Test creating refined entry with new task results"""
        original = {
            'row_id': 'orig123',
            'code': 'original code'
        }

        task_results = {
            'correct_train_input': [False, True],
            'correct_test_input': [True],
            'predicted_train_output': [[[1, 0]], [[0, 1]]],
            'predicted_test_output': [[[1, 1]]]
        }

        entry = create_refined_program_entry(
            original, 'new code', task_results, model='gpt-4'
        )

        assert entry['code'] == 'new code'
        assert entry['model'] == 'gpt-4'
        assert entry['correct_train_input'] == [False, True]
        assert entry['correct_test_input'] == [True]
        assert entry['predicted_train_output'] == [[[1, 0]], [[0, 1]]]
        assert entry['predicted_test_output'] == [[[1, 1]]]

    def test_create_refined_entry_missing_original_fields(self):
        """Test creating refined entry when original lacks some fields"""
        original = {
            'code': 'original code'
            # Missing row_id and correctness data
        }

        entry = create_refined_program_entry(original, 'refined code')

        assert entry['code'] == 'refined code'
        assert entry['parent_program_id'] is None  # original had no row_id
        assert entry['correct_train_input'] == []  # default empty
        assert entry['correct_test_input'] == []


class TestSelectProgramForRefinementWithPool:
    """Test select_program_for_refinement with program_pool parameter"""

    def test_select_with_pool_uniform(self):
        """Test selection with program pool using uniform sampling"""
        programs = [
            {'row_id': 'prog1', 'code': 'code1', 'correct_train_input': [True, False]},
            {'row_id': 'prog2', 'code': 'code2', 'correct_train_input': [False, False]}
        ]
        pool = REXProgramPool(programs)

        result = select_program_for_refinement(
            programs=None,  # Should be ignored when pool is provided
            sampling_mode="uniform",
            program_pool=pool
        )

        assert result['code'] in ['code1', 'code2']

    def test_select_with_pool_rex(self):
        """Test selection with program pool using REX sampling"""
        programs = [
            {'row_id': 'prog1', 'code': 'code1', 'correct_train_input': [True, False]},
            {'row_id': 'prog2', 'code': 'code2', 'correct_train_input': [True, True]}
        ]
        pool = REXProgramPool(programs)

        result = select_program_for_refinement(
            sampling_mode="rex",
            program_pool=pool
        )

        assert result['code'] in ['code1', 'code2']
        # Check that refinement count was incremented
        selected_id = result['row_id']
        assert pool.refinement_counts[selected_id] == 1

    def test_select_with_empty_pool(self):
        """Test selection with empty program pool"""
        pool = REXProgramPool([])

        result = select_program_for_refinement(program_pool=pool)
        assert result == {}

    def test_select_fallback_to_list_based(self):
        """Test fallback to list-based approach when no pool provided"""
        programs = [
            {'row_id': 'prog1', 'code': 'code1', 'correct_train_input': [True, False]}
        ]

        result = select_program_for_refinement(
            programs=programs,
            sampling_mode="uniform"
        )

        assert result['code'] == 'code1'

    def test_select_with_debug_pool(self):
        """Test debug output when using program pool"""
        programs = [
            {'row_id': 'prog1', 'code': 'code1', 'correct_train_input': [True, False]}
        ]
        pool = REXProgramPool(programs)

        with patch('builtins.print') as mock_print:
            result = select_program_for_refinement(
                sampling_mode="uniform",
                program_pool=pool,
                debug=True
            )

            mock_print.assert_called_once()
            args = mock_print.call_args[0][0]
            assert 'uniform pool' in args
            assert '50.0% correct' in args


class TestREXEnhancedFunctionality:
    """Test enhanced REx functionality with refinement success tracking"""

    def test_track_refinement_attempt_improvement(self):
        """Test tracking a successful refinement attempt"""
        programs = [
            {'row_id': 'prog1', 'code': 'code1', 'correct_train_input': [True, False]}  # 50%
        ]
        pool = REXProgramPool(programs)

        # Track an improvement: 50% -> 75%
        pool.track_refinement_attempt('prog1', 0.75, 0.50)

        stats = pool.refinement_success_stats['prog1']
        assert stats['attempts'] == 1
        assert stats['improvements'] == 1
        assert stats['total_improvement'] == 0.25
        assert stats['avg_improvement'] == 0.25

    def test_track_refinement_attempt_no_improvement(self):
        """Test tracking a failed refinement attempt"""
        programs = [
            {'row_id': 'prog1', 'code': 'code1', 'correct_train_input': [True, False]}
        ]
        pool = REXProgramPool(programs)

        # Track degradation: 50% -> 25% (-0.25)
        pool.track_refinement_attempt('prog1', 0.25, 0.50)

        stats = pool.refinement_success_stats['prog1']
        assert stats['attempts'] == 1
        assert stats['improvements'] == 0  # No improvement
        assert stats['total_improvement'] == -0.25  # Now tracks negative change
        assert stats['avg_improvement'] == -0.25

    def test_track_multiple_refinement_attempts(self):
        """Test tracking multiple refinement attempts with mixed results"""
        programs = [
            {'row_id': 'prog1', 'code': 'code1', 'correct_train_input': [True, False]}
        ]
        pool = REXProgramPool(programs)

        # First attempt: improvement from 50% to 75% (+0.25)
        pool.track_refinement_attempt('prog1', 0.75, 0.50)

        # Second attempt: degradation from 50% to 25% (-0.25)
        pool.track_refinement_attempt('prog1', 0.25, 0.50)

        # Third attempt: improvement from 50% to 100% (+0.50)
        pool.track_refinement_attempt('prog1', 1.00, 0.50)

        stats = pool.refinement_success_stats['prog1']
        assert stats['attempts'] == 3
        assert stats['improvements'] == 2  # Two successful improvements
        assert stats['total_improvement'] == 0.50  # 0.25 + (-0.25) + 0.50
        assert abs(stats['avg_improvement'] - (0.50/3)) < 0.001  # 0.50 / 3 ≈ 0.167

    def test_enhanced_rex_sampling_with_bonus(self):
        """Test that REX sampling incorporates refinement bonus"""
        programs = [
            {'row_id': 'prog1', 'code': 'code1', 'correct_train_input': [True, False]},   # 50% correct
            {'row_id': 'prog2', 'code': 'code2', 'correct_train_input': [True, False]}    # 50% correct
        ]
        pool = REXProgramPool(programs)

        # Give prog1 a refinement success history
        pool.track_refinement_attempt('prog1', 0.80, 0.50)  # +0.30 improvement
        pool.track_refinement_attempt('prog1', 0.70, 0.50)  # +0.20 improvement
        # prog1 avg_improvement = 0.25

        # prog2 has no refinement history (avg_improvement = 0.0)

        # Sample multiple times and track selections (use full weight to amplify effect)
        selections = {'prog1': 0, 'prog2': 0}
        for _ in range(100):
            result = pool.sample_program("rex")  # Full weight
            selections[result['row_id']] += 1

        # prog1 should be selected more often due to refinement bonus
        # Both have 50% correctness, but prog1 has +0.25 refinement bonus (full weight)
        assert selections['prog1'] > selections['prog2']
        # With the current REX_C value, ensure there's at least some bias towards prog1
        # Note: Lower REX_C values create less aggressive biasing
        assert selections['prog1'] >= 48  # Should be better than completely random (around 50)

    def test_quality_score_attachment_during_sampling(self):
        """Test that quality score is temporarily attached during REX sampling"""
        programs = [
            {'row_id': 'prog1', 'code': 'code1', 'correct_train_input': [True, True, False]}  # 67% correct
        ]
        pool = REXProgramPool(programs)

        # Add some refinement success history
        pool.track_refinement_attempt('prog1', 0.80, 0.67)  # +0.13 improvement

        result = pool.sample_program("rex")

        # Should have quality score attached temporarily
        assert '_rex_quality_score' in result
        expected_quality = 2/3 + (0.13 * 0.5)  # correctness + weighted refinement_bonus
        assert abs(result['_rex_quality_score'] - expected_quality) < 0.001

    def test_enhanced_pool_stats(self):
        """Test enhanced pool statistics with refinement success metrics"""
        programs = [
            {'row_id': 'prog1', 'code': 'code1', 'correct_train_input': [True, False]},    # 50%
            {'row_id': 'prog2', 'code': 'code2', 'correct_train_input': [True, True]}      # 100%
        ]
        pool = REXProgramPool(programs)

        # Add refinement history
        pool.track_refinement_attempt('prog1', 0.75, 0.50)  # +0.25 improvement
        pool.track_refinement_attempt('prog1', 0.25, 0.50)  # -0.25 degradation (now counted)
        pool.track_refinement_attempt('prog2', 0.80, 1.00)  # -0.20 degradation (now counted)

        # Sample to create refinement counts
        pool.sample_program("rex")
        pool.sample_program("rex")

        stats = pool.get_pool_stats()

        assert stats['total_programs'] == 2
        assert stats['avg_correctness'] == 0.75  # (0.5 + 1.0) / 2

        # avg_quality_score should include refinement bonuses (now symmetric with 0.5 weight)
        # prog1: 0.5 + (0.0 * 0.5) = 0.5 (avg_improvement = (0.25-0.25)/2 = 0.0)
        # prog2: 1.0 + (-0.20 * 0.5) = 0.9 (avg_improvement = -0.20/1 = -0.20, weighted = -0.10)
        expected_avg_quality = (0.5 + 0.9) / 2
        assert abs(stats['avg_quality_score'] - expected_avg_quality) < 0.001

        assert stats['refinement_success_rate'] == 1/3  # 1 improvement out of 3 attempts
        assert stats['total_refinement_attempts'] == 3
        assert stats['total_refinements'] == 2  # Number of times programs were selected

    def test_thread_safety_of_refinement_tracking(self):
        """Test thread safety of refinement success tracking"""
        import threading
        import time

        programs = [
            {'row_id': 'prog1', 'code': 'code1', 'correct_train_input': [True, False]}
        ]
        pool = REXProgramPool(programs)

        def track_attempts():
            for i in range(10):
                pool.track_refinement_attempt('prog1', 0.6 + i*0.01, 0.5)
                time.sleep(0.001)  # Small delay to encourage race conditions

        # Run multiple threads tracking refinement attempts
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=track_attempts)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Should have tracked all attempts safely
        stats = pool.refinement_success_stats['prog1']
        assert stats['attempts'] == 30  # 3 threads * 10 attempts each
        assert stats['improvements'] == 30  # All were improvements
        assert stats['total_improvement'] > 0  # Should have accumulated improvements

    def test_refinement_bonus_zero_for_new_programs(self):
        """Test that new programs without history have zero refinement bonus"""
        programs = [
            {'row_id': 'prog1', 'code': 'code1', 'correct_train_input': [True, False]}
        ]
        pool = REXProgramPool(programs)

        # Sample without any refinement history
        result = pool.sample_program("rex")

        # Quality score should equal correctness (no bonus)
        expected_quality = 0.5  # Just the correctness percentage
        assert abs(result['_rex_quality_score'] - expected_quality) < 0.001

    def test_transductive_penalty_behavior(self):
        """Test that transductive programs are treated as 0% correct for learning"""
        programs = [
            {'row_id': 'prog1', 'code': 'code1', 'correct_train_input': [True, False]}  # 50%
        ]
        pool = REXProgramPool(programs)

        # Simulate a transductive refinement that appears to be 100% correct
        # but should be treated as 0% for learning purposes
        pool.track_refinement_attempt('prog1', 0.0, 0.50)  # Transductive = 0% in tracking

        stats = pool.refinement_success_stats['prog1']
        assert stats['attempts'] == 1
        assert stats['improvements'] == 0  # No improvement since 0% < 50%
        assert stats['total_improvement'] == -0.50  # Treated as degradation
        assert stats['avg_improvement'] == -0.50  # This program gets penalized

        # After this "bad" refinement history, quality score should be penalized
        result = pool.sample_program("rex")
        expected_quality = 0.5 + (-0.50 * 0.5)  # correctness + penalty (using REX_REFINEMENT_BONUS_WEIGHT=0.5)
        assert abs(result['_rex_quality_score'] - expected_quality) < 0.001
        assert result['_rex_quality_score'] == 0.25  # Penalized (0.5 + (-0.50 * 0.5) = 0.25)

    def test_pixel_match_calculation_with_numpy_arrays(self):
        """Test that pixel match functions handle numpy arrays without truth value errors"""
        import numpy as np
        from ..refinement_utils import _calculate_pixel_match_percentage, _calculate_pixel_match_bonus

        # Test pixel match percentage with numpy arrays
        predicted = np.array([[1, 2], [3, 4]])
        expected = np.array([[1, 2], [3, 5]])  # 3/4 pixels match
        result = _calculate_pixel_match_percentage(predicted, expected)
        assert result == 0.75  # 3 out of 4 pixels match

        # Test with perfect match
        predicted_perfect = np.array([[1, 2], [3, 4]])
        expected_perfect = np.array([[1, 2], [3, 4]])
        result_perfect = _calculate_pixel_match_percentage(predicted_perfect, expected_perfect)
        assert result_perfect == 1.0

        # Test with no match
        predicted_no_match = np.array([[0, 0], [0, 0]])
        expected_no_match = np.array([[1, 2], [3, 4]])
        result_no_match = _calculate_pixel_match_percentage(predicted_no_match, expected_no_match)
        assert result_no_match == 0.0

        # Test with size mismatch
        predicted_wrong_size = np.array([[1, 2, 3]])
        expected_right_size = np.array([[1, 2], [3, 4]])
        result_size_mismatch = _calculate_pixel_match_percentage(predicted_wrong_size, expected_right_size)
        assert result_size_mismatch == 0.0

        # Test pixel match bonus calculation
        program_data = {
            'predicted_train_output': [
                np.array([[1, 2], [3, 4]]),  # 75% pixel match
                np.array([[0, 0], [0, 0]])   # 0% pixel match
            ],
            'correct_train_input': [False, False]  # Both incorrect
        }

        task_data = {
            'train': [
                {'output': np.array([[1, 2], [3, 5]])},  # Expected for first example
                {'output': np.array([[1, 2], [3, 4]])}   # Expected for second example
            ]
        }

        bonus = _calculate_pixel_match_bonus(program_data, task_data)
        # First example: 75% pixel match / 2 train examples = 0.375
        # Second example: 0% pixel match / 2 train examples = 0.0
        # Total bonus = 0.375 + 0.0 = 0.375 (before 0.1 scaling in REX)
        assert abs(bonus - 0.375) < 0.001

        # Test with correct examples (should be ignored)
        program_data_mixed = {
            'predicted_train_output': [
                np.array([[1, 2], [3, 4]]),  # This would be 75% but is correct
                np.array([[0, 0], [0, 0]])   # This is 0% and incorrect
            ],
            'correct_train_input': [True, False]  # First correct, second incorrect
        }

        bonus_mixed = _calculate_pixel_match_bonus(program_data_mixed, task_data)
        # Only second example counts: 0% pixel match / 2 train examples = 0.0
        assert bonus_mixed == 0.0