#!/usr/bin/env python3

import pytest
import random
from unittest.mock import patch
from llm_python.utils.refinement_utils import (
    calculate_program_metrics,
    select_best_program_for_refinement,
    is_program_valid_for_refinement
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
    
    def test_select_ranking_by_correctness(self):
        """Test that higher correctness programs are preferred"""
        programs = [
            {'correct_train_input': [True, False], 'code': 'low'},     # 50%
            {'correct_train_input': [True, True, False], 'code': 'high'},  # 67%
            {'correct_train_input': [False, False], 'code': 'zero'}   # 0%
        ]
        
        # With top_k=1, should always select the highest scoring program
        with patch('random.choice', side_effect=lambda x: x[0]):  # Always pick first from list
            result = select_best_program_for_refinement(programs, top_k=1)
            assert result['code'] == 'high'  # 67% correctness
    
    def test_select_tie_breaking_by_length(self):
        """Test tie-breaking by code length (shorter preferred)"""
        programs = [
            {'correct_train_input': [True, False], 'code': 'this is longer code'},  # 50%, long
            {'correct_train_input': [True, False], 'code': 'short'},              # 50%, short
        ]
        
        # With top_k=1, should always select the shorter program when correctness is tied
        with patch('random.choice', side_effect=lambda x: x[0]):
            result = select_best_program_for_refinement(programs, top_k=1)
            assert result['code'] == 'short'
    
    def test_select_top_k_selection(self):
        """Test that only top-k programs are considered"""
        programs = [
            {'correct_train_input': [True, True], 'code': 'best'},      # 100%
            {'correct_train_input': [True, False], 'code': 'second'},   # 50%  
            {'correct_train_input': [False, False], 'code': 'worst'},   # 0%
        ]
        
        # With top_k=2, should never select the worst program
        random.seed(42)  # Set seed for reproducible test
        results = set()
        for _ in range(100):  # Run many times to see all possible selections
            result = select_best_program_for_refinement(programs, top_k=2)
            results.add(result['code'])
        
        assert 'best' in results
        assert 'second' in results
        assert 'worst' not in results  # Should never be selected
    
    def test_select_debug_output(self):
        """Test debug output"""
        programs = [{'correct_train_input': [True, False], 'code': 'test'}]
        
        with patch('builtins.print') as mock_print:
            select_best_program_for_refinement(programs, debug=True)
            mock_print.assert_called_once()
            args = mock_print.call_args[0][0]
            assert '50.0% correct' in args
            assert '4 chars' in args
    
    def test_select_complex_ranking(self):
        """Test complex ranking scenario"""
        programs = [
            {'correct_train_input': [True, True, True], 'code': 'a'},        # 100%, 1 char
            {'correct_train_input': [True, True, False], 'code': 'bb'},      # 67%, 2 chars
            {'correct_train_input': [True, True, False], 'code': 'ccc'},     # 67%, 3 chars  
            {'correct_train_input': [True, False], 'code': 'dddd'},          # 50%, 4 chars
            {'correct_train_input': [False], 'code': 'eeeee'},               # 0%, 5 chars
        ]
        
        # Expected order: 100%/1char, 67%/2char, 67%/3char, 50%/4char, 0%/5char
        with patch('random.choice', side_effect=lambda x: x[0]):
            # Top 1: should be 100%
            result = select_best_program_for_refinement(programs, top_k=1)
            assert result['code'] == 'a'
            
        # Top 3: test without mocking random.choice to see actual selection
        results = set()
        for _ in range(50):
            result = select_best_program_for_refinement(programs, top_k=3)  
            results.add(result['code'])
        
        assert 'a' in results    # 100%
        assert 'bb' in results   # 67%, shorter
        assert 'ccc' in results  # 67%, longer
        assert 'dddd' not in results  # 50% - excluded
        assert 'eeeee' not in results  # 0% - excluded


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
    
    def test_ranking_consistency(self):
        """Test that ranking is consistent and deterministic given same input"""
        programs = [
            {'correct_train_input': [True, False, True], 'code': 'a' * 10, 'is_transductive': False},    # 67%, 10 chars
            {'correct_train_input': [True, False, True], 'code': 'b' * 5, 'is_transductive': False},     # 67%, 5 chars  
            {'correct_train_input': [True, False], 'code': 'c' * 15, 'is_transductive': False},          # 50%, 15 chars
        ]
        
        # With top_k=1, should always get the same result (shortest among highest correctness)
        with patch('random.choice', side_effect=lambda x: x[0]):
            result1 = select_best_program_for_refinement(programs, top_k=1)
            result2 = select_best_program_for_refinement(programs, top_k=1)
            result3 = select_best_program_for_refinement(programs, top_k=1)
            
            assert result1 == result2 == result3
            assert result1['code'] == 'b' * 5  # Shortest among 67% correct programs