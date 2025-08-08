import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from llm_python.utils.scoring import GridScorer
from llm_python.progdb.arc_tester import ArcTester

class TestGridScorer:
    """Basic tests for GridScorer"""
    
    def test_perfect_match(self):
        """Test perfect grid match"""
        scorer = GridScorer()
        predicted = [[1, 2], [3, 4]]
        actual = [[1, 2], [3, 4]]
        
        result = scorer.score_grid(predicted, actual)
        assert result['correct'] is True
        assert result['pixel_accuracy'] == 1.0
        assert result['error'] is None
    
    def test_partial_match(self):
        """Test partial grid match"""
        scorer = GridScorer()
        predicted = [[1, 2], [3, 5]]  # One wrong cell
        actual = [[1, 2], [3, 4]]
        
        result = scorer.score_grid(predicted, actual)
        assert result['correct'] is False
        assert result['pixel_accuracy'] == 0.75
        assert result['error'] is None
    
    def test_dimension_mismatch(self):
        """Test grid dimension mismatch"""
        scorer = GridScorer()
        predicted = [[1, 2]]
        actual = [[1, 2], [3, 4]]
        
        result = scorer.score_grid(predicted, actual)
        assert result['correct'] is False
        assert result['error'] == 'Grid height mismatch'


class TestArcTester:
    """Basic tests for ArcTester using real executor"""
    
    def test_simple_transform(self):
        """Test executing a simple transform function"""
        executor = ArcTester(timeout=1.0, executor_type="unrestricted")
        
        program = """
def transform(grid):
    return [[cell + 1 for cell in row] for row in grid]
"""
        test_input = [[1, 2], [3, 4]]
        
        result, error, timed_out = executor.execute_program_with_timeout(program, test_input)
        
        assert result == [[2, 3], [4, 5]]
        assert error == ""
        assert timed_out is False
    
    def test_solve_function(self):
        """Test executing a solve function"""
        executor = ArcTester(timeout=1.0, executor_type="unrestricted")
        
        program = """
def solve(grid):
    return [[0 for _ in row] for row in grid]
"""
        test_input = [[1, 2], [3, 4]]
        
        result, error, timed_out = executor.execute_program_with_timeout(program, test_input)
        
        assert result == [[0, 0], [0, 0]]
        assert error == ""
        assert timed_out is False
    
    def test_syntax_error(self):
        """Test program with syntax error"""
        executor = ArcTester(timeout=1.0, executor_type="unrestricted")
        
        program = """
def transform(grid):
    return [[cell + for cell in row] for row in grid]  # Syntax error
"""
        test_input = [[1, 2]]
        
        result, error, timed_out = executor.execute_program_with_timeout(program, test_input)
        
        assert result is None
        assert error != ""
        assert timed_out is False
    
    def test_no_function(self):
        """Test program with no valid function"""
        executor = ArcTester(timeout=1.0, executor_type="unrestricted")
        
        program = "x = 5"  # No function
        test_input = [[1, 2]]
        
        result, error, timed_out = executor.execute_program_with_timeout(program, test_input)
        
        assert result is None
        assert "No valid transformation function found" in error
        assert timed_out is False
    
    @classmethod
    def teardown_class(cls):
        """Cleanup the executor after tests"""
        ArcTester.cleanup_executor()
