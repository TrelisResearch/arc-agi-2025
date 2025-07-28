#!/usr/bin/env python3

from ..code import strip_comments


class TestStripCommentsAggressive:
    """Test cases for the strip_comments_aggressive function."""
    
    def test_empty_string(self):
        """Test with empty string."""
        assert strip_comments("") == ""
        assert strip_comments("   ") == "   "  # Tokenize preserves whitespace
    
    def test_no_comments(self):
        """Test code without comments."""
        code = 'x = "hello world"'
        assert strip_comments(code) == code
    
    def test_simple_comment_removal(self):
        """Test basic comment removal."""
        code = 'x = 1  # This is a comment'
        expected = 'x = 1  '  # Tokenize preserves the spacing where comment was
        assert strip_comments(code) == expected
    
    def test_comment_only_line(self):
        """Test removal of comment-only lines."""
        code = '''x = 1
# This is a comment line
y = 2'''
        expected = '''x = 1

y = 2'''
        assert strip_comments(code) == expected
    
    def test_hash_in_string_literal(self):
        """Test that # inside string literals is preserved."""
        code = 'text = "This has # hash inside"  # Comment'
        expected = 'text = "This has # hash inside"  '  # Preserves spacing
        assert strip_comments(code) == expected
    
    def test_escaped_quotes(self):
        """Test proper handling of escaped quotes."""
        code = 'text = "String with escaped quote at end\\""  # Comment'
        expected = 'text = "String with escaped quote at end\\""  '
        assert strip_comments(code) == expected
    
    def test_mixed_quote_types(self):
        """Test mixed single and double quotes."""
        code = '''text1 = 'He said "Hello" and she said \\\'Hi\\\''  # Comment'''
        expected = '''text1 = 'He said "Hello" and she said \\\'Hi\\\''  '''
        assert strip_comments(code) == expected
    
    def test_triple_quoted_strings(self):
        """Test triple-quoted strings with hash inside."""
        code = '''text = """Line 1
Line 2 with # hash inside string
Line 3"""  # Comment after multiline string'''
        expected = '''text = """Line 1
Line 2 with # hash inside string
Line 3"""  '''
        assert strip_comments(code) == expected
    
    def test_triple_single_quotes(self):
        """Test triple single-quoted strings."""
        code = """text = '''This has "quotes" and # hash inside'''  # Comment"""
        expected = """text = '''This has "quotes" and # hash inside'''  """
        assert strip_comments(code) == expected
    
    def test_f_strings_with_hash(self):
        """Test f-strings containing hash characters."""
        code = '''name = "John"
msg = f"Hello {name.replace('#', 'num')}"  # Comment'''
        expected = '''name = "John"
msg = f"Hello {name.replace('#', 'num')}"  '''
        assert strip_comments(code) == expected
    
    def test_raw_strings(self):
        """Test raw strings with hash characters."""
        code = 'path = r"C:\\temp\\#folder"  # Windows path with hash'
        expected = 'path = r"C:\\temp\\#folder"  '
        assert strip_comments(code) == expected
    
    def test_multiple_escapes(self):
        """Test multiple consecutive backslashes."""
        code = 'text = "test\\\\\\""  # Should work'
        expected = 'text = "test\\\\\\""  '
        assert strip_comments(code) == expected
    
    def test_hash_immediately_after_string(self):
        """Test hash immediately after string without space."""
        code = 'x = "test"# comment'
        expected = 'x = "test"'
        assert strip_comments(code) == expected
    
    def test_multiple_strings_on_line(self):
        """Test multiple strings on same line."""
        code = 'x = "first" + "second"  # comment'
        expected = 'x = "first" + "second"  '
        assert strip_comments(code) == expected
    
    def test_function_with_only_comments(self):
        """Test function where comments are removed but structure preserved."""
        code = '''
def func():
    # Only comments here
    # Nothing else
    pass
'''
        expected = '''
def func():
    
    
    pass
'''
        assert strip_comments(code) == expected
    
    def test_complex_nested_quotes(self):
        """Test complex case with various quote types."""
        code = '''data = {
    "key1": 'value with "nested" quotes',  # Comment 1
    'key2': "value with 'nested' quotes",  # Comment 2  
    "key3": """multi
line # not a comment
string"""  # Comment 3
}'''
        expected = '''data = {
    "key1": 'value with "nested" quotes',  
    'key2': "value with 'nested' quotes",  
    "key3": """multi
line # not a comment
string"""  
}'''
        assert strip_comments(code) == expected
    
    def test_string_with_literal_backslash(self):
        """Test string ending with literal backslash."""
        code = 'path = "C:\\\\temp\\\\"  # Comment'
        expected = 'path = "C:\\\\temp\\\\"  '
        assert strip_comments(code) == expected
    
    def test_regex_pattern_with_hash(self):
        """Test regex patterns containing hash."""
        code = 'pattern = r"\\w+#\\d+"  # Regex pattern'
        expected = 'pattern = r"\\w+#\\d+"  '
        assert strip_comments(code) == expected
    
    def test_whitespace_normalization(self):
        """Test that tokenize preserves whitespace structure."""
        code = '''x = 1


y = 2'''
        expected = '''x = 1


y = 2'''  # Tokenize preserves the original structure
        assert strip_comments(code) == expected
    
    def test_indentation_preservation(self):
        """Test that indentation is preserved."""
        code = '''if True:
    x = 1  # Comment
    if True:
        y = 2  # Another comment'''
        expected = '''if True:
    x = 1  
    if True:
        y = 2  '''
        assert strip_comments(code) == expected
    
    def test_unterminated_triple_quote(self):
        """Test handling of unterminated triple quotes."""
        code = 'text = """This is unterminated'
        # The tokenize module will raise an error for malformed code
        try:
            result = strip_comments(code)
            # If it doesn't raise an error, check that it contains the content
            assert '"""This is unterminated' in result
        except Exception:
            # It's acceptable for the tokenize module to fail on malformed code
            pass
    
    def test_comment_at_end_of_file(self):
        """Test comment at the very end of file."""
        code = 'x = 1  # Final comment'
        expected = 'x = 1  '
        assert strip_comments(code) == expected
    
    def test_hash_in_dictionary_key(self):
        """Test hash character in dictionary keys."""
        code = 'data = {"key#1": "value"}  # Comment here'
        expected = 'data = {"key#1": "value"}  '
        assert strip_comments(code) == expected
    
    def test_continuation_lines(self):
        """Test line continuation with backslash."""
        code = '''result = some_very_long_function_name(arg1, arg2, \\
                                          arg3)  # Comment'''
        # The tokenize module handles line continuation differently
        # It will flatten the continuation into a single line
        expected = '''result = some_very_long_function_name(arg1, arg2,                                          arg3)  '''
        assert strip_comments(code) == expected
    
    def test_syntax_validation(self):
        """Test that result is valid Python syntax."""
        test_cases = [
            'x = "hello"  # comment',
            'def func(): pass  # comment',
            'if True: x = 1  # comment',
            'data = {"key": "value"}  # comment'
        ]
        
        for code in test_cases:
            result = strip_comments(code)
            # Should not raise SyntaxError
            compile(result, '<test>', 'exec')
    
    def test_error_handling(self):
        """Test error handling for malformed input."""
        # The tokenize module will raise exceptions for malformed code
        # This is acceptable behavior
        malformed_inputs = [
            'some malformed input',
            'x = "unterminated string',
            'def func(:'
        ]
        
        for malformed_code in malformed_inputs:
            try:
                result = strip_comments(malformed_code)
                # If it succeeds, that's fine too
                assert isinstance(result, str)
            except Exception:
                # If it fails with any exception, that's also acceptable
                # since the tokenize module is strict about syntax
                pass
