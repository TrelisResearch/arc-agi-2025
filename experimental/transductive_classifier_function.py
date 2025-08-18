import ast
import re
import numpy as np

def classify_transductive_program(program: str) -> tuple[bool, float]:
    """
    Classify whether an ARC program is transductive or inductive.

    Args:
        program: The program code as a string

    Returns:
        tuple: (is_transductive: bool, confidence: float)
    """

    def extract_literals_ast(program_code):
        """Extract numeric literals using AST parsing"""
        try:
            tree = ast.parse(program_code)
            literals = []

            class LiteralVisitor(ast.NodeVisitor):
                def visit_Constant(self, node):
                    if isinstance(node.value, (int, float)):
                        literals.append(node.value)
                    self.generic_visit(node)

                def visit_Num(self, node):
                    literals.append(node.n)
                    self.generic_visit(node)

            visitor = LiteralVisitor()
            visitor.visit(tree)
            return literals
        except SyntaxError:
            numeric_literals = re.findall(r'\b\d+(?:\.\d+)?\b', program_code)
            return [float(lit) if '.' in lit else int(lit) for lit in numeric_literals]

    def extract_ast_features(program_code):
        """Extract AST structural features"""
        try:
            tree = ast.parse(program_code)
            counts = {
                'function_definitions': 0, 'for_loops': 0, 'while_loops': 0,
                'if_statements': 0, 'assignments': 0, 'comparisons': 0,
                'binary_ops': 0, 'subscripts': 0, 'method_calls': 0,
                'list_comprehensions': 0, 'lambda_functions': 0,
                'try_statements': 0, 'return_statements': 0
            }

            class ASTVisitor(ast.NodeVisitor):
                def visit_FunctionDef(self, node): counts['function_definitions'] += 1; self.generic_visit(node)
                def visit_For(self, node): counts['for_loops'] += 1; self.generic_visit(node)
                def visit_While(self, node): counts['while_loops'] += 1; self.generic_visit(node)
                def visit_If(self, node): counts['if_statements'] += 1; self.generic_visit(node)
                def visit_Assign(self, node): counts['assignments'] += 1; self.generic_visit(node)
                def visit_Compare(self, node): counts['comparisons'] += 1; self.generic_visit(node)
                def visit_BinOp(self, node): counts['binary_ops'] += 1; self.generic_visit(node)
                def visit_Subscript(self, node): counts['subscripts'] += 1; self.generic_visit(node)
                def visit_Call(self, node): counts['method_calls'] += 1; self.generic_visit(node)
                def visit_ListComp(self, node): counts['list_comprehensions'] += 1; self.generic_visit(node)
                def visit_Lambda(self, node): counts['lambda_functions'] += 1; self.generic_visit(node)
                def visit_Try(self, node): counts['try_statements'] += 1; self.generic_visit(node)
                def visit_Return(self, node): counts['return_statements'] += 1; self.generic_visit(node)

            visitor = ASTVisitor()
            visitor.visit(tree)
            return counts
        except SyntaxError:
            return {key: 0 for key in ['function_definitions', 'for_loops', 'while_loops', 'if_statements',
                                       'assignments', 'comparisons', 'binary_ops', 'subscripts', 'method_calls',
                                       'list_comprehensions', 'lambda_functions', 'try_statements', 'return_statements']}

    # Extract all features
    literals = extract_literals_ast(program)
    ast_features = extract_ast_features(program)

    # Calculate features
    features = {
        'total_chars': len(program),
        'total_lines': len(program.split('\n')),
        'avg_line_length': len(program) / max(1, len(program.split('\n'))),
        'indentation_variance': np.var([len(line) - len(line.lstrip()) for line in program.split('\n')]),
        'max_indentation': max([len(line) - len(line.lstrip()) for line in program.split('\n')]),
        'comments': len(re.findall(r'#.*', program)),
        'docstrings': len(re.findall(r'""".*?"""', program, re.DOTALL)),
        'total_literals': len(literals),
        'literals_over_9': len([lit for lit in literals if lit > 9]),
        'literals_over_99': len([lit for lit in literals if lit > 99]),
        'max_literal': max(literals) if literals else 0,
        'unique_literals': len(set(literals)),
        'zero_literals': len([lit for lit in literals if lit == 0]),
        'single_digit_literals': len([lit for lit in literals if 0 <= lit <= 9]),
        'elif_chains': len(re.findall(r'elif\b', program)),
        'nested_loops': len(re.findall(r'for.*?:\s*.*?for.*?:', program, re.DOTALL)),
        'range_calls': len(re.findall(r'\brange\s*\(', program)),
        'len_calls': len(re.findall(r'\blen\s*\(', program)),
        'enumerate_calls': len(re.findall(r'\benumerate\s*\(', program)),
        'zip_calls': len(re.findall(r'\bzip\s*\(', program)),
        'numpy_usage': len(re.findall(r'\bnp\.|numpy\.', program)),
        'grid_shape_access': len(re.findall(r'\.shape\b|len\s*\(\s*grid\s*\)|len\s*\(\s*\w+\[0\]\s*\)', program)),
        'coordinate_patterns': len(re.findall(r'\[\s*\d+\s*\]\s*\[\s*\d+\s*\]', program)),
        'hardcoded_coordinates': len(re.findall(r'\[\s*\d+\s*\]', program)),
        'specific_conditionals': len(re.findall(r'==\s*\d+|!=\s*\d+|>\s*\d+|<\s*\d+|>=\s*\d+|<=\s*\d+', program)),
        'brackets_count': program.count('[') + program.count(']'),
        'general_loops': ast_features['for_loops'] + ast_features['while_loops'],
        'generic_variables': len(re.findall(r'\b(i|j|k|x|y|row|col|idx)\b', program)),
        'shape_operations': len(re.findall(r'\.shape|len\(.*\)', program)),
        'mathematical_ops': len(re.findall(r'\+|\-|\*|\/|\%|\*\*', program)),
        'array_creation': len(re.findall(r'\[\s*\[|np\.array|np\.zeros|np\.ones', program)),
        'imports': len(re.findall(r'^\s*import|^\s*from', program, re.MULTILINE)),
    }

    # Add AST features
    features.update(ast_features)

    # Extract and scale only the active features
    feature_names = ['max_indentation', 'max_literal', 'unique_literals', 'function_definitions', 'if_statements', 'method_calls', 'elif_chains', 'enumerate_calls', 'coordinate_patterns', 'specific_conditionals', 'brackets_count', 'generic_variables', 'array_creation']
    feature_values = np.array([features.get(name, 0) for name in feature_names])

    # Standardize features
    means = np.array([np.float64(15.50943396226415), np.float64(8.68867924528302), np.float64(6.349056603773585), np.float64(1.4150943396226414), np.float64(4.377358490566038), np.float64(8.962264150943396), np.float64(1.509433962264151), np.float64(0.04716981132075472), np.float64(2.641509433962264), np.float64(4.433962264150943), np.float64(54.9811320754717), np.float64(13.226415094339623), np.float64(1.2075471698113207)])
    scales = np.array([np.float64(6.447531753883583), np.float64(10.782028380200826), np.float64(4.229372701669386), np.float64(0.9097783736785806), np.float64(2.9698257054145647), np.float64(6.554451934116182), np.float64(2.0477566327010375), np.float64(0.2526118460187903), np.float64(7.326855073548195), np.float64(3.959747628523171), np.float64(48.07246517656825), np.float64(15.340342065795706), np.float64(1.4123243329207051)])
    scaled_features = (feature_values - means) / scales

    # Apply logistic regression
    coefficients = np.array([np.float64(-0.19200163632604209), np.float64(0.3290678302014203), np.float64(1.1821002685300024), np.float64(-0.0997574551447266), np.float64(-0.08991160874428443), np.float64(-0.7212949338101916), np.float64(0.7789118055904987), np.float64(-0.20197725309509285), np.float64(0.6404413289401605), np.float64(0.11604449603185256), np.float64(1.5666116312766467), np.float64(-0.4183994878606218), np.float64(0.16657497697757676)])
    intercept = -0.06590329109909544

    logit = np.dot(scaled_features, coefficients) + intercept
    probability = 1 / (1 + np.exp(-logit))

    is_transductive = probability > 0.5
    confidence = probability if is_transductive else 1 - probability

    return bool(is_transductive), float(confidence)

