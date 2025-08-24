import ast
from pathlib import Path
import re
from typing import Optional, Tuple
import joblib
import pandas as pd
import numpy as np

from llm_python.utils.task_loader import TaskData


class CodeTransductionClassifier:
    """
    Classifier that determines if a program is transductive based on code analysis.
    """

    def __init__(self):
        """Initialize the code-based transduction classifier."""
        model_path = Path(__file__).parent / "code_classifier.joblib"
        self.clf_v2 = joblib.load(model_path)

    def is_transductive(
        self, program: str, task_data: Optional[TaskData] = None
    ) -> Tuple[bool, float]:
        """
        Classify whether an ARC program is transductive or inductive.

        Args:
            program: The program code as a string
            task_data: Optional task data (not used in code-based classification)

        Returns:
            tuple: (is_transductive: bool, confidence: float)
        """
        feats = extract_code_features(program)
        X = pd.DataFrame([feats])
        pred = self.clf_v2.predict(X)[0]
        prob = self.clf_v2.predict_proba(X)[0, 1]
        return bool(pred), prob


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
        numeric_literals = re.findall(r"\b\d+(?:\.\d+)?\b", program_code)
        return [float(lit) if "." in lit else int(lit) for lit in numeric_literals]


def extract_ast_features(program):
    """Extract AST structural features"""
    try:
        tree = ast.parse(program)
        counts = {
            "function_definitions": 0,
            "for_loops": 0,
            "while_loops": 0,
            "if_statements": 0,
            "assignments": 0,
            "comparisons": 0,
            "binary_ops": 0,
            "subscripts": 0,
            "method_calls": 0,
            "list_comprehensions": 0,
            "lambda_functions": 0,
            "try_statements": 0,
            "return_statements": 0,
        }

        class ASTVisitor(ast.NodeVisitor):
            def visit_FunctionDef(self, node):
                counts["function_definitions"] += 1
                self.generic_visit(node)

            def visit_For(self, node):
                counts["for_loops"] += 1
                self.generic_visit(node)

            def visit_While(self, node):
                counts["while_loops"] += 1
                self.generic_visit(node)

            def visit_If(self, node):
                counts["if_statements"] += 1
                self.generic_visit(node)

            def visit_Assign(self, node):
                counts["assignments"] += 1
                self.generic_visit(node)

            def visit_Compare(self, node):
                counts["comparisons"] += 1
                self.generic_visit(node)

            def visit_BinOp(self, node):
                counts["binary_ops"] += 1
                self.generic_visit(node)

            def visit_Subscript(self, node):
                counts["subscripts"] += 1
                self.generic_visit(node)

            def visit_Call(self, node):
                counts["method_calls"] += 1
                self.generic_visit(node)

            def visit_ListComp(self, node):
                counts["list_comprehensions"] += 1
                self.generic_visit(node)

            def visit_Lambda(self, node):
                counts["lambda_functions"] += 1
                self.generic_visit(node)

            def visit_Try(self, node):
                counts["try_statements"] += 1
                self.generic_visit(node)

            def visit_Return(self, node):
                counts["return_statements"] += 1
                self.generic_visit(node)

        visitor = ASTVisitor()
        visitor.visit(tree)
        return counts
    except SyntaxError:
        return {
            key: 0
            for key in [
                "function_definitions",
                "for_loops",
                "while_loops",
                "if_statements",
                "assignments",
                "comparisons",
                "binary_ops",
                "subscripts",
                "method_calls",
                "list_comprehensions",
                "lambda_functions",
                "try_statements",
                "return_statements",
            ]
        }


def extract_code_features(program):
    # Extract all features
    literals = extract_literals_ast(program)
    ast_features = extract_ast_features(program)

    # Calculate features
    features = {
        "total_chars": len(program),
        "total_lines": len(program.split("\n")),
        "avg_line_length": len(program) / max(1, len(program.split("\n"))),
        "indentation_variance": np.var(
            [len(line) - len(line.lstrip()) for line in program.split("\n")]
        ),
        "max_indentation": max(
            [len(line) - len(line.lstrip()) for line in program.split("\n")]
        ),
        "comments": len(re.findall(r"#.*", program)),
        "docstrings": len(re.findall(r'""".*?"""', program, re.DOTALL)),
        "total_literals": len(literals),
        "literals_over_9": len([lit for lit in literals if lit > 9]),
        "literals_over_99": len([lit for lit in literals if lit > 99]),
        "max_literal": max(literals) if literals else 0,
        "unique_literals": len(set(literals)),
        "zero_literals": len([lit for lit in literals if lit == 0]),
        "single_digit_literals": len([lit for lit in literals if 0 <= lit <= 9]),
        "elif_chains": len(re.findall(r"elif\b", program)),
        "nested_loops": len(re.findall(r"for.*?:\s*.*?for.*?:", program, re.DOTALL)),
        "range_calls": len(re.findall(r"\brange\s*\(", program)),
        "len_calls": len(re.findall(r"\blen\s*\(", program)),
        "enumerate_calls": len(re.findall(r"\benumerate\s*\(", program)),
        "zip_calls": len(re.findall(r"\bzip\s*\(", program)),
        "numpy_usage": len(re.findall(r"\bnp\.|numpy\.", program)),
        "grid_shape_access": len(
            re.findall(
                r"\.shape\b|len\s*\(\s*grid\s*\)|len\s*\(\s*\w+\[0\]\s*\)", program
            )
        ),
        "coordinate_patterns": len(
            re.findall(r"\[\s*\d+\s*\]\s*\[\s*\d+\s*\]", program)
        ),
        "hardcoded_coordinates": len(re.findall(r"\[\s*\d+\s*\]", program)),
        "specific_conditionals": len(
            re.findall(r"==\s*\d+|!=\s*\d+|>\s*\d+|<\s*\d+|>=\s*\d+|<=\s*\d+", program)
        ),
        "brackets_count": program.count("[") + program.count("]"),
        "general_loops": ast_features["for_loops"] + ast_features["while_loops"],
        "generic_variables": len(re.findall(r"\b(i|j|k|x|y|row|col|idx)\b", program)),
        "shape_operations": len(re.findall(r"\.shape|len\(.*\)", program)),
        "mathematical_ops": len(re.findall(r"\+|\-|\*|\/|\%|\*\*", program)),
        "array_creation": len(
            re.findall(r"\[\s*\[|np\.array|np\.zeros|np\.ones", program)
        ),
        "imports": len(re.findall(r"^\s*import|^\s*from", program, re.MULTILINE)),
    }

    # Add AST features
    features.update(ast_features)

    # return features
    # return features
    # Extract and scale only the active features
    feature_names = [
        "max_indentation",
        "max_literal",
        "unique_literals",
        "function_definitions",
        "if_statements",
        "method_calls",
        "elif_chains",
        "enumerate_calls",
        "coordinate_patterns",
        "specific_conditionals",
        "brackets_count",
        "generic_variables",
        "array_creation",
    ]
    return {k: features[k] for k in feature_names if k in features}
