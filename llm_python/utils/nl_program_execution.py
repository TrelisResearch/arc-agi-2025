"""
Utilities for executing natural language programs using LLM calls.
"""

import re
import ast
from typing import List, Optional, Tuple, Dict, Any
from llm_python.utils.prompt_loader import PromptLoader


def extract_program_from_response(response, debug: bool = False) -> Optional[str]:
    """Extract natural language program from LLM response"""
    full_text = ""

    if hasattr(response, "choices") and len(response.choices) > 0:
        message = response.choices[0].message
        if hasattr(message, "content") and message.content:
            full_text = message.content

    if debug and len(full_text) > 0:
        print(f"🔍 Response content: {len(full_text)} chars")

    # Extract content between <PROGRAM> tags
    tag_match = re.search(r'<PROGRAM>\s*(.*?)\s*</PROGRAM>', full_text, re.DOTALL | re.IGNORECASE)
    if tag_match:
        return tag_match.group(1).strip()

    # If no tags found, return None (no program extracted)
    return None


def extract_grid_from_response(response) -> Optional[List[List[int]]]:
    """Extract output grid from LLM execution response"""
    if not response:
        return None

    full_text = ""
    if hasattr(response, "choices") and len(response.choices) > 0:
        message = response.choices[0].message
        if hasattr(message, "content") and message.content:
            full_text = message.content

    if not full_text:
        return None

    # Extract grid from <OUTPUT> tags
    tag_match = re.search(r'<OUTPUT>\s*(\[.*?\])\s*</OUTPUT>', full_text, re.DOTALL)
    if tag_match:
        try:
            grid_str = tag_match.group(1).strip()
            # Use ast.literal_eval for safe evaluation
            grid = ast.literal_eval(grid_str)

            # Validate it's a proper 2D list of integers
            if isinstance(grid, list) and all(isinstance(row, list) for row in grid):
                if all(isinstance(cell, int) and 0 <= cell <= 9 for row in grid for cell in row):
                    return grid
        except (ValueError, SyntaxError):
            pass

    return None


class NaturalLanguageProgramExecutor:
    """Executes natural language programs using LLM calls"""

    def __init__(self, api_client, debug: bool = False):
        self.api_client = api_client
        self.debug = debug
        self.prompt_loader = PromptLoader()

    def call_chat_completions_api(self, messages: List[Dict]) -> tuple:
        """Call the OpenAI Chat Completions API"""
        result = self.api_client.call_chat_completions_api(messages)
        if result["success"]:
            return result["response"], result["sampling_params"]
        else:
            # Return None response with error info
            return None, result["sampling_params"]

    def execute_program(self, program: str, input_grid: List[List[int]]) -> Tuple[Optional[List[List[int]]], str, bool]:
        """Execute a natural language program using LLM to produce output grid"""
        try:
            # Format the input grid for display
            input_grid_str = str(input_grid).replace(',', '')

            # Load execution prompt template
            system_prompt = self.prompt_loader.get_system_message("program-execution")
            user_prompt = self.prompt_loader.get_initial_turn_prompt("program-execution")

            # Format the prompt
            formatted_prompt = user_prompt.format(
                program=program,
                input_grid=input_grid_str
            )

            conversation = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": formatted_prompt}
            ]

            # Make the API call using existing infrastructure
            response, _ = self.call_chat_completions_api(conversation)

            if response is None:
                return None, "API call failed", False

            # Extract the output grid
            output_grid = extract_grid_from_response(response)

            if output_grid is None:
                return None, "Could not extract valid grid from response", False

            return output_grid, "", False

        except Exception as e:
            return None, str(e), False