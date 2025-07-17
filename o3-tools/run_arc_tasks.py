#!/usr/bin/env python3

import os
import json
import argparse
import datetime
import time
import threading
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

from task_loader import TaskLoader
from scoring import GridScorer, ProgramExecutor

load_dotenv()

def execute_with_timeout(func, *args, timeout=1000, **kwargs):
    """Execute a function with timeout using ThreadPoolExecutor"""
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            result = future.result(timeout=timeout)
            return result
        except Exception as e:
            # Cancel the future if it's still running
            future.cancel()
            raise e

class ARCTaskRunner:
    """Run ARC tasks using the OpenAI Chat Completions API"""
    
    def __init__(self, model: str = "gpt-4.1-nano", max_workers: int = 1, rate_limit_delay: float = 0.0, max_turns: int = 3, run_number: int = 0, independent_attempts: bool = False, base_url: str = None, reasoning_effort: str = "low", debug: bool = False, qwen_no_think: bool = False):
        self.model = model
        self.max_workers = max_workers
        self.rate_limit_delay = rate_limit_delay
        self.max_turns = max_turns
        self.run_number = run_number  # Track run number for repeated runs
        self.independent_attempts = independent_attempts  # Track independent attempts mode
        self.reasoning_effort = reasoning_effort
        self.qwen_no_think = qwen_no_think  # Track whether to disable thinking for Qwen models
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.base_url = base_url
        self.debug = debug
        
        # Initialize OpenAI client with optional base URL
        if base_url:
            self.client = OpenAI(api_key=self.api_key, base_url=base_url)
            print(f"📝 Using custom endpoint: {base_url}")
        else:
            self.client = OpenAI(api_key=self.api_key)
            print(f"📝 Using OpenAI endpoint")
        
        print(f"📝 Text-only mode for {self.model}")
        
        self.task_loader = TaskLoader()
        self.scorer = GridScorer()
        self.executor = ProgramExecutor(timeout=0.5)
        
        # Create logs directory
        self.logs_dir = Path("logs")
        self.logs_dir.mkdir(exist_ok=True)
        
        # Thread-safe cost and token tracking
        self._cost_lock = threading.Lock()
        self.total_cost = 0.0
        self.total_tokens = 0
        
        # Thread-safe progress tracking
        self._progress_lock = threading.Lock()
        self.completed_tasks = 0
        
        # Enable debug extraction for troubleshooting
        self._debug_extraction = True
        
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
    
    def _update_costs(self, cost: float, tokens: int):
        """Thread-safe method to update total costs and tokens"""
        with self._cost_lock:
            self.total_cost += cost
            self.total_tokens += tokens
    
    def _update_progress(self, total_tasks: int, task_id: str = None, success: bool = True):
        """Thread-safe method to update and print progress"""
        with self._progress_lock:
            self.completed_tasks += 1
            if self.max_workers > 1:  # Only show progress for parallel execution
                progress_pct = (self.completed_tasks / total_tasks) * 100
                status = "✅ COMPLETED" if success else "❌ FAILED"
                task_info = f" ({task_id})" if task_id else ""
                print(f"Progress: {self.completed_tasks}/{total_tasks} tasks processed ({progress_pct:.1f}%) - {status}{task_info}")
    

    
    def get_model_pricing(self, model: str) -> tuple[float, float]:
        """Get input and output pricing rates for a model in $/1M tokens"""
        model_lower = model.lower()
        
        # Reasoning models
        if model_lower.startswith('o3-pro'):
            return (20.00, 80.00)
        elif model_lower.startswith('o3-deep-research'):
            return (10.00, 40.00)
        elif model_lower.startswith('o3-mini'):
            return (1.10, 4.40)
        elif model_lower.startswith('o3'):
            return (2.00, 8.00)
        elif model_lower.startswith('o4-mini-deep-research'):
            return (2.00, 8.00)
        elif model_lower.startswith('o4-mini'):
            return (1.10, 4.40)
        elif model_lower.startswith('o1-pro'):
            return (150.00, 600.00)
        elif model_lower.startswith('o1-mini'):
            return (1.10, 4.40)
        elif model_lower.startswith('o1'):
            return (15.00, 60.00)
        
        # GPT-4 models
        elif model_lower.startswith('gpt-4.1-nano'):
            return (0.10, 0.40)
        elif model_lower.startswith('gpt-4.1-mini'):
            return (0.40, 1.60)
        elif model_lower.startswith('gpt-4.1'):
            return (2.00, 8.00)
        elif model_lower.startswith('gpt-4.5-preview'):
            return (75.00, 150.00)
        elif model_lower.startswith('gpt-4o-mini-realtime-preview'):
            return (0.60, 2.40)
        elif model_lower.startswith('gpt-4o-mini'):
            return (0.15, 0.60)
        elif model_lower.startswith('gpt-4o-realtime-preview'):
            return (5.00, 20.00)
        elif model_lower.startswith('gpt-4o-search-preview'):
            return (2.50, 10.00)
        elif model_lower.startswith('gpt-4o'):
            return (2.50, 10.00)
        
        # Default fallback (gpt-4o-mini pricing)
        else:
            return (0.15, 0.60)
    
    def create_prompt(self, task_data: Dict, is_first_turn: bool = True, task_id: str = None, turn: int = None) -> str:
        """Create a prompt for the model to solve an ARC task (returns text content for chat messages)"""
        if not is_first_turn:
            # For subsequent turns, encourage partial solutions
            return """Please analyze the training feedback and programs from all prior turns, and write an improved transformation.

**Important**:
1. Always start by attempting to find a complete rule that solves all training examples and should generalize to the test input.
2. If you cannot find a perfect solution, provide your best attempt at a transformation that solves as many training examples as possible - ideally more than in the best previous turn.
3. Even if your solution doesn't solve all examples perfectly, ensure it demonstrates meaningful pattern recognition and provides reasonable outputs that minimize pixel-level errors.

Final answer:
```python
def transform(grid):
    # Your improved transformation logic here (even if partial)
    return transformed_grid
```"""
        
        # Get the consistent output grid dimensions from the first training example
        if task_data['train'] and task_data['train'][0]['output']:
            output_grid = task_data['train'][0]['output']
            output_height = len(output_grid)
            output_width = len(output_grid[0]) if output_grid else 0
            grid_size_info = f"\n**IMPORTANT: Your transformation must always produce a {output_height}×{output_width} output grid.**\n"
        else:
            grid_size_info = ""
        
        # Create the text content with text grid representation
        task_str = self.task_loader.format_task_for_prompt(task_data, include_test=True)
        
        text_content = f"""You are solving an ARC (Abstraction and Reasoning Corpus) task. 
I will show you training examples with input and output grids, plus a test input grid. Your task is to:

1. **Analyze the training examples** to discover patterns that map input grids to output grids
2. **Write a Python program** that implements your best understanding of the transformation  
3. **DO NOT predict or generate the test output** - your job is only to write the transformation program
4. **Attempt a solution** - even if the pattern isn't completely clear, provide your best hypothesis
5. **Do not repeat the same transformation** - if you have already tried a transformation, do not repeat it.
{grid_size_info}
The test input is shown for context so you understand what type of grid your program will eventually process. Focus on learning patterns from training examples and writing code that captures your understanding.

{task_str}

Analyze the patterns in the training examples and write a Python function that performs this transformation.

**Approach Guidelines:**
- Look for patterns in shapes, colors, positions, sizes, rotations, reflections, etc.
- Even if you can't solve all training examples perfectly, implement what patterns you do observe
- A partial solution that captures some aspects is better than returning the input unchanged
- If the pattern is unclear, make your best educated guess based on what you can see

Requirements:
- The function takes a 2D list (grid) where grid[row][col] gives the value at that position
- Values are integers from 0-9
- Return a new grid (2D list) with the transformation applied
- You can use numpy if needed - just add 'import numpy as np' at the start of your function
- Aim to handle the training examples as well as possible, even if not perfectly
- Your function should attempt some meaningful transformation based on the patterns you observe

You MUST end your response with the following exact format:

Final answer:
```python
def transform(grid):
    # Your transformation logic here (implement your best understanding)
    return transformed_grid
```
"""
        
        return text_content
    
    def call_chat_completions_api(self, messages: List[Dict]) -> Dict:
        """Call the OpenAI Chat Completions API"""
        try:
            # Prepare the request
            kwargs = {
                "model": self.model,
                "messages": messages
            }
            
            # Add reasoning parameters for OpenRouter and reasoning models
            if self.base_url and "openrouter" in self.base_url.lower():
                # OpenRouter reasoning token allocation
                reasoning_tokens = {
                    "low": 4000,
                    "medium": 16000, 
                    "high": 64000
                }
                if self.reasoning_effort in reasoning_tokens:
                    kwargs["max_tokens"] = reasoning_tokens[self.reasoning_effort]
            
            # Add Qwen-specific parameters for thinking models (only for custom endpoints)
            if "qwen" in self.model.lower() and self.base_url:
                if self.qwen_no_think:
                    # Parameters for non-thinking Qwen models
                    kwargs.update({
                        "temperature": 0.7,
                        "top_p": 0.8,
                        "extra_body": {"top_k": 20, "chat_template_kwargs": {"enable_thinking": False}}
                    })
                else:
                    # Parameters for thinking Qwen models (original behavior)
                    kwargs.update({
                        "temperature": 0.6,
                        "top_p": 0.95,
                        "extra_body": {"top_k": 20}
                    })
            
            # Make the API call
            response = self.client.chat.completions.create(**kwargs)
            
            return response
            
        except Exception as e:
            raise Exception(f"API call failed: {e}")
    
    def create_training_feedback(self, program: str, training_examples: List[Dict], test_correct: bool = None, task_id: str = None, turn: int = None) -> tuple[List, int, float]:
        """Generate detailed training feedback with stats, actual outputs for LLM"""
        results = []
        predicted_outputs = []
        total_pixels = 0
        correct_pixels = 0
        solved_count = 0
        has_grid_size_mismatch = False
        
        for i, example in enumerate(training_examples):
            predicted_output, error, timed_out = self.executor.execute_program(program, example['input'])
            expected_output = example['output']
            predicted_outputs.append(predicted_output)
            
            if predicted_output is not None and not error and not timed_out:
                # Calculate pixel accuracy for this example
                if isinstance(predicted_output, list) and isinstance(expected_output, list):
                    example_correct_pixels = 0
                    example_total_pixels = 0
                    
                    # Handle dimension mismatches gracefully
                    min_rows = min(len(predicted_output), len(expected_output))
                    for r in range(min_rows):
                        if r < len(predicted_output) and r < len(expected_output):
                            min_cols = min(len(predicted_output[r]), len(expected_output[r]))
                            for c in range(min_cols):
                                example_total_pixels += 1
                                if predicted_output[r][c] == expected_output[r][c]:
                                    example_correct_pixels += 1
                    
                    # Account for dimension mismatches as errors
                    expected_size = len(expected_output) * len(expected_output[0]) if expected_output else 0
                    predicted_size = len(predicted_output) * len(predicted_output[0]) if predicted_output else 0
                    example_total_pixels = max(expected_size, predicted_size, example_total_pixels)
                    
                    total_pixels += example_total_pixels
                    correct_pixels += example_correct_pixels
                    
                    pixel_accuracy = example_correct_pixels / example_total_pixels if example_total_pixels > 0 else 0.0
                    is_solved = (pixel_accuracy == 1.0)
                else:
                    pixel_accuracy = 0.0
                    is_solved = False
                    if expected_output:
                        total_pixels += len(expected_output) * len(expected_output[0])
            else:
                pixel_accuracy = 0.0
                is_solved = False
                if expected_output:
                    total_pixels += len(expected_output) * len(expected_output[0])
                    
            if is_solved:
                solved_count += 1
            
            # Check for grid size mismatch
            if predicted_output and expected_output:
                predicted_shape = (len(predicted_output), len(predicted_output[0]) if predicted_output else 0)
                expected_shape = (len(expected_output), len(expected_output[0]) if expected_output else 0)
                if predicted_shape != expected_shape:
                    has_grid_size_mismatch = True
                
            results.append({
                'index': i + 1,
                'predicted': predicted_output,
                'expected': expected_output,
                'solved': is_solved,
                'pixel_accuracy': pixel_accuracy,
                'error': error,
                'timed_out': timed_out
            })
        
        # Calculate overall accuracy
        overall_accuracy = correct_pixels / total_pixels if total_pixels > 0 else 0.0
        
        # Format feedback for LLM with encouraging language
        feedback_text = f"Training results: {solved_count}/{len(training_examples)} examples solved, {overall_accuracy:.1%} pixel accuracy"
        
        # Add encouraging context for partial solutions
        if solved_count == len(training_examples) and test_correct is False:
            feedback_text += " - Perfect training accuracy but test failed! Your transformation is overfitting to the training examples. Try to generalize your approach - look for broader patterns that work across all cases, not just the specific training examples.\n\n"
        elif solved_count == 0 and overall_accuracy > 0:
            feedback_text += " - Good partial progress! Your approach is capturing some patterns. Next, try to identify a transformation that solves at least one of the training examples.\n\n"
        elif solved_count == 0 and overall_accuracy == 0:
            feedback_text += " - Keep experimenting! Try a different approach to the transformation.\n\n"
        elif solved_count > 0:
            feedback_text += f" - Great progress! You're on the right track. Next, try to identify a transformation that solves more than {solved_count} of the training examples.\n\n"
        else:
            feedback_text += "\n\n"
        
        for result in results:
            status = "✓" if result['solved'] else "✗"
            feedback_text += f"Training Example {result['index']} {status}:\n"
            feedback_text += f"Expected: {result['expected']}\n"
            
            if result['error']:
                feedback_text += f"Error: {result['error']}\n"
            elif result['timed_out']:
                feedback_text += f"Error: Code execution timed out\n"
            else:
                feedback_text += f"Your output: {result['predicted']}\n"
                if not result['solved']:
                    feedback_text += f"Pixel accuracy: {result['pixel_accuracy']:.1%}\n"
            feedback_text += "\n"
        
        # Add general grid size reminder if there were any mismatches
        if has_grid_size_mismatch and training_examples:
            expected_output = training_examples[0]['output']
            if expected_output:
                expected_height = len(expected_output)
                expected_width = len(expected_output[0]) if expected_output else 0
                feedback_text += f"⚠️  GRID SIZE ERROR: One or more of your output grids are incorrect in size. They should ALL be {expected_height}×{expected_width} for this task.\n\n"
        
        # Create message list with text feedback
        feedback_messages = [{"type": "input_text", "text": feedback_text}]
        
        return feedback_messages, solved_count, overall_accuracy
    
    def extract_code_from_response(self, response) -> str:
        """Extract Python code from the Chat Completions API result using simple regex"""
        import re
        
        # Get the full text from response
        full_text = ""
        reasoning_text = ""
        
        # Extract from Chat Completions API structure
        if hasattr(response, 'choices') and len(response.choices) > 0:
            message = response.choices[0].message
            if hasattr(message, 'content') and message.content:
                full_text = message.content
            # Also check reasoning field (for models like Qwen via OpenRouter)
            if hasattr(message, 'reasoning') and message.reasoning:
                reasoning_text = message.reasoning
            # Also check reasoning_content field (for models like Qwen via RunPod)
            if hasattr(message, 'reasoning_content') and message.reasoning_content:
                reasoning_text += "\n\n" + message.reasoning_content if reasoning_text else message.reasoning_content
        
        # Combine both content and reasoning for code extraction
        combined_text = full_text + "\n\n" + reasoning_text if reasoning_text else full_text
        
        # First priority: look for code after "Final answer:"
        final_answer_match = re.search(r'Final answer:\s*```python\s*\n(.*?)\n```', combined_text, re.DOTALL | re.IGNORECASE)
        if final_answer_match:
            return final_answer_match.group(1).strip()
        
        # Second priority: last ```python block 
        python_blocks = re.findall(r'```python\s*\n(.*?)\n```', combined_text, re.DOTALL)
        if python_blocks:
            return python_blocks[-1].strip()
        
        # Third priority: any ``` block with def transform
        code_blocks = re.findall(r'```\s*\n(.*?)\n```', combined_text, re.DOTALL)
        for block in reversed(code_blocks):  # Check from last to first
            if 'def transform' in block:
                return block.strip()
        
        # Last resort: extract def transform function without code blocks
        transform_match = re.search(r'(def transform.*?)(?=\n\S|\n*$)', combined_text, re.DOTALL)
        if transform_match:
            return transform_match.group(1).strip()
        
        return ""
    
    def create_success_result(self, task_id: str, program: str, response, test_score: Dict, total_cost: float, total_tokens: int, turns_used: int, task_data: Dict, conversation_history: List = None, all_responses: List = None, turn_details: List = None, is_independent_attempts: bool = False) -> Dict:
        """Create a successful task result with complete multi-turn data"""

        # Convert response to JSON-serializable format
        response_dict = None
        if response:
            try:
                response_dict = {
                    'id': response.id,
                    'model': response.model,
                    'usage': {
                        'prompt_tokens': response.usage.prompt_tokens if response.usage else 0,
                        'completion_tokens': response.usage.completion_tokens if response.usage else 0,
                        'total_tokens': response.usage.total_tokens if response.usage else 0
                    },
                    'content': response.choices[0].message.content if response.choices else "",
                    'reasoning': response.choices[0].message.reasoning if response.choices and hasattr(response.choices[0].message, 'reasoning') else None,
                    'reasoning_content': response.choices[0].message.reasoning_content if response.choices and hasattr(response.choices[0].message, 'reasoning_content') else None
                }
            except Exception as e:
                response_dict = {'error': f'Failed to serialize response: {str(e)}'}

        # Convert all responses to JSON-serializable format
        all_responses_dict = []
        if all_responses:
            for resp in all_responses:
                try:
                    all_responses_dict.append({
                        'id': resp.id,
                        'model': resp.model,
                        'usage': {
                            'prompt_tokens': resp.usage.prompt_tokens if resp.usage else 0,
                            'completion_tokens': resp.usage.completion_tokens if resp.usage else 0,
                            'total_tokens': resp.usage.total_tokens if resp.usage else 0
                        },
                        'content': resp.choices[0].message.content if resp.choices else "",
                        'reasoning': resp.choices[0].message.reasoning if resp.choices and hasattr(resp.choices[0].message, 'reasoning') else None,
                        'reasoning_content': resp.choices[0].message.reasoning_content if resp.choices and hasattr(resp.choices[0].message, 'reasoning_content') else None
                    })
                except Exception as e:
                    all_responses_dict.append({'error': f'Failed to serialize response: {str(e)}'})

        # Serialize conversation history - much simpler for chat completions
        conversation_history_dict = conversation_history or []

        # Determine API type and data structure based on mode
        api_type = 'chat_completions_independent_attempts' if is_independent_attempts else 'chat_completions_multiturn'
        data_key = 'independent_attempts_data' if is_independent_attempts else 'multiturn_data'
        details_key = 'attempt_details' if is_independent_attempts else 'turn_details'
        total_key = 'total_attempts' if is_independent_attempts else 'total_turns'
        
        result = {
            'task_id': task_id,
            'model': self.model,
            'reasoning_effort': self.reasoning_effort,
            'api_type': api_type,
            'program': program,
            'task_failure_reason': '',
            'timed_out': False,
            'tokens_used': total_tokens,
            'request_cost': total_cost,
            'turns_used': turns_used,
            'raw_response': response_dict,  # Final response only (for backward compatibility)
            'score': test_score,
            'predicted_output': test_score.get('predicted_output'),
            'actual_output': test_score.get('actual_output'),
            'api_success': True,
        }
        
        # Add mode-specific data structure
        result[data_key] = {
            details_key: turn_details or [],
            total_key: turns_used
        }
        
        # Add conversation history only for multi-turn mode
        if not is_independent_attempts:
            result[data_key]['conversation_history'] = conversation_history_dict
            
        # Always add all_responses
        result[data_key]['all_responses'] = all_responses_dict
        
        # Add mode indicator for independent attempts
        if is_independent_attempts:
            result[data_key]['mode'] = 'independent_attempts'
            
        return result
    
    def _serialize_api_response(self, response):
        """Helper function to serialize API responses (handles both o3 and ChatCompletion formats)"""
        try:
            # Handle ChatCompletion format (standard OpenAI)
            if hasattr(response, 'choices'):
                choices_data = []
                for choice in response.choices:
                    choice_data = {
                        'index': getattr(choice, 'index', 0),
                        'finish_reason': getattr(choice, 'finish_reason', None),
                        'message': {
                            'role': getattr(choice.message, 'role', 'assistant'),
                            'content': getattr(choice.message, 'content', None),
                            'reasoning_content': getattr(choice.message, 'reasoning_content', None)
                        }
                    }
                    choices_data.append(choice_data)
                
                # Handle different token field names
                usage_data = {}
                if hasattr(response, 'usage') and response.usage:
                    usage = response.usage
                    usage_data = {
                        'input_tokens': getattr(usage, 'input_tokens', None) or getattr(usage, 'prompt_tokens', None),
                        'output_tokens': getattr(usage, 'output_tokens', None) or getattr(usage, 'completion_tokens', None),
                        'total_tokens': getattr(usage, 'total_tokens', None)
                    }
                
                return {
                    'id': getattr(response, 'id', 'unknown'),
                    'model': getattr(response, 'model', 'unknown'),
                    'usage': usage_data,
                    'choices': choices_data,
                    'format': 'chat_completion'
                }
            
            # Handle o3 reasoning format (legacy)
            elif hasattr(response, 'output'):
                output_items = []
                for item in response.output:
                    if hasattr(item, 'type') and item.type == 'message':
                        # Extract text content from message content items
                        content_texts = []
                        if hasattr(item, 'content'):
                            for content_item in item.content:
                                if hasattr(content_item, 'text'):
                                    content_texts.append(content_item.text)
                        output_items.append({'type': item.type, 'content': content_texts})
                    else:
                        output_items.append({'type': getattr(item, 'type', 'unknown'), 'content': str(getattr(item, 'content', ''))})
                
                return {
                    'id': getattr(response, 'id', 'unknown'),
                    'model': getattr(response, 'model', 'unknown'),
                    'usage': {
                        'input_tokens': getattr(response.usage, 'input_tokens', None) if hasattr(response, 'usage') else None,
                        'output_tokens': getattr(response.usage, 'output_tokens', None) if hasattr(response, 'usage') else None,
                        'total_tokens': getattr(response.usage, 'total_tokens', None) if hasattr(response, 'usage') else None
                    },
                    'output': output_items,
                    'format': 'o3_reasoning'
                }
            
            # Fallback for unknown formats
            else:
                return {
                    'error': f'Unknown response format: {type(response)}',
                    'attributes': [attr for attr in dir(response) if not attr.startswith('_')]
                }
                
        except Exception as e:
            return {'error': f'Failed to serialize response: {str(e)}'}

    def create_failure_result(self, task_id: str, program: str, all_responses: List, total_cost: float, total_tokens: int, turns_used: int, task_data: Dict, error_msg: str, conversation_history: List = None, turn_details: List = None, is_independent_attempts: bool = False) -> Dict:
        """Create a failed task result with complete multi-turn data"""
        # Get test output for pixel counting
        actual_output = task_data['test'][0]['output']
        total_pixels = len(actual_output) * len(actual_output[0]) if actual_output else 0
        
        # Convert last response to JSON-serializable format
        response_dict = None
        if all_responses:
            response_dict = self._serialize_api_response(all_responses[-1])

        # Convert all responses to JSON-serializable format
        all_responses_dict = []
        if all_responses:
            for resp in all_responses:
                all_responses_dict.append(self._serialize_api_response(resp))

        # Serialize conversation history (handle response objects)
        conversation_history_dict = []
        if conversation_history:
            for item in conversation_history:
                try:
                    if hasattr(item, 'choices') or hasattr(item, 'output'):  # API response object
                        conversation_history_dict.append(self._serialize_api_response(item))
                    else:  # Regular dict message
                        conversation_history_dict.append(item)
                except Exception as e:
                    conversation_history_dict.append({'error': f'Failed to serialize conversation item: {str(e)}'})
        
        # Determine API type and data structure based on mode
        api_type = 'chat_completions_independent_attempts' if is_independent_attempts else 'chat_completions_multiturn'
        data_key = 'independent_attempts_data' if is_independent_attempts else 'multiturn_data'
        details_key = 'attempt_details' if is_independent_attempts else 'turn_details'
        total_key = 'total_attempts' if is_independent_attempts else 'total_turns'
        
        result = {
            'task_id': task_id,
            'model': self.model,
            'reasoning_effort': self.reasoning_effort,
            'api_type': api_type,
            'program': program,
            'task_failure_reason': error_msg,
            'timed_out': False,
            'tokens_used': total_tokens,
            'request_cost': total_cost,
            'turns_used': turns_used,
            'raw_response': response_dict,  # Final response only (for backward compatibility)
            'score': {
                'correct': False,
                'pixel_accuracy': 0.0,
                'total_pixels': total_pixels,
                'correct_pixels': 0,
                'error': error_msg
            },
            'actual_output': actual_output,
            'api_success': True,
        }
        
        # Add mode-specific data structure
        result[data_key] = {
            details_key: turn_details or [],
            total_key: turns_used
        }
        
        # Add conversation history only for multi-turn mode
        if not is_independent_attempts:
            result[data_key]['conversation_history'] = conversation_history_dict
            
        # Always add all_responses
        result[data_key]['all_responses'] = all_responses_dict
        
        # Add mode indicator for independent attempts
        if is_independent_attempts:
            result[data_key]['mode'] = 'independent_attempts'
            
        return result
    
    def create_timeout_failure_result(self, task_id: str, total_cost: float, total_tokens: int, turns_completed: int, task_data: Dict, conversation_history: List = None, all_responses: List = None, turn_details: List = None) -> Dict:
        """Create a timeout failure result - separate from regular task failures"""
        # Get test output for pixel counting
        actual_output = task_data['test'][0]['output']
        total_pixels = len(actual_output) * len(actual_output[0]) if actual_output else 0
        
        # Convert all responses to JSON-serializable format
        all_responses_dict = []
        if all_responses:
            for resp in all_responses:
                all_responses_dict.append(self._serialize_api_response(resp))

        # Serialize conversation history (handle response objects)
        conversation_history_dict = []
        if conversation_history:
            for item in conversation_history:
                try:
                    if hasattr(item, 'choices') or hasattr(item, 'output'):  # API response object
                        conversation_history_dict.append(self._serialize_api_response(item))
                    else:  # Regular dict message
                        conversation_history_dict.append(item)
                except Exception as e:
                    conversation_history_dict.append({'error': f'Failed to serialize conversation item: {str(e)}'})
        
        return {
            'task_id': task_id,
            'model': self.model,
            'reasoning_effort': self.reasoning_effort,
            'api_type': 'chat_completions_multiturn',
            'program': '',
            'task_failure_reason': 'API timeout after retries',
            'timed_out': True,
            'tokens_used': total_tokens,
            'request_cost': total_cost,
            'turns_used': turns_completed,
            'raw_response': None,
            'score': {
                'correct': False,
                'pixel_accuracy': 0.0,
                'total_pixels': total_pixels,
                'correct_pixels': 0,
                'error': 'API timeout after retries'
            },
            'actual_output': actual_output,
            'api_success': False,  # This is a timeout failure, not a regular failure
            'timeout_failure': True,  # NEW: Mark as timeout failure
            # NEW: Complete multi-turn conversation data
            'multiturn_data': {
                'conversation_history': conversation_history_dict,
                'all_responses': all_responses_dict,
                'turn_details': turn_details or [],
                'total_turns': turns_completed
            }
        }

    def run_task(self, task_id: str, task_data: Dict, total_tasks: int = 1) -> Dict:
        """Run a single ARC task using multi-turn local code execution"""
        if self.debug:
            print(f"🔍 DEBUG TASK: Starting run_task for {task_id}")
        
        try:
            # Validate task_data structure
            if not isinstance(task_data, dict):
                raise ValueError(f"task_data is not a dict, got {type(task_data)}")
            
            if 'train' not in task_data:
                raise ValueError("task_data missing 'train' key")
            
            if 'test' not in task_data:
                raise ValueError("task_data missing 'test' key")
            
            if not isinstance(task_data['train'], list):
                raise ValueError(f"task_data['train'] is not a list, got {type(task_data['train'])}")
            
            if not isinstance(task_data['test'], list):
                raise ValueError(f"task_data['test'] is not a list, got {type(task_data['test'])}")
            
            if len(task_data['test']) == 0:
                raise ValueError("task_data['test'] is empty")
            
            if self.debug:
                print(f"🔍 DEBUG TASK: Task data validation passed for {task_id}")
                print(f"🔍 DEBUG TASK: Train examples: {len(task_data['train'])}, Test examples: {len(task_data['test'])}")
            
            if self.max_workers == 1:  # Only print for sequential execution
                print(f"\nProcessing task: {task_id}")
            
            # Apply rate limiting if configured
            if self.rate_limit_delay > 0:
                time.sleep(self.rate_limit_delay)
            
            if self.debug:
                print(f"🔍 DEBUG TASK: About to choose execution mode. independent_attempts = {self.independent_attempts}")
            
            # Choose execution mode based on independent_attempts flag
            if self.independent_attempts:
                if self.debug:
                    print(f"🔍 DEBUG TASK: Calling run_task_independent_attempts for {task_id}")
                result = self.run_task_independent_attempts(task_id, task_data, total_tasks)
            else:
                if self.debug:
                    print(f"🔍 DEBUG TASK: Calling run_task_multiturn for {task_id}")
                result = self.run_task_multiturn(task_id, task_data, total_tasks)
            
            if self.debug:
                print(f"🔍 DEBUG TASK: Execution completed for {task_id}, result type: {type(result)}")
            return result
            
        except Exception as e:
            if self.debug:
                print(f"🔍 DEBUG TASK: Exception in run_task for {task_id}: {type(e).__name__}: {str(e)}")
                import traceback
                print(f"🔍 DEBUG TASK: Full traceback:")
                traceback.print_exc()
            
            # Create a failure result for the exception
            return {
                'task_id': task_id,
                'model': self.model,
                'reasoning_effort': self.reasoning_effort,
                'api_type': 'chat_completions_independent_attempts' if self.independent_attempts else 'chat_completions_multiturn',
                'program': '',
                'task_failure_reason': f'Task setup failed: {str(e)}',
                'timed_out': False,
                'tokens_used': 0,
                'request_cost': 0.0,
                'turns_used': 0,
                'raw_response': None,
                'score': {
                    'correct': False,
                    'pixel_accuracy': 0.0,
                    'total_pixels': 0,
                    'correct_pixels': 0,
                    'error': f'Task setup failed: {str(e)}'
                },
                'api_success': False,
            }
    
    def run_task_independent_attempts(self, task_id: str, task_data: Dict, total_tasks: int = 1) -> Dict:
        """Run independent attempts without feedback - multiple fresh starts with the same initial prompt"""
        if self.debug:
            print(f"🔍 DEBUG INDEPENDENT: Starting independent attempts for task {task_id}")
            print(f"🔍 DEBUG INDEPENDENT: max_turns = {self.max_turns}")
        
        total_cost = 0.0
        total_tokens = 0
        all_responses = []
        attempt_details = []  # Track details of each attempt
        
        try:
            if self.debug:
                print(f"🔍 DEBUG INDEPENDENT: About to start attempt loop for {task_id}")
            for attempt in range(self.max_turns):
                if self.debug:
                    print(f"🔍 DEBUG INDEPENDENT: Starting attempt {attempt + 1}/{self.max_turns} for {task_id}")
                attempt_start_time = datetime.datetime.now()
                if self.max_workers == 1:  # Only print detailed logs for sequential execution
                    print(f"  🔄 Attempt {attempt + 1}/{self.max_turns}...")
                
                # Create fresh conversation for each attempt
                if self.debug:
                    print(f"🔍 DEBUG INDEPENDENT: Creating system message for {task_id}")
                system_msg = {"role": "system", "content": "You are an expert at solving abstract reasoning puzzles. Write clean, efficient Python code."}
                if self.debug:
                    print(f"🔍 DEBUG INDEPENDENT: Creating initial prompt for {task_id}")
                initial_prompt_messages = self.create_prompt(task_data, is_first_turn=True, task_id=task_id, turn=attempt + 1)
                if self.debug:
                    print(f"🔍 DEBUG INDEPENDENT: Prompt created successfully for {task_id}")
                conversation_history = [system_msg, {"role": "user", "content": initial_prompt_messages}]
                
                # Make API call with timeout and retry logic
                if self.debug:
                    print(f"🔍 DEBUG INDEPENDENT: About to make API call for {task_id} attempt {attempt + 1}")
                response = None
                api_call_successful = False
                
                for retry_attempt in range(3):  # 3 attempts total (initial + 2 retries)
                    try:
                        if self.debug:
                            if retry_attempt == 0:
                                print(f"🔍 DEBUG INDEPENDENT: API call (first attempt) for {task_id} attempt {attempt + 1}")
                            else:
                                print(f"🔍 DEBUG INDEPENDENT: API call (retry {retry_attempt}/2) for {task_id} attempt {attempt + 1}")
                        
                        if self.max_workers == 1 and retry_attempt > 0:
                            print(f"     🔄 Attempt {attempt + 1} retry {retry_attempt}/2...")
                        
                        response = execute_with_timeout(self.call_chat_completions_api, conversation_history, timeout=1000)
                        api_call_successful = True
                        break  # Success!
                        
                    except Exception as e:
                        error_type = type(e).__name__
                        error_msg = str(e)
                        if retry_attempt < 2:  # Can still retry
                            if self.max_workers == 1:
                                print(f"     ⏰ Attempt {attempt + 1} retry {retry_attempt + 1} failed ({error_type}: {error_msg}), retrying in 2s...")
                            if self.debug:
                                print(f"🔍 DEBUG INDEPENDENT: API retry {retry_attempt + 1}/2 for {task_id} attempt {attempt + 1} - Error: {error_type}: {error_msg}")
                            time.sleep(2)  # Brief backoff
                            continue
                        else:  # All retries exhausted
                            if self.max_workers == 1:
                                print(f"     ❌ Attempt {attempt + 1} failed after 3 retries: {error_type}: {error_msg}")
                            if self.debug:
                                print(f"🔍 DEBUG INDEPENDENT: All retries exhausted for {task_id} attempt {attempt + 1} - Final error: {error_type}: {error_msg}")
                            # Return timeout failure result
                            self._update_costs(total_cost, total_tokens)
                            return self.create_timeout_failure_result(task_id, total_cost, total_tokens, attempt, task_data, conversation_history, all_responses, attempt_details)
                
                if not api_call_successful or response is None:
                    # This shouldn't happen, but just in case
                    self._update_costs(total_cost, total_tokens)
                    return self.create_timeout_failure_result(task_id, total_cost, total_tokens, attempt, task_data, conversation_history, all_responses, attempt_details)
                
                all_responses.append(response)
                
                # Track costs
                usage = response.usage
                input_rate, output_rate = self.get_model_pricing(self.model)
                input_tokens = usage.prompt_tokens if usage else 0
                output_tokens = usage.completion_tokens if usage else 0
                attempt_cost = (input_tokens / 1_000_000) * input_rate + (output_tokens / 1_000_000) * output_rate
                total_cost += attempt_cost
                total_tokens += usage.total_tokens if usage else 0
                
                if self.max_workers == 1:
                    print(f"     💰 Attempt cost: ${attempt_cost:.6f} (input: {input_tokens}, output: {output_tokens})")
                
                # Extract code
                program = self.extract_code_from_response(response)
                
                # Initialize attempt detail
                attempt_detail = {
                    'attempt_number': attempt + 1,
                    'timestamp': attempt_start_time.isoformat(),
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'attempt_cost': attempt_cost,
                    'program_extracted': bool(program),
                    'program': program,
                    'test_result': None,
                    'status': 'in_progress'
                }
                
                if not program:
                    if self.max_workers == 1:
                        print(f"     ❌ No code found in response")
                    
                    attempt_detail['status'] = 'no_code_found'
                    attempt_details.append(attempt_detail)
                    
                    # Continue to next attempt (no feedback in independent mode)
                    continue
                
                # Test on test input
                test_input = task_data['test'][0]['input']
                test_expected = task_data['test'][0]['output']
                predicted_output, error, timed_out = self.executor.execute_program(program, test_input)
                
                if predicted_output is not None and not error and not timed_out:
                    # Check if test is correct
                    test_score = self.scorer.score_grid(predicted_output, test_expected)
                    attempt_detail['test_result'] = test_score
                    
                    if test_score['correct']:
                        # SUCCESS! 
                        attempt_detail['status'] = 'success'
                        attempt_details.append(attempt_detail)
                        
                        if self.max_workers == 1:
                            print(f"     ✅ Perfect solution found on attempt {attempt + 1}!")
                        
                        # Update costs only (progress handled by parallel executor)
                        self._update_costs(total_cost, total_tokens)
                        
                        # Add actual outputs to test_score for logging
                        test_score['predicted_output'] = predicted_output
                        test_score['actual_output'] = test_expected
                        
                        return self.create_success_result(task_id, program, response, test_score, total_cost, total_tokens, attempt + 1, task_data, None, all_responses, attempt_details, is_independent_attempts=True)
                else:
                    # Execution failed - record the failure and continue to next attempt
                    attempt_detail['test_result'] = {
                        'task_failure_reason': error,
                        'timed_out': timed_out,
                        'predicted_output': predicted_output
                    }
                    attempt_detail['status'] = 'failed_test'
                    attempt_details.append(attempt_detail)
                    
                    if self.max_workers == 1:
                        print(f"     📊 Attempt {attempt + 1} failed test - continuing to next attempt")
                    
                    # Continue to next attempt (no feedback in independent mode)
                    continue
                
                # If we get here, the test passed but wasn't correct - also continue to next attempt
                attempt_detail['status'] = 'failed_test'
                attempt_details.append(attempt_detail)
                
                if self.max_workers == 1:
                    print(f"     📊 Attempt {attempt + 1} failed test - continuing to next attempt")
            
            # Failed after all attempts
            if self.debug:
                print(f"🔍 DEBUG INDEPENDENT: Completed all {self.max_turns} attempts for {task_id}")
                print(f"🔍 DEBUG INDEPENDENT: Total responses collected: {len(all_responses)}")
                print(f"🔍 DEBUG INDEPENDENT: Total cost: ${total_cost:.6f}")
            self._update_costs(total_cost, total_tokens)
            
            return self.create_failure_result(task_id, program if 'program' in locals() else "", all_responses, total_cost, total_tokens, self.max_turns, task_data, "All attempts failed", None, attempt_details, is_independent_attempts=True)
        
        except Exception as e:
            if self.debug:
                print(f"🔍 DEBUG INDEPENDENT: Exception in independent attempts for {task_id}: {type(e).__name__}: {str(e)}")
                import traceback
                print(f"🔍 DEBUG INDEPENDENT: Full traceback:")
                traceback.print_exc()
            print(f"     ❌ Independent attempts execution failed: {e}")
            self._update_costs(total_cost, total_tokens)
            
            return self.create_failure_result(task_id, "", all_responses, total_cost, total_tokens, len(attempt_details), task_data, str(e), None, attempt_details, is_independent_attempts=True)

    def run_task_multiturn(self, task_id: str, task_data: Dict, total_tasks: int = 1) -> Dict:
        """Run multi-turn conversation with local code execution"""
        conversation_history = []
        total_cost = 0.0
        total_tokens = 0
        all_responses = []
        turn_details = []  # NEW: Collect detailed turn-by-turn data
        
        # Start conversation
        system_msg = {"role": "system", "content": "You are an expert at solving abstract reasoning puzzles. Write clean, efficient Python code."}
        initial_prompt_messages = self.create_prompt(task_data, is_first_turn=True, task_id=task_id, turn=1)
        conversation_history = [system_msg, {"role": "user", "content": initial_prompt_messages}]
        
        try:
            for turn in range(self.max_turns):
                turn_start_time = datetime.datetime.now()
                if self.max_workers == 1:  # Only print detailed logs for sequential execution
                    print(f"  🔄 Turn {turn + 1}/{self.max_turns}...")
                
                # Make API call with timeout and retry logic
                response = None
                api_call_successful = False
                
                for attempt in range(3):  # 3 attempts total (initial + 2 retries)
                    try:
                        if self.max_workers == 1 and attempt > 0:
                            print(f"     🔄 Turn {turn + 1} retry {attempt}/2...")
                        
                        response = execute_with_timeout(self.call_chat_completions_api, conversation_history, timeout=1000)
                        api_call_successful = True
                        break  # Success!
                        
                    except Exception as e:
                        error_type = type(e).__name__
                        error_msg = str(e)
                        if attempt < 2:  # Can still retry
                            if self.max_workers == 1:
                                if attempt == 0:
                                    print(f"     ⏰ Turn {turn + 1} first attempt failed ({error_type}: {error_msg}), retrying in 2s...")
                                else:
                                    print(f"     ⏰ Turn {turn + 1} retry {attempt} failed ({error_type}: {error_msg}), retrying in 2s...")
                            if self.debug:
                                print(f"🔍 DEBUG MULTITURN: API retry {attempt + 1}/2 for {task_id} turn {turn + 1} - Error: {error_type}: {error_msg}")
                            time.sleep(2)  # Brief backoff
                            continue
                        else:  # All retries exhausted
                            if self.max_workers == 1:
                                print(f"     ❌ Turn {turn + 1} failed after 3 attempts: {error_type}: {error_msg}")
                            if self.debug:
                                print(f"🔍 DEBUG MULTITURN: All retries exhausted for {task_id} turn {turn + 1} - Final error: {error_type}: {error_msg}")
                            # Return timeout failure result
                            self._update_costs(total_cost, total_tokens)
                            return self.create_timeout_failure_result(task_id, total_cost, total_tokens, turn, task_data, conversation_history, all_responses, turn_details)
                
                if not api_call_successful or response is None:
                    # This shouldn't happen, but just in case
                    self._update_costs(total_cost, total_tokens)
                    return self.create_timeout_failure_result(task_id, total_cost, total_tokens, turn, task_data, conversation_history, all_responses, turn_details)
                
                all_responses.append(response)
                
                # Track costs
                usage = response.usage
                input_rate, output_rate = self.get_model_pricing(self.model)
                input_tokens = usage.prompt_tokens if usage else 0
                output_tokens = usage.completion_tokens if usage else 0
                turn_cost = (input_tokens / 1_000_000) * input_rate + (output_tokens / 1_000_000) * output_rate
                total_cost += turn_cost
                total_tokens += usage.total_tokens if usage else 0
                
                if self.max_workers == 1:
                    print(f"     💰 Turn cost: ${turn_cost:.6f} (input: {input_tokens}, output: {output_tokens})")
                
                # Extract code
                program = self.extract_code_from_response(response)
                
                # Initialize turn detail
                turn_detail = {
                    'turn_number': turn + 1,
                    'timestamp': turn_start_time.isoformat(),
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'turn_cost': turn_cost,
                    'program_extracted': bool(program),
                    'program': program,
                    'training_feedback': None,
                    'test_result': None,
                    'status': 'in_progress'
                }
                
                if not program:
                    if self.max_workers == 1:
                        print(f"     ❌ No code found in response")
                    
                    turn_detail['status'] = 'no_code_found'
                    turn_details.append(turn_detail)
                    
                    # Stop if this is the last turn
                    if turn == self.max_turns - 1:
                        if self.max_workers == 1:
                            print(f"     ⏰ Max turns ({self.max_turns}) reached")
                        
                        # Add assistant response to conversation history before breaking
                        if response.choices and response.choices[0].message:
                            conversation_history.append({"role": "assistant", "content": response.choices[0].message.content})
                        
                        break
                    
                    # Continue conversation with request for code
                    # Add assistant response to conversation history
                    if response.choices and response.choices[0].message:
                        conversation_history.append({"role": "assistant", "content": response.choices[0].message.content})
                    
                    # Add request for code
                    code_request = """I need you to provide Python code to attempt this task. Even if you're not completely certain about the pattern, please provide your best hypothesis as a working transformation function.

**Remember**: A partial solution that captures some observed patterns is much more valuable than refusing to attempt the task. Your goal is to implement whatever understanding you have, even if incomplete.

You MUST end your response with the following exact format:

Final answer:
```python
def transform(grid):
    # Your transformation logic here (implement your best understanding)
    return transformed_grid
```

Make sure to include the function definition inside a proper code block."""
                    
                    conversation_history.append({"role": "user", "content": code_request})
                    turn_detail['code_request_sent'] = True
                    if self.max_workers == 1:
                        print(f"     💬 Requesting code from model...")
                    continue
                
                # Test on test input first
                test_input = task_data['test'][0]['input']
                test_expected = task_data['test'][0]['output']
                predicted_output, error, timed_out = self.executor.execute_program(program, test_input)
                
                # Determine test correctness for training feedback
                test_correct = None
                if predicted_output is not None and not error and not timed_out:
                    test_score = self.scorer.score_grid(predicted_output, test_expected)
                    test_correct = test_score['correct']
                else:
                    test_correct = False
                
                # Get training feedback for this turn (including test result context)
                training_feedback_messages, solved_count, pixel_acc = self.create_training_feedback(program, task_data['train'], test_correct, task_id, turn + 1)
                turn_detail['training_feedback'] = {
                    'feedback_messages': training_feedback_messages,
                    'solved_count': solved_count,
                    'total_training_examples': len(task_data['train']),
                    'pixel_accuracy': pixel_acc
                }
                
                if predicted_output is not None and not error and not timed_out:
                    # Check if test is correct
                    test_score = self.scorer.score_grid(predicted_output, test_expected)
                    turn_detail['test_result'] = test_score
                    
                    if test_score['correct']:
                        # SUCCESS! 
                        turn_detail['status'] = 'success'
                        turn_details.append(turn_detail)
                        
                        if self.max_workers == 1:
                            print(f"     ✅ Perfect solution found! Training: {solved_count}/{len(task_data['train'])} solved, {pixel_acc:.1%} accuracy")
                        
                        # Update costs only (progress handled by parallel executor)
                        self._update_costs(total_cost, total_tokens)
                        
                        # Add actual outputs to test_score for logging
                        test_score['predicted_output'] = predicted_output
                        test_score['actual_output'] = test_expected
                        
                        return self.create_success_result(task_id, program, response, test_score, total_cost, total_tokens, turn + 1, task_data, conversation_history, all_responses, turn_details)
                else:
                    # Execution failed
                    turn_detail['test_result'] = {
                        'task_failure_reason': error,
                        'timed_out': timed_out,
                        'predicted_output': predicted_output
                    }
                
                turn_detail['status'] = 'failed_test'
                turn_details.append(turn_detail)
                
                if self.max_workers == 1:
                    print(f"     📊 Training: {solved_count}/{len(task_data['train'])} solved, {pixel_acc:.1%} accuracy")
                
                # Stop if this is the last turn
                if turn == self.max_turns - 1:
                    if self.max_workers == 1:
                        print(f"     ⏰ Max turns ({self.max_turns}) reached")
                    
                    # Add final response to conversation history before breaking
                    if response.choices and response.choices[0].message:
                        conversation_history.append({"role": "assistant", "content": response.choices[0].message.content})
                    
                    break
                
                # Continue conversation with feedback
                # Add assistant response to conversation history
                if response.choices and response.choices[0].message:
                    conversation_history.append({"role": "assistant", "content": response.choices[0].message.content})
                
                # Add training feedback - combine prompt messages with feedback messages
                subsequent_prompt_messages = self.create_prompt(task_data, is_first_turn=False, task_id=task_id, turn=turn + 2)
                combined_feedback_content = subsequent_prompt_messages + training_feedback_messages
                conversation_history.append({"role": "user", "content": combined_feedback_content})
                turn_detail['feedback_sent'] = True
            
            # Failed after max turns
            self._update_costs(total_cost, total_tokens)
            
            return self.create_failure_result(task_id, program if 'program' in locals() else "", all_responses, total_cost, total_tokens, self.max_turns, task_data, "Max turns reached", conversation_history, turn_details)
        
        except Exception as e:
            print(f"     ❌ Multi-turn execution failed: {e}")
            self._update_costs(total_cost, total_tokens)
            
            return self.create_failure_result(task_id, "", all_responses, total_cost, total_tokens, len(turn_details), task_data, str(e), conversation_history, turn_details)

    def run_subset(self, subset_name: str, dataset: str = "arc-agi-1", limit: Optional[int] = None) -> List[Dict]:
        """Run all tasks in a subset with optional parallelization"""
        if self.debug:
            print(f"🔍 DEBUG SUBSET: Loading tasks from {dataset}/{subset_name}")
        
        try:
            tasks = self.task_loader.load_tasks_from_subset(subset_name, dataset)
            if self.debug:
                print(f"🔍 DEBUG SUBSET: Loaded {len(tasks)} tasks successfully")
            
            if limit:
                tasks = tasks[:limit]
                if self.debug:
                    print(f"🔍 DEBUG SUBSET: Limited to {len(tasks)} tasks")
            
            total_tasks = len(tasks)
            if self.debug:
                print(f"🔍 DEBUG SUBSET: Total tasks to process: {total_tasks}")
                
                # Debug first few task IDs
                if tasks:
                    first_few = [task_id for task_id, _ in tasks[:3]]
                    print(f"🔍 DEBUG SUBSET: First few task IDs: {first_few}")
            
        except Exception as e:
            if self.debug:
                print(f"🔍 DEBUG SUBSET: Error loading tasks: {type(e).__name__}: {str(e)}")
                import traceback
                traceback.print_exc()
            return []
        
        # Print configuration info
        print(f"\nRunning {total_tasks} tasks from {dataset}/{subset_name}")
        print(f"Model: {self.model}")
        print(f"Reasoning effort: {self.reasoning_effort}")
        
        # Show execution mode
        if self.independent_attempts:
            print(f"API: Chat Completions (independent attempts, max {self.max_turns} attempts)")
            print("Mode: Independent attempts - multiple fresh starts, no feedback")
        else:
            print(f"API: Chat Completions (multi-turn, max {self.max_turns} turns)")
            print("Mode: Multi-turn feedback - conversation with training examples")
        
        # Show visual/text mode
        print("Input mode: Text-only")
            
        if self.max_workers > 1:
            print(f"Parallelization: ENABLED ({self.max_workers} workers)")
            if self.rate_limit_delay > 0:
                print(f"Rate limiting: {self.rate_limit_delay}s delay between requests")
        else:
            print("Parallelization: DISABLED (sequential execution)")
        print("-" * 50)
        
        # Reset progress counter for this subset
        with self._progress_lock:
            self.completed_tasks = 0
        
        results = []
        
        if self.max_workers == 1:
            # Sequential execution (original behavior)
            for task_id, task_data in tasks:
                result = self.run_task(task_id, task_data, total_tasks)
                results.append(result)
                
                # Save individual result
                self.save_result(result)
        else:
            # Parallel execution
            print(f"Starting parallel execution with {self.max_workers} workers...")
            
            def process_task(task_info):
                task_id, task_data = task_info
                if self.debug:
                    print(f"🔍 DEBUG PARALLEL: [Thread {threading.get_ident()}] Starting process_task for {task_id}")
                
                try:
                    if self.max_workers > 1:
                        print(f"[{task_id}] Starting...")
                    
                    if self.debug:
                        print(f"🔍 DEBUG PARALLEL: [Thread {threading.get_ident()}] About to call run_task for {task_id}")
                    result = self.run_task(task_id, task_data, total_tasks)
                    if self.debug:
                        print(f"🔍 DEBUG PARALLEL: [Thread {threading.get_ident()}] run_task completed for {task_id}")
                    
                    # Save individual result
                    self.save_result(result)
                    if self.debug:
                        print(f"🔍 DEBUG PARALLEL: [Thread {threading.get_ident()}] Result saved for {task_id}")
                    
                    if self.max_workers > 1:
                        status = "✅ SOLVED" if result.get('score', {}).get('correct', False) else "❌ FAILED"
                        cost = result.get('request_cost', 0.0)
                        turns = result.get('turns_used', 1)
                        units = "attempts" if self.independent_attempts else "turns"
                        print(f"[{task_id}] {status} (${cost:.6f}, {turns} {units})")
                    
                    return result
                    
                except Exception as e:
                    if self.debug:
                        print(f"🔍 DEBUG PARALLEL: [Thread {threading.get_ident()}] Exception in process_task for {task_id}: {type(e).__name__}: {str(e)}")
                        import traceback
                        traceback.print_exc()
                    
                    # Return a minimal error result
                    return {
                        'task_id': task_id,
                        'model': self.model,
                        'reasoning_effort': self.reasoning_effort,
                        'task_failure_reason': f'process_task failed: {str(e)}',
                        'tokens_used': 0,
                        'request_cost': 0.0,
                        'turns_used': 0,
                        'score': {'correct': False, 'pixel_accuracy': 0.0, 'total_pixels': 0, 'correct_pixels': 0},
                        'api_success': False,
                    }
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_task = {executor.submit(process_task, task_info): task_info[0] 
                                for task_info in tasks}
                
                # Collect results as they complete
                completed = 0
                for future in as_completed(future_to_task):
                    task_id = future_to_task[future]
                    completed += 1
                    try:
                        result = future.result()
                        results.append(result)
                        status = "✅ COMPLETED" if result.get('score', {}).get('correct', False) else "❌ FAILED"
                        turns = result.get('turns_used', 1)
                        units = "attempts" if self.independent_attempts else "turns"
                        status_with_turns = f"{status} ({turns} {units})"
                    except Exception as e:
                        print(f"[{task_id}] ❌ SAVE FAILED: {e}")
                        # Create a minimal error result
                        error_result = {
                            'task_id': task_id,
                            'error': str(e),
                            'score': {'correct': False, 'pixel_accuracy': 0.0, 'total_pixels': 0, 'correct_pixels': 0}
                        }
                        results.append(error_result)
                        status_with_turns = "❌ SAVE FAILED"
                    
                    # Show overall progress
                    progress_pct = (completed / total_tasks) * 100
                    print(f"Progress: {completed}/{total_tasks} tasks processed ({progress_pct:.1f}%) - {status_with_turns}")
            
            # Sort results by task_id to maintain consistent order
            results.sort(key=lambda x: x.get('task_id', ''))
            print(f"\nParallel execution completed. All {total_tasks} tasks processed.")
        
        # Save summary
        self.save_summary(results, subset_name, dataset)
        
        return results
    
    def save_result(self, result: Dict):
        """Save individual task result with thread-safe unique filename"""
        # Add microseconds and thread ID to ensure unique filenames in parallel execution
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        thread_id = threading.get_ident()
        
        # Include run number in filename when doing repeated runs
        if self.run_number > 0:
            filename = f"{timestamp}_{thread_id}_{result['task_id']}_run{self.run_number}.json"
        else:
            filename = f"{timestamp}_{thread_id}_{result['task_id']}.json"
        
        filepath = self.logs_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(result, f, indent=2)
    
    def save_summary(self, results: List[Dict], subset_name: str, dataset: str):
        """Save summary of all results"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Calculate statistics
        total_tasks = len(results)
        correct_tasks = sum(1 for r in results if r.get('score', {}).get('correct', False))
        
        # Separate successful API calls from complete failures and timeout failures
        api_successes = [r for r in results if r.get('api_success', True)]  # Default True for backward compatibility
        api_failures = [r for r in results if not r.get('api_success', True) and not r.get('timeout_failure', False)]
        timeout_failures = [r for r in results if r.get('timeout_failure', False)]
        failed_tasks = len(api_failures)
        timeout_tasks = len(timeout_failures)
        successful_api_calls = len(api_successes)
        
        total_pixels = sum(r.get('score', {}).get('total_pixels', 0) for r in results)
        correct_pixels = sum(r.get('score', {}).get('correct_pixels', 0) for r in results)
        

        
        # Calculate turn usage statistics
        total_turns_used = sum(r.get('turns_used', 1) for r in results)  # Default to 1 for single-shot
        avg_turns_used = total_turns_used / total_tasks if total_tasks > 0 else 0
        
        summary = {
            'timestamp': timestamp,
            'dataset': dataset,
            'subset': subset_name,
            'model': self.model,
            'reasoning_effort': self.reasoning_effort,
            'api_type': 'chat_completions',
            'run_number': self.run_number,  # NEW: Include run number
            'total_tasks': total_tasks,
            'successful_api_calls': successful_api_calls,
            'failed_api_calls': failed_tasks,
            'timeout_failed_tasks': timeout_tasks,
            'correct_tasks': correct_tasks,
            'task_accuracy': correct_tasks / total_tasks if total_tasks > 0 else 0.0,
            'success_rate': successful_api_calls / total_tasks if total_tasks > 0 else 0.0,
            'total_pixels': total_pixels,
            'correct_pixels': correct_pixels,
            'pixel_accuracy': correct_pixels / total_pixels if total_pixels > 0 else 0.0,
            'total_turns_used': total_turns_used,
            'avg_turns_used': avg_turns_used,
            'total_tokens': self.total_tokens,
            'total_cost': self.total_cost,
            'results': results
        }
        
        # Include run number in filename when doing repeated runs
        if self.run_number > 0:
            filename = f"{timestamp}_summary_{dataset}_{subset_name}_run{self.run_number}.json"
        else:
            filename = f"{timestamp}_summary_{dataset}_{subset_name}.json"
        
        filepath = self.logs_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        run_info = f" (Run {self.run_number})" if self.run_number > 0 else ""
        print("\n" + "="*50)
        print(f"SUMMARY{run_info}")
        print("="*50)
        print(f"Dataset: {dataset}")
        print(f"Subset: {subset_name}")
        print(f"Model: {self.model}")
        print(f"Reasoning effort: {self.reasoning_effort}")
        
        if self.independent_attempts:
            print(f"API: Chat Completions (independent attempts, max {self.max_turns} attempts)")
        else:
            print(f"API: Chat Completions (multi-turn, max {self.max_turns} turns)")
        print(f"Total tasks attempted: {total_tasks}")
        print(f"Successful API calls: {successful_api_calls}/{total_tasks} ({summary['success_rate']:.1%})")
        if failed_tasks > 0:
            print(f"Failed API calls: {failed_tasks}/{total_tasks} ({failed_tasks/total_tasks:.1%}) ❌")
        if timeout_tasks > 0:
            print(f"Timeout failures: {timeout_tasks}/{total_tasks} ({timeout_tasks/total_tasks:.1%}) ⏰")
        print(f"Tasks solved correctly: {correct_tasks}/{total_tasks} ({summary['task_accuracy']:.1%})")
        print(f"Pixel accuracy: {correct_pixels}/{total_pixels} ({summary['pixel_accuracy']:.1%})")
        units = "attempts" if self.independent_attempts else "turns"
        print(f"Total {units} used: {total_turns_used}")
        print(f"Average {units} per task: {avg_turns_used:.1f}")
        print(f"Total tokens used: {self.total_tokens:,}")
        print(f"Total cost: ${self.total_cost:.6f}")
        
        # List failed tasks for debugging
        if failed_tasks > 0:
            print(f"\n❌ FAILED TASKS ({failed_tasks}):")
            for result in api_failures:
                task_id = result.get('task_id', 'unknown')
                error = result.get('error', 'Unknown error')
                print(f"  - {task_id}: {error}")
        
        if timeout_tasks > 0:
            print(f"\n⏰ TIMEOUT FAILURES ({timeout_tasks}):")
            for result in timeout_failures:
                task_id = result.get('task_id', 'unknown')
                turns_completed = result.get('turns_used', 0)
                units = "attempts" if self.independent_attempts else "turns"
                print(f"  - {task_id}: API timeout after {turns_completed} {units} and 3 retries")
        
        print(f"\nResults saved to: {filepath}")

    def run_repeated_subset(self, subset_name: str, dataset: str = "arc-agi-1", limit: Optional[int] = None, repeat_runs: int = 3) -> List[List[Dict]]:
        """Run the same subset multiple times and calculate aggregate statistics"""
        print(f"\nRunning {repeat_runs} repeated tests of {dataset}/{subset_name}")
        print(f"Model: {self.model}")
        print(f"Reasoning effort: {self.reasoning_effort}")
        
        # Show execution mode
        if self.independent_attempts:
            print(f"API: Chat Completions (independent attempts, max {self.max_turns} attempts)")
            print("Mode: Independent attempts - multiple fresh starts, no feedback")
        else:
            print(f"API: Chat Completions (multi-turn, max {self.max_turns} turns)")
            print("Mode: Multi-turn feedback - conversation with training examples")
        
        # Show visual/text mode
        print("Input mode: Text-only")
            
        if self.max_workers > 1:
            print(f"Parallelization: ENABLED ({self.max_workers} workers)")
        else:
            print(f"Parallelization: DISABLED (sequential execution)")
        print("="*70)
        
        all_run_results = []
        
        # Run the subset multiple times
        for run_num in range(1, repeat_runs + 1):
            print(f"\n🚀 STARTING RUN {run_num}/{repeat_runs}")
            print("-" * 50)
            
            # Create a new runner for this run to ensure clean state
            runner = ARCTaskRunner(
                model=self.model,
                max_workers=self.max_workers,
                rate_limit_delay=self.rate_limit_delay,
                max_turns=self.max_turns,
                run_number=run_num,
                independent_attempts=self.independent_attempts,
                base_url=self.base_url,
                reasoning_effort=self.reasoning_effort,
                debug=self.debug,
                qwen_no_think=self.qwen_no_think
            )
            
            # Run the subset
            results = runner.run_subset(subset_name, dataset, limit)
            all_run_results.append(results)
            
            print(f"\n✅ COMPLETED RUN {run_num}/{repeat_runs}")
        
        # Calculate and display aggregate statistics
        self._calculate_and_display_aggregate_stats(all_run_results, subset_name, dataset, repeat_runs)
        
        return all_run_results
    
    def _calculate_and_display_aggregate_stats(self, all_run_results: List[List[Dict]], subset_name: str, dataset: str, repeat_runs: int):
        """Calculate and display mean and standard deviation across multiple runs"""
        
        # Calculate statistics for each run
        run_stats = []
        
        for run_num, results in enumerate(all_run_results, 1):
            # Filter out API failures (timeout failures)
            successful_api_results = [r for r in results if r.get('api_success', True)]
            
            if not successful_api_results:
                # Handle case where all API calls failed
                run_stats.append({
                    'run_number': run_num,
                    'attempted_tasks': 0,
                    'turn1_solved': 0,
                    'all_turns_solved': 0,
                    'turn1_success_rate': 0.0,
                    'all_turns_success_rate': 0.0
                })
                continue
            
            attempted_tasks = len(successful_api_results)
            
            # Count tasks solved on turn 1 only
            turn1_solved = sum(1 for r in successful_api_results 
                             if r.get('score', {}).get('correct', False) and r.get('turns_used', 1) == 1)
            
            # Count tasks solved by end of all turns
            all_turns_solved = sum(1 for r in successful_api_results 
                                 if r.get('score', {}).get('correct', False))
            
            # Calculate success rates
            turn1_success_rate = turn1_solved / attempted_tasks if attempted_tasks > 0 else 0.0
            all_turns_success_rate = all_turns_solved / attempted_tasks if attempted_tasks > 0 else 0.0
            
            run_stats.append({
                'run_number': run_num,
                'attempted_tasks': attempted_tasks,
                'turn1_solved': turn1_solved,
                'all_turns_solved': all_turns_solved,
                'turn1_success_rate': turn1_success_rate,
                'all_turns_success_rate': all_turns_success_rate
            })
        
        # Calculate aggregate statistics
        if run_stats:
            turn1_rates = [s['turn1_success_rate'] for s in run_stats]
            all_turns_rates = [s['all_turns_success_rate'] for s in run_stats]
            
            turn1_mean = np.mean(turn1_rates)
            turn1_std = np.std(turn1_rates, ddof=1) if len(turn1_rates) > 1 else 0.0
            
            all_turns_mean = np.mean(all_turns_rates)
            all_turns_std = np.std(all_turns_rates, ddof=1) if len(all_turns_rates) > 1 else 0.0
            
            # Save aggregate summary
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            aggregate_summary = {
                'timestamp': timestamp,
                'dataset': dataset,
                'subset': subset_name,
                'model': self.model,
                'reasoning_effort': self.reasoning_effort,
                'repeat_runs': repeat_runs,
                'run_statistics': run_stats,
                'turn1_success_rate_mean': turn1_mean,
                'turn1_success_rate_std': turn1_std,
                'all_turns_success_rate_mean': all_turns_mean,
                'all_turns_success_rate_std': all_turns_std
            }
            
            filename = f"{timestamp}_aggregate_summary_{dataset}_{subset_name}_{repeat_runs}runs.json"
            filepath = self.logs_dir / filename
            
            with open(filepath, 'w') as f:
                json.dump(aggregate_summary, f, indent=2)
            
            # Display results
            print("\n" + "="*70)
            print("AGGREGATE STATISTICS ACROSS MULTIPLE RUNS")
            print("="*70)
            print(f"Dataset: {dataset}")
            print(f"Subset: {subset_name}")
            print(f"Model: {self.model}")
            print(f"Number of runs: {repeat_runs}")
            print(f"API failures excluded from analysis: YES")
            print("")
            
            # Individual run results
            print("INDIVIDUAL RUN RESULTS:")
            print("-" * 70)
            if self.independent_attempts:
                print(f"{'Run':<4} {'Attempted':<10} {'Attempt 1 Only':<14} {'All Attempts':<14} {'Attempt 1 Rate':<14} {'All Attempts Rate':<14}")
            else:
                print(f"{'Run':<4} {'Attempted':<10} {'Turn 1 Only':<12} {'All Turns':<12} {'Turn 1 Rate':<12} {'All Turns Rate':<12}")
            print("-" * 70)
            
            for stats in run_stats:
                run_num = stats['run_number']
                attempted = stats['attempted_tasks']
                turn1_solved = stats['turn1_solved']
                all_turns_solved = stats['all_turns_solved']
                turn1_rate = stats['turn1_success_rate']
                all_turns_rate = stats['all_turns_success_rate']
                
                if self.independent_attempts:
                    print(f"{run_num:<4} {attempted:<10} {turn1_solved:<14} {all_turns_solved:<14} {turn1_rate:<14.1%} {all_turns_rate:<14.1%}")
                else:
                    print(f"{run_num:<4} {attempted:<10} {turn1_solved:<12} {all_turns_solved:<12} {turn1_rate:<12.1%} {all_turns_rate:<12.1%}")
            
            print("")
            print("AGGREGATE STATISTICS:")
            print("-" * 70)
            if self.independent_attempts:
                print(f"Attempt 1 Only Success Rate:")
                print(f"  Mean: {turn1_mean:.1%}")
                print(f"  Std Dev: {turn1_std:.1%}")
                print(f"  95% CI: [{turn1_mean - 1.96*turn1_std:.1%}, {turn1_mean + 1.96*turn1_std:.1%}]")
                print("")
                print(f"All Attempts Success Rate:")
                print(f"  Mean: {all_turns_mean:.1%}")
                print(f"  Std Dev: {all_turns_std:.1%}")
                print(f"  95% CI: [{all_turns_mean - 1.96*all_turns_std:.1%}, {all_turns_mean + 1.96*all_turns_std:.1%}]")
            else:
                print(f"Turn 1 Only Success Rate:")
                print(f"  Mean: {turn1_mean:.1%}")
                print(f"  Std Dev: {turn1_std:.1%}")
                print(f"  95% CI: [{turn1_mean - 1.96*turn1_std:.1%}, {turn1_mean + 1.96*turn1_std:.1%}]")
                print("")
                print(f"All Turns Success Rate:")
                print(f"  Mean: {all_turns_mean:.1%}")
                print(f"  Std Dev: {all_turns_std:.1%}")
                print(f"  95% CI: [{all_turns_mean - 1.96*all_turns_std:.1%}, {all_turns_mean + 1.96*all_turns_std:.1%}]")
            print("")
            print(f"Aggregate results saved to: {filepath}")
        else:
            print("\n❌ No valid run statistics to aggregate")

def main():
    parser = argparse.ArgumentParser(description="Run ARC tasks with OpenAI Chat Completions API")
    parser.add_argument("--dataset", default="arc-agi-1", choices=["arc-agi-1", "arc-agi-2"],
                       help="Dataset to use")
    parser.add_argument("--subset", default="shortest_1",
                       help="Subset name (e.g., shortest_1, shortest_10, shortest_100)")
    parser.add_argument("--model", default="gpt-4.1-mini",
                       help="OpenAI model to use")
    parser.add_argument("--limit", type=int,
                       help="Limit number of tasks to run")
    parser.add_argument("--max_tool_calls", type=int, default=64,
                       help="Maximum number of tool calls allowed for the model (default: 64, legacy parameter)")
    parser.add_argument("--base-url", type=str,
                       help="Base URL for OpenAI-compatible API endpoint (default: OpenAI)")
    parser.add_argument("--reasoning_effort", type=str, default="low", choices=["low", "medium", "high"],
                       help="Reasoning effort for models that support it (default: low)")
    parser.add_argument("--max_workers", type=int, default=1,
                       help="Maximum number of parallel workers (default: 1)")
    parser.add_argument("--rate_limit_delay", type=float, default=0.0,
                       help="Delay between API calls in seconds (default: 0.0)")
    parser.add_argument("--max_turns", type=int, default=3,
                       help="Maximum number of turns for multi-turn execution (default: 3)")
    parser.add_argument("--repeat-runs", type=int, default=1,
                       help="Number of times to repeat the entire test (default: 1)")
    parser.add_argument("--independent-attempts", action="store_true",
                       help="Use independent attempts mode instead of multi-turn feedback")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug output (default: disabled)")
    parser.add_argument("--qwen-no-think", action="store_true",
                       help="Disable thinking for Qwen models (sets enable_thinking=false and uses non-thinking sampling parameters)")
    
    args = parser.parse_args()
    
    # Validate max_workers
    if args.max_workers < 1:
        parser.error("--max_workers must be at least 1")
    if args.max_workers > 128:
        parser.error("--max_workers cannot exceed 128 (practical limit)")
    
    # Validate rate_limit_delay
    if args.rate_limit_delay < 0:
        parser.error("--rate_limit_delay cannot be negative")
    
    # Validate repeat_runs
    if args.repeat_runs < 1:
        parser.error("--repeat-runs must be at least 1")
    if args.repeat_runs > 10:
        parser.error("--repeat-runs cannot exceed 10 (practical limit)")
    
    # Create runner and run tasks
    runner = ARCTaskRunner(
        model=args.model, 
        max_workers=args.max_workers, 
        rate_limit_delay=args.rate_limit_delay, 
        max_turns=args.max_turns, 
        run_number=0,
        independent_attempts=args.independent_attempts,
        base_url=getattr(args, 'base_url', None),
        reasoning_effort=args.reasoning_effort,
        debug=args.debug,
        qwen_no_think=args.qwen_no_think
    )
    
    if args.repeat_runs > 1:
        # Run repeated tests with aggregate statistics
        runner.run_repeated_subset(args.subset, args.dataset, args.limit, args.repeat_runs)
    else:
        # Single run (original behavior)
        runner.run_subset(args.subset, args.dataset, args.limit)


if __name__ == "__main__":
    main()