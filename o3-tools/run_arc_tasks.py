#!/usr/bin/env python3

import os
import json
import argparse
import datetime
import requests
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

from task_loader import TaskLoader
from scoring import GridScorer, ProgramExecutor

load_dotenv()

class ARCTaskRunner:
    """Run ARC tasks using the OpenAI Responses API (single-shot with tool execution)"""
    
    def __init__(self, model: str = "gpt-4.1-nano", use_tools: bool = False, max_tool_calls: int = 64, reasoning_effort: str = "low", max_workers: int = 1, rate_limit_delay: float = 0.0, max_turns: int = 3):
        self.model = model
        self.use_tools = use_tools
        self.max_tool_calls = max_tool_calls
        self.reasoning_effort = reasoning_effort
        self.max_workers = max_workers
        self.rate_limit_delay = rate_limit_delay
        self.max_turns = max_turns
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.client = OpenAI()
        self.task_loader = TaskLoader()
        self.scorer = GridScorer()
        self.executor = ProgramExecutor(timeout=0.1)
        
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
    
    def is_reasoning_model(self, model: str = None) -> bool:
        """Check if the model supports reasoning effort parameter"""
        model_to_check = model or self.model
        model_lower = model_to_check.lower()
        return model_lower.startswith(('o3', 'o4', 'o1'))
    
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
        
        # Other models
        elif model_lower.startswith('codex-mini'):
            return (1.50, 6.00)
        elif model_lower.startswith('computer-use-preview'):
            return (3.00, 12.00)
        elif model_lower.startswith('gpt-image-1'):
            return (5.00, 0.00)  # No output pricing for image model
        
        # Default fallback (gpt-4o-mini pricing)
        else:
            return (0.15, 0.60)
    
    def create_prompt(self, task_data: Dict, is_first_turn: bool = True) -> str:
        """Create a prompt for the model to solve an ARC task"""
        if not is_first_turn:
            # For subsequent turns, just return a simple retry instruction
            return """Please analyze the training feedback and modify your approach. Write an improved transform function that handles the failing cases.

Final answer:
```python
def transform(grid):
    # Your improved transformation logic here
    return transformed_grid
```"""
        
        # Include test input in the prompt for context
        task_str = self.task_loader.format_task_for_prompt(task_data, include_test=True)
        
        prompt = f"""You are solving an ARC (Abstraction and Reasoning Corpus) task. 
I will show you training examples with input and output grids, plus a test input grid. Your task is to:

1. **Analyze the training examples** to discover the transformation pattern that maps each input grid to its corresponding output grid
2. **Write a Python program** that implements this transformation pattern  
3. **DO NOT predict or generate the test output** - your job is only to write the transformation program
4. **Ensure your program generalizes** - it will be applied to the test input grid (which may differ in size or complexity from training examples) after you provide it

The test input is shown for context so you understand what type of grid your program will eventually process. Focus on learning the pattern from training examples and writing robust code that can handle the test case.

{task_str}

Analyze the pattern in the training examples and write a Python function that performs this transformation.

Your function should work correctly on all the training examples shown above and be robust enough to handle the test input.

Final answer:
```python
def transform(grid):
    # Your transformation logic here
    return transformed_grid
```

Requirements:
- The function takes a 2D list (grid) where grid[row][col] gives the value at that position
- Values are integers from 0-9
- Return a new grid (2D list) with the transformation applied
- You can use numpy if needed - just add 'import numpy as np' at the start of your function
- Your function MUST produce the correct output for all training examples
- Your function must be robust enough to work on the test input grid
"""
        
        return prompt
    
    def call_responses_api(self, input_messages: List[Dict]) -> Dict:
        """Call the OpenAI Responses API for multi-turn local execution"""
        try:
            # Prepare the request
            kwargs = {
                "model": self.model,
                "input": input_messages
            }
            
            # Add reasoning effort and encrypted content only for reasoning models
            if self.is_reasoning_model():
                kwargs["reasoning"] = {"effort": self.reasoning_effort}
                
                # For multi-turn with encrypted reasoning traces (only supported by reasoning models)
                if self.use_tools:
                    kwargs["include"] = ["reasoning.encrypted_content"]
                    kwargs["store"] = False  # Enable stateless mode for encrypted content
            
            # Make the API call
            response = self.client.responses.create(**kwargs)
            
            return response
            
        except Exception as e:
            raise Exception(f"API call failed: {e}")
    
    def create_training_feedback(self, program: str, training_examples: List[Dict]) -> tuple[str, int, float]:
        """Generate detailed training feedback with stats and actual outputs for LLM"""
        results = []
        total_pixels = 0
        correct_pixels = 0
        solved_count = 0
        
        for i, example in enumerate(training_examples):
            predicted_output, error, timed_out = self.executor.execute_program(program, example['input'])
            expected_output = example['output']
            
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
        
        # Format feedback for LLM
        feedback = f"Training results: {solved_count}/{len(training_examples)} examples solved, {overall_accuracy:.1%} pixel accuracy\n\n"
        
        for result in results:
            status = "✓" if result['solved'] else "✗"
            feedback += f"Training Example {result['index']} {status}:\n"
            feedback += f"Expected: {result['expected']}\n"
            
            if result['error']:
                feedback += f"Error: {result['error']}\n"
            elif result['timed_out']:
                feedback += f"Error: Code execution timed out\n"
            else:
                feedback += f"Your output: {result['predicted']}\n"
                if not result['solved']:
                    feedback += f"Pixel accuracy: {result['pixel_accuracy']:.1%}\n"
            feedback += "\n"
        
        return feedback, solved_count, overall_accuracy
    
    def extract_code_from_response(self, response) -> str:
        """Extract Python code from the Responses API result using simple regex"""
        import re
        
        # Get the full text from response
        full_text = ""
        
        # Extract from Responses API structure (response object)
        for output_item in response.output:
            if output_item.type == 'message':
                for content_item in output_item.content:
                    if content_item.type == 'output_text':
                        full_text += content_item.text + "\n"
        
        # First priority: look for code after "Final answer:"
        final_answer_match = re.search(r'Final answer:\s*```python\s*\n(.*?)\n```', full_text, re.DOTALL | re.IGNORECASE)
        if final_answer_match:
            return final_answer_match.group(1).strip()
        
        # Second priority: last ```python block 
        python_blocks = re.findall(r'```python\s*\n(.*?)\n```', full_text, re.DOTALL)
        if python_blocks:
            return python_blocks[-1].strip()
        
        # Third priority: any ``` block with def transform
        code_blocks = re.findall(r'```\s*\n(.*?)\n```', full_text, re.DOTALL)
        for block in reversed(code_blocks):  # Check from last to first
            if 'def transform' in block:
                return block.strip()
        
        # Last resort: extract def transform function without code blocks
        transform_match = re.search(r'(def transform.*?)(?=\n\S|\n*$)', full_text, re.DOTALL)
        if transform_match:
            return transform_match.group(1).strip()
        
        return ""
    
    def create_success_result(self, task_id: str, program: str, response, test_score: Dict, total_cost: float, total_tokens: int, turns_used: int, task_data: Dict) -> Dict:
        """Create a successful task result"""

        # Convert response to JSON-serializable format
        response_dict = None
        if response:
            try:
                # Extract content safely from response objects
                output_items = []
                for item in response.output:
                    if item.type == 'message':
                        # Extract text content from message content items
                        content_texts = []
                        if hasattr(item, 'content'):
                            for content_item in item.content:
                                if hasattr(content_item, 'text'):
                                    content_texts.append(content_item.text)
                        output_items.append({'type': item.type, 'content': content_texts})
                    else:
                        output_items.append({'type': item.type, 'content': str(getattr(item, 'content', ''))})
                
                response_dict = {
                    'id': response.id,
                    'model': response.model,
                    'usage': {
                        'input_tokens': response.usage.input_tokens,
                        'output_tokens': response.usage.output_tokens,
                        'total_tokens': response.usage.total_tokens
                    },
                    'output': output_items
                }
            except Exception as e:
                response_dict = {'error': f'Failed to serialize response: {str(e)}'}

        return {
            'task_id': task_id,
            'model': self.model,
            'reasoning_effort': self.reasoning_effort if self.is_reasoning_model() else "N/A",
            'use_tools': self.use_tools,
            'api_type': 'responses_api_multiturn',
            'program': program,
            'execution_error': '',
            'timed_out': False,
            'tokens_used': total_tokens,
            'request_cost': total_cost,
            'turns_used': turns_used,
            'raw_response': response_dict,
            'score': test_score,
            'predicted_output': test_score.get('predicted_output'),
            'actual_output': test_score.get('actual_output'),
            'api_success': True
        }
    
    def create_failure_result(self, task_id: str, program: str, all_responses: List, total_cost: float, total_tokens: int, turns_used: int, task_data: Dict, error_msg: str) -> Dict:
        """Create a failed task result"""
        # Get test output for pixel counting
        actual_output = task_data['test'][0]['output']
        total_pixels = len(actual_output) * len(actual_output[0]) if actual_output else 0
        
        # Convert last response to JSON-serializable format
        response_dict = None
        if all_responses:
            response = all_responses[-1]
            try:
                # Extract content safely from response objects
                output_items = []
                for item in response.output:
                    if item.type == 'message':
                        # Extract text content from message content items
                        content_texts = []
                        if hasattr(item, 'content'):
                            for content_item in item.content:
                                if hasattr(content_item, 'text'):
                                    content_texts.append(content_item.text)
                        output_items.append({'type': item.type, 'content': content_texts})
                    else:
                        output_items.append({'type': item.type, 'content': str(getattr(item, 'content', ''))})
                
                response_dict = {
                    'id': response.id,
                    'model': response.model,
                    'usage': {
                        'input_tokens': response.usage.input_tokens,
                        'output_tokens': response.usage.output_tokens,
                        'total_tokens': response.usage.total_tokens
                    },
                    'output': output_items
                }
            except Exception as e:
                response_dict = {'error': f'Failed to serialize response: {str(e)}'}
        
        return {
            'task_id': task_id,
            'model': self.model,
            'reasoning_effort': self.reasoning_effort if self.is_reasoning_model() else "N/A",
            'use_tools': self.use_tools,
            'api_type': 'responses_api_multiturn',
            'program': program,
            'execution_error': error_msg,
            'timed_out': False,
            'tokens_used': total_tokens,
            'request_cost': total_cost,
            'turns_used': turns_used,
            'raw_response': response_dict,
            'score': {
                'correct': False,
                'pixel_accuracy': 0.0,
                'total_pixels': total_pixels,
                'correct_pixels': 0,
                'error': error_msg
            },
            'actual_output': actual_output,
            'api_success': True
        }
    

    
    def run_task(self, task_id: str, task_data: Dict, total_tasks: int = 1) -> Dict:
        """Run a single ARC task using multi-turn local code execution"""
        if self.max_workers == 1:  # Only print for sequential execution
            print(f"\nProcessing task: {task_id}")
        
        # Apply rate limiting if configured
        if self.rate_limit_delay > 0:
            time.sleep(self.rate_limit_delay)
        
        # Multi-turn conversation if tools enabled, otherwise single-shot
        if self.use_tools:
            return self.run_task_multiturn(task_id, task_data, total_tasks)
        else:
            return self.run_task_single_shot(task_id, task_data, total_tasks)
    
    def run_task_multiturn(self, task_id: str, task_data: Dict, total_tasks: int = 1) -> Dict:
        """Run multi-turn conversation with local code execution"""
        conversation_history = []
        total_cost = 0.0
        total_tokens = 0
        all_responses = []
        
        # Start conversation
        system_msg = {"role": "system", "content": "You are an expert at solving abstract reasoning puzzles. Write clean, efficient Python code."}
        initial_prompt = self.create_prompt(task_data, is_first_turn=True)
        conversation_history = [system_msg, {"role": "user", "content": initial_prompt}]
        
        try:
            for turn in range(self.max_turns):
                if self.max_workers == 1:  # Only print detailed logs for sequential execution
                    print(f"  🔄 Turn {turn + 1}/{self.max_turns}...")
                
                # Make API call
                response = self.call_responses_api(conversation_history)
                all_responses.append(response)
                
                # Track costs
                usage = response.usage
                input_rate, output_rate = self.get_model_pricing(self.model)
                input_tokens = usage.input_tokens
                output_tokens = usage.output_tokens
                turn_cost = (input_tokens / 1_000_000) * input_rate + (output_tokens / 1_000_000) * output_rate
                total_cost += turn_cost
                total_tokens += usage.total_tokens
                
                if self.max_workers == 1:
                    print(f"     💰 Turn cost: ${turn_cost:.6f} (input: {input_tokens}, output: {output_tokens})")
                
                # Extract code
                program = self.extract_code_from_response(response)
                
                if not program:
                    if self.max_workers == 1:
                        print(f"     ❌ No code found in response")
                    
                    # Stop if this is the last turn
                    if turn == self.max_turns - 1:
                        if self.max_workers == 1:
                            print(f"     ⏰ Max turns ({self.max_turns}) reached")
                        break
                    
                    # Continue conversation with request for code
                    # Extract reasoning item and assistant message for context preservation
                    reasoning_item = None
                    assistant_msg = None
                    for item in response.output:
                        if item.type == "reasoning":
                            reasoning_item = item
                        elif item.type == "message" and item.role == "assistant":
                            assistant_msg = item
                    
                    # Add to conversation history
                    if reasoning_item and assistant_msg:
                        conversation_history.extend([reasoning_item, assistant_msg])
                    elif assistant_msg:
                        conversation_history.append(assistant_msg)
                    
                    # Add request for code
                    code_request = """I need you to provide Python code to solve this task. Please provide a complete solution in this format:

```python
def transform(grid):
    # Your transformation logic here
    return transformed_grid
```

Make sure to include the function definition inside a proper code block."""
                    
                    conversation_history.append({"role": "user", "content": code_request})
                    if self.max_workers == 1:
                        print(f"     💬 Requesting code from model...")
                    continue
                
                # Test on test input first
                test_input = task_data['test'][0]['input']
                test_expected = task_data['test'][0]['output']
                predicted_output, error, timed_out = self.executor.execute_program(program, test_input)
                
                if predicted_output is not None and not error and not timed_out:
                    # Check if test is correct
                    test_score = self.scorer.score_grid(predicted_output, test_expected)
                    
                    if test_score['correct']:
                        # SUCCESS! Also get training stats for logging
                        training_feedback, solved_count, pixel_acc = self.create_training_feedback(program, task_data['train'])
                        if self.max_workers == 1:
                            print(f"     ✅ Perfect solution found! Training: {solved_count}/{len(task_data['train'])} solved, {pixel_acc:.1%} accuracy")
                        
                        # Update costs only (progress handled by parallel executor)
                        self._update_costs(total_cost, total_tokens)
                        
                        # Add actual outputs to test_score for logging
                        test_score['predicted_output'] = predicted_output
                        test_score['actual_output'] = test_expected
                        
                        return self.create_success_result(task_id, program, response, test_score, total_cost, total_tokens, turn + 1, task_data)
                
                # Not correct, get training feedback
                training_feedback, solved_count, pixel_acc = self.create_training_feedback(program, task_data['train'])
                if self.max_workers == 1:
                    print(f"     📊 Training: {solved_count}/{len(task_data['train'])} solved, {pixel_acc:.1%} accuracy")
                
                # Stop if this is the last turn
                if turn == self.max_turns - 1:
                    if self.max_workers == 1:
                        print(f"     ⏰ Max turns ({self.max_turns}) reached")
                    break
                
                # Continue conversation with feedback
                # Extract reasoning item and assistant message for context preservation
                reasoning_item = None
                assistant_msg = None
                for item in response.output:
                    if item.type == "reasoning":
                        reasoning_item = item
                    elif item.type == "message" and item.role == "assistant":
                        assistant_msg = item
                
                # Add to conversation history
                if reasoning_item and assistant_msg:
                    conversation_history.extend([reasoning_item, assistant_msg])
                elif assistant_msg:
                    conversation_history.append(assistant_msg)
                
                # Add training feedback
                feedback_prompt = self.create_prompt(task_data, is_first_turn=False) + "\n\n" + training_feedback
                conversation_history.append({"role": "user", "content": feedback_prompt})
            
            # Failed after max turns
            self._update_costs(total_cost, total_tokens)
            
            return self.create_failure_result(task_id, program if 'program' in locals() else "", all_responses, total_cost, total_tokens, self.max_turns, task_data, "Max turns reached")
        
        except Exception as e:
            print(f"     ❌ Multi-turn execution failed: {e}")
            self._update_costs(total_cost, total_tokens)
            
            return self.create_failure_result(task_id, "", all_responses, total_cost, total_tokens, 0, task_data, str(e))
    
    def run_task_single_shot(self, task_id: str, task_data: Dict, total_tasks: int = 1) -> Dict:
        """Run single-shot task without tools (legacy mode)"""
        print(f"  📝 Single-shot mode...")
        
        # Create initial prompt
        prompt = self.create_prompt(task_data, is_first_turn=True)
        
        # Prepare initial messages
        messages = [
            {"role": "system", "content": "You are an expert at solving abstract reasoning puzzles. Write clean, efficient Python code."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = self.call_responses_api(messages)
            
            # Extract usage data from response
            usage = response.usage
            total_tokens = usage.total_tokens
            
            # Calculate cost based on model
            input_rate, output_rate = self.get_model_pricing(self.model)
            
            # Get token counts
            input_tokens = usage.input_tokens
            output_tokens = usage.output_tokens
            
            # Calculate request cost with proper input/output breakdown
            prompt_cost = (input_tokens / 1_000_000) * input_rate
            completion_cost = (output_tokens / 1_000_000) * output_rate
            request_cost = prompt_cost + completion_cost
            
            # Show reasoning token breakdown for reasoning models
            reasoning_tokens = getattr(usage, 'output_tokens_details', {}).get('reasoning_tokens', 0) if hasattr(usage, 'output_tokens_details') else 0
            if self.max_workers == 1:
                if reasoning_tokens > 0:
                    visible_tokens = output_tokens - reasoning_tokens
                    print(f"  💰 Cost: ${request_cost:.6f} (input: {input_tokens}, output: {output_tokens})")
                    print(f"     ↳ Output breakdown: {reasoning_tokens} reasoning + {visible_tokens} visible tokens")
                else:
                    print(f"  💰 Cost: ${request_cost:.6f} (input: {input_tokens}, output: {output_tokens})")
            
            self._update_costs(request_cost, total_tokens)
            
            # Extract code
            program = self.extract_code_from_response(response)
            
            if not program:
                if self.max_workers == 1:
                    print(f"  ❌ No code found in response")
                # No code means 0 pixel accuracy (total pixels from expected output)
                actual_output = task_data['test'][0]['output']
                total_pixels = len(actual_output) * len(actual_output[0]) if actual_output else 0
                
                # No progress update needed (handled by parallel executor)
                
                return self.create_failure_result(task_id, '', [response], request_cost, total_tokens, 1, task_data, 'No code generated')
            
            # Execute program on test input
            test_input = task_data['test'][0]['input']
            predicted_output, error, timed_out = self.executor.execute_program(program, test_input)
            
            if predicted_output is not None and not timed_out and not error:
                # Score the result
                test_expected = task_data['test'][0]['output']
                test_score = self.scorer.score_grid(predicted_output, test_expected)
                
                # Also get training stats for logging
                training_feedback, solved_count, pixel_acc = self.create_training_feedback(program, task_data['train'])
                if self.max_workers == 1:
                    print(f"  📊 Training: {solved_count}/{len(task_data['train'])} solved, {pixel_acc:.1%} accuracy")
                
                # Progress tracking handled by parallel executor
                
                if test_score['correct']:
                    if self.max_workers == 1:
                        print(f"  ✅ Perfect solution found!")
                    # Add actual outputs to test_score for logging
                    test_score['predicted_output'] = predicted_output
                    test_score['actual_output'] = test_expected
                    return self.create_success_result(task_id, program, response, test_score, request_cost, total_tokens, 1, task_data)
                else:
                    if self.max_workers == 1:
                        print(f"  ❌ Incorrect solution")
                    return self.create_failure_result(task_id, program, [response], request_cost, total_tokens, 1, task_data, 'Incorrect solution')
            else:
                # Execution failed
                if self.max_workers == 1:
                    print(f"  ❌ Execution failed: {error or 'Timed out' if timed_out else 'Unknown error'}")
                # Progress tracking handled by parallel executor
                
                return self.create_failure_result(task_id, program, [response], request_cost, total_tokens, 1, task_data, error or 'Execution failed')

            
        except Exception as e:
            # Exception means 0 pixel accuracy (try to get total pixels from expected output)
            try:
                actual_output = task_data['test'][0]['output']
                total_pixels = len(actual_output) * len(actual_output[0]) if actual_output else 0
            except:
                total_pixels = 0
            
            # Print explicit error message for console visibility
            if self.max_workers == 1:
                print(f"  ❌ TASK FAILED: {task_id}")
                print(f"     Error: {str(e)}")
            
            # Progress tracking handled by parallel executor
                
            return {
                'task_id': task_id,
                'error': str(e),
                'score': {
                    'correct': False, 
                    'pixel_accuracy': 0.0,
                    'total_pixels': total_pixels,
                    'correct_pixels': 0
                },
                'api_success': False  # API call failed
            }
    
    def run_subset(self, subset_name: str, dataset: str = "arc-agi-1", limit: Optional[int] = None) -> List[Dict]:
        """Run all tasks in a subset with optional parallelization"""
        tasks = self.task_loader.load_tasks_from_subset(subset_name, dataset)
        
        if limit:
            tasks = tasks[:limit]
        
        total_tasks = len(tasks)
        
        # Print configuration info
        print(f"\nRunning {total_tasks} tasks from {dataset}/{subset_name}")
        print(f"Model: {self.model}")
        if self.use_tools:
            print(f"API: Responses API (multi-turn, max {self.max_turns} turns)")
            print("Tools: ENABLED (multi-turn local execution with training feedback)")
        else:
            print(f"API: Responses API (single-shot)")
            print("Tools: DISABLED (model outputs final code, we execute it locally)")
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
                if self.max_workers > 1:
                    print(f"[{task_id}] Starting...")
                result = self.run_task(task_id, task_data, total_tasks)
                # Save individual result
                self.save_result(result)
                if self.max_workers > 1:
                    status = "✅ SOLVED" if result.get('score', {}).get('correct', False) else "❌ FAILED"
                    cost = result.get('request_cost', 0.0)
                    print(f"[{task_id}] {status} (${cost:.6f})")
                return result
            
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
                    except Exception as e:
                        print(f"[{task_id}] ❌ SAVE FAILED: {e}")
                        # Create a minimal error result
                        error_result = {
                            'task_id': task_id,
                            'error': str(e),
                            'score': {'correct': False, 'pixel_accuracy': 0.0, 'total_pixels': 0, 'correct_pixels': 0}
                        }
                        results.append(error_result)
                        status = "❌ SAVE FAILED"
                    
                    # Show overall progress
                    progress_pct = (completed / total_tasks) * 100
                    print(f"Progress: {completed}/{total_tasks} tasks processed ({progress_pct:.1f}%) - {status}")
            
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
        
        # Separate successful API calls from complete failures
        api_successes = [r for r in results if r.get('api_success', True)]  # Default True for backward compatibility
        api_failures = [r for r in results if not r.get('api_success', True)]
        failed_tasks = len(api_failures)
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
            'reasoning_effort': self.reasoning_effort if self.is_reasoning_model() else "N/A",
            'use_tools': self.use_tools,
            'api_type': 'responses_api',
            'total_tasks': total_tasks,
            'successful_api_calls': successful_api_calls,
            'failed_api_calls': failed_tasks,
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
        
        filename = f"{timestamp}_summary_{dataset}_{subset_name}.json"
        filepath = self.logs_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        print("\n" + "="*50)
        print("SUMMARY")
        print("="*50)
        print(f"Dataset: {dataset}")
        print(f"Subset: {subset_name}")
        print(f"Model: {self.model}")
        # Only show reasoning effort for reasoning models
        if self.is_reasoning_model():
            print(f"Reasoning effort: {self.reasoning_effort}")
        if self.use_tools:
            print(f"API: Responses (multi-turn, max {self.max_turns} turns)")
        else:
            print(f"API: Responses (single-shot)")
        print(f"Multi-turn enabled: {self.use_tools}")
        print(f"Total tasks attempted: {total_tasks}")
        print(f"Successful API calls: {successful_api_calls}/{total_tasks} ({summary['success_rate']:.1%})")
        if failed_tasks > 0:
            print(f"Failed API calls: {failed_tasks}/{total_tasks} ({failed_tasks/total_tasks:.1%}) ❌")
        print(f"Tasks solved correctly: {correct_tasks}/{total_tasks} ({summary['task_accuracy']:.1%})")
        print(f"Pixel accuracy: {correct_pixels}/{total_pixels} ({summary['pixel_accuracy']:.1%})")
        print(f"Total turns used: {total_turns_used}")
        print(f"Average turns per task: {avg_turns_used:.1f}")
        print(f"Total tokens used: {self.total_tokens:,}")
        print(f"Total cost: ${self.total_cost:.6f}")
        
        # List failed tasks for debugging
        if failed_tasks > 0:
            print(f"\n❌ FAILED TASKS ({failed_tasks}):")
            for result in api_failures:
                task_id = result.get('task_id', 'unknown')
                error = result.get('error', 'Unknown error')
                print(f"  - {task_id}: {error}")
        
        print(f"\nResults saved to: {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Run ARC tasks with OpenAI o3/o4 models")
    parser.add_argument("--dataset", default="arc-agi-1", choices=["arc-agi-1", "arc-agi-2"],
                       help="Dataset to use")
    parser.add_argument("--subset", default="shortest_1",
                       help="Subset name (e.g., shortest_1, shortest_10, shortest_100)")
    parser.add_argument("--model", default="gpt-4.1-mini",
                       help="OpenAI model to use")
    parser.add_argument("--tools", action="store_true",
                       help="Enable multi-turn execution with local code testing and training feedback")
    parser.add_argument("--limit", type=int,
                       help="Limit number of tasks to run")
    parser.add_argument("--max_tool_calls", type=int, default=64,
                       help="Maximum number of tool calls allowed for the model (default: 64, legacy parameter)")
    parser.add_argument("--reasoning_effort", type=str, default="low", choices=["low", "medium", "high"],
                       help="Reasoning effort for the model (default: low)")
    parser.add_argument("--max_workers", type=int, default=1,
                       help="Maximum number of parallel workers (default: 1)")
    parser.add_argument("--rate_limit_delay", type=float, default=0.0,
                       help="Delay between API calls in seconds (default: 0.0)")
    parser.add_argument("--max_turns", type=int, default=3,
                       help="Maximum number of turns for multi-turn execution (default: 3)")
    
    args = parser.parse_args()
    
    # Validate max_workers
    if args.max_workers < 1:
        parser.error("--max_workers must be at least 1")
    if args.max_workers > 30:
        parser.error("--max_workers cannot exceed 30 (OpenAI rate limits)")
    
    # Validate rate_limit_delay
    if args.rate_limit_delay < 0:
        parser.error("--rate_limit_delay cannot be negative")
    
    # Create runner and run tasks
    runner = ARCTaskRunner(model=args.model, use_tools=args.tools, max_tool_calls=args.max_tool_calls, reasoning_effort=args.reasoning_effort, max_workers=args.max_workers, rate_limit_delay=args.rate_limit_delay, max_turns=args.max_turns)
    runner.run_subset(args.subset, args.dataset, args.limit)


if __name__ == "__main__":
    main()