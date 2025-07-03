#!/usr/bin/env python3

import os
import json
import argparse
import datetime
import requests
from pathlib import Path
from typing import Dict, List, Optional
from dotenv import load_dotenv

from task_loader import TaskLoader
from scoring import GridScorer, ProgramExecutor

load_dotenv()


class ARCTaskRunner:
    """Run ARC tasks using the OpenAI Responses API (single-shot with tool execution)"""
    
    def __init__(self, model: str = "gpt-4o-mini", use_tools: bool = False):
        self.model = model
        self.use_tools = use_tools
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.task_loader = TaskLoader()
        self.scorer = GridScorer()
        self.executor = ProgramExecutor(timeout=0.1)
        
        # Create logs directory
        self.logs_dir = Path("logs")
        self.logs_dir.mkdir(exist_ok=True)
        
        # Track costs
        self.total_cost = 0.0
        self.total_tokens = 0
        
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
    
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
    
    def create_prompt(self, task_data: Dict) -> str:
        """Create a prompt for the model to solve an ARC task"""
        task_str = self.task_loader.format_task_for_prompt(task_data)
        
        # Same prompt for both cases - just specify code block format clearly
        tools_instruction = """
You have access to a live Python code interpreter. Use it to explore the examples and develop your solution iteratively.""" if self.use_tools else ""
        
        prompt = f"""You are solving an ARC (Abstraction and Reasoning Corpus) task. 
I will show you training examples with input and output grids, and then give you a test input to solve.

{task_str}

Please analyze the pattern in the training examples and write a Python function that transforms the input grid to the output grid.{tools_instruction}

You can think through the problem and explain your reasoning, but you must end your response with:

Final answer:
```python
def transform(grid):
    # Your transformation logic here
    return transformed_grid
```

Requirements:
- The grid is a 2D list where grid[row][col] gives you the value at that position
- Values are integers from 0-9
- Return a new grid with the same structure
- Make sure your function handles the test input correctly
- Always include "Final answer:" followed by your ```python code block
"""
        
        return prompt
    
    def call_responses_api(self, messages: List[Dict]) -> Dict:
        """Call the OpenAI Responses API"""
        data = {
            "model": self.model,
            "input": messages
        }
        
        if self.use_tools:
            data["tools"] = [{"type": "code_interpreter", "container": {"type": "auto"}}]
            # Request detailed outputs for tool calls
            data["include"] = ["code_interpreter_call.outputs"]
        
        try:
            response = requests.post(
                'https://api.openai.com/v1/responses',
                headers=self.headers,
                json=data
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                error_data = response.json() if response.content else {}
                raise Exception(f"API error {response.status_code}: {error_data}")
                
        except Exception as e:
            raise Exception(f"API call failed: {str(e)}")
    
    
    def extract_code_from_response(self, response_data: Dict) -> str:
        """Extract Python code from the Responses API result using simple regex"""
        import re
        
        # Get the full text from response
        full_text = ""
        
        # Extract from Responses API structure
        for output_item in response_data.get('output', []):
            if output_item.get('type') == 'message':
                content_items = output_item.get('content', [])
                for content_item in content_items:
                    if content_item.get('type') == 'output_text':
                        full_text += content_item.get('text', '') + "\n"
        
        # Fallback: check legacy format if no text found
        if not full_text.strip():
            for choice in response_data.get('choices', []):
                message = choice.get('message', {})
                full_text += message.get('content', '') + "\n"
        
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
    
    def count_tool_calls(self, response_data: Dict) -> int:
        """Count the number of tool calls made in the response"""
        count = 0
        
        # For built-in tools in Responses API, count items in output array by type
        for output_item in response_data.get('output', []):
            output_type = output_item.get('type', '')
            
            # Built-in tools show up as their specific type in the output array
            if output_type == 'code_interpreter_call':
                count += 1
            elif output_type == 'web_search_preview_call':
                count += 1
            elif output_type == 'image_generation_call':
                count += 1
            elif output_type == 'file_search_call':
                count += 1
            # Add other built-in tool types as needed
            
            # For custom function tools, they appear in message.tool_calls
            elif output_type == 'message':
                content_items = output_item.get('content', [])
                for content_item in content_items:
                    if content_item.get('type') == 'tool_call':
                        count += 1
        
        return count
    
    def run_task(self, task_id: str, task_data: Dict) -> Dict:
        """Run a single ARC task using Responses API (single-shot for built-in tools)"""
        print(f"\nProcessing task: {task_id}")
        
        # Create initial prompt
        prompt = self.create_prompt(task_data)
        
        # Prepare initial messages
        messages = [
            {"role": "system", "content": "You are an expert at solving abstract reasoning puzzles. Write clean, efficient Python code."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            # Single API call - built-in tools execute internally at OpenAI
            print(f"  Making single Responses API call...")
            response_data = self.call_responses_api(messages)
            
            # Extract usage data from response
            usage = response_data.get('usage', {})
            total_tokens = usage.get('total_tokens', 0)
            self.total_tokens += total_tokens
            
            # Calculate cost based on model - with fallback calculation
            input_rate, output_rate = self.get_model_pricing(self.model)
            
            # Get token counts (Responses API uses input_tokens/output_tokens)
            input_tokens = usage.get('input_tokens', usage.get('prompt_tokens', 0))
            output_tokens = usage.get('output_tokens', usage.get('completion_tokens', 0))
            
            # Calculate request cost with proper input/output breakdown
            if input_tokens > 0 or output_tokens > 0:
                prompt_cost = (input_tokens / 1_000_000) * input_rate
                completion_cost = (output_tokens / 1_000_000) * output_rate
                request_cost = prompt_cost + completion_cost
                
                # Show reasoning token breakdown for reasoning models
                reasoning_tokens = usage.get('output_tokens_details', {}).get('reasoning_tokens', 0)
                if reasoning_tokens > 0:
                    visible_tokens = output_tokens - reasoning_tokens
                    print(f"  💰 Cost: ${request_cost:.6f} (input: {input_tokens} @ ${input_rate}, output: {output_tokens} @ ${output_rate})")
                    print(f"     ↳ Output breakdown: {reasoning_tokens} reasoning + {visible_tokens} visible tokens")
                else:
                    print(f"  💰 Cost: ${request_cost:.6f} (input: {input_tokens} @ ${input_rate}, output: {output_tokens} @ ${output_rate})")
            else:
                # Fallback: estimate 50/50 split if individual token counts missing
                if total_tokens > 0:
                    estimated_input = total_tokens // 2
                    estimated_output = total_tokens - estimated_input
                    prompt_cost = (estimated_input / 1_000_000) * input_rate
                    completion_cost = (estimated_output / 1_000_000) * output_rate
                    request_cost = prompt_cost + completion_cost
                    print(f"  💰 Cost: ${request_cost:.6f} (estimated from {total_tokens} total tokens)")
                else:
                    request_cost = 0.0
                    print(f"  ⚠️  No usage data available - cost calculation failed")
            
            self.total_cost += request_cost
            
            # Extract code and count tool calls
            program = self.extract_code_from_response(response_data)
            tool_calls_count = self.count_tool_calls(response_data)
            
            if not program:
                print(f"  No code found in response")
                # Count pixels even when no code is generated
                actual_output = task_data['test'][0]['output']
                total_pixels = len(actual_output) * len(actual_output[0]) if actual_output else 0
                
                return {
                    'task_id': task_id,
                    'model': self.model,
                    'use_tools': self.use_tools,
                    'api_type': 'responses_api',
                    'program': '',
                    'execution_error': 'No code generated',
                    'timed_out': False,
                    'tokens_used': usage.get('total_tokens', 0),
                    'tool_calls_count': tool_calls_count,
                    'request_cost': request_cost,
                    'raw_response': response_data,
                    'score': {
                        'correct': False, 
                        'pixel_accuracy': 0.0, 
                        'total_pixels': total_pixels,
                        'correct_pixels': 0,
                        'error': 'No code generated'
                    },
                    'mdl': None,
                    'actual_output': actual_output
                }
            
            # Execute program on test input
            test_input = task_data['test'][0]['input']
            predicted_output, error, timed_out = self.executor.execute_program(program, test_input)
            
            # Create base results
            results = {
                'task_id': task_id,
                'model': self.model,
                'use_tools': self.use_tools,
                'api_type': 'responses_api',
                'program': program,
                'execution_error': error,
                'timed_out': timed_out,
                'tokens_used': usage.get('total_tokens', 0),
                'tool_calls_count': tool_calls_count,
                'request_cost': request_cost,
                'raw_response': response_data
            }
            
            if predicted_output is not None:
                # Compare with actual output
                actual_output = task_data['test'][0]['output']
                score = self.scorer.score_grid(predicted_output, actual_output)
                results['score'] = score
                
                # Calculate MDL score for predicted program
                residual = self.scorer.calculate_residual_grid(predicted_output, actual_output)
                mdl = self.scorer.calculate_mdl_score(program, residual)
                results['mdl'] = mdl
                
                # Calculate null program MDL for comparison
                null_mdl = self.scorer.calculate_null_program_mdl(actual_output)
                results['null_mdl'] = null_mdl
                
                # Show MDL comparison
                mdl_improvement = null_mdl['null_mdl_score'] - mdl['mdl_score']
                if mdl_improvement > 0:
                    print(f"  📊 MDL: {mdl['mdl_score']:.1f} vs null {null_mdl['null_mdl_score']:.1f} (↓{mdl_improvement:.1f} better)")
                else:
                    print(f"  📊 MDL: {mdl['mdl_score']:.1f} vs null {null_mdl['null_mdl_score']:.1f} (↑{-mdl_improvement:.1f} worse)")
                
                results['predicted_output'] = predicted_output
                results['actual_output'] = actual_output
                
                if score.get('correct', False):
                    print(f"  ✅ Perfect solution found!")
                else:
                    print(f"  📊 Pixel accuracy: {score.get('pixel_accuracy', 0):.1%}")
            else:
                # Count pixels even when execution fails
                actual_output = task_data['test'][0]['output']
                total_pixels = len(actual_output) * len(actual_output[0]) if actual_output else 0
                results['score'] = {
                    'correct': False,
                    'pixel_accuracy': 0.0,
                    'total_pixels': total_pixels,
                    'correct_pixels': 0,
                    'error': 'Program execution failed'
                }
                results['mdl'] = None
                results['actual_output'] = actual_output
                
                # Still calculate null program MDL for comparison
                null_mdl = self.scorer.calculate_null_program_mdl(actual_output)
                results['null_mdl'] = null_mdl
                
                print(f"  ❌ Execution failed: {error}")
                print(f"  📊 Null program MDL baseline: {null_mdl['null_mdl_score']:.1f}")
            
            return results
            
        except Exception as e:
            # Count pixels even for exceptions
            try:
                actual_output = task_data['test'][0]['output']
                total_pixels = len(actual_output) * len(actual_output[0]) if actual_output else 0
            except:
                total_pixels = 0
                
            return {
                'task_id': task_id,
                'error': str(e),
                'score': {
                    'correct': False, 
                    'pixel_accuracy': 0.0,
                    'total_pixels': total_pixels,
                    'correct_pixels': 0
                }
            }
    
    def run_subset(self, subset_name: str, dataset: str = "arc-agi-1", limit: Optional[int] = None) -> List[Dict]:
        """Run all tasks in a subset"""
        tasks = self.task_loader.load_tasks_from_subset(subset_name, dataset)
        
        if limit:
            tasks = tasks[:limit]
        
        # Print configuration info
        print(f"\nRunning {len(tasks)} tasks from {dataset}/{subset_name}")
        print(f"Model: {self.model}")
        print(f"API: Responses API (single-shot)")
        if self.use_tools:
            print("Tools: ENABLED (code interpreter - OpenAI runs code internally, model can iterate)")
        else:
            print("Tools: DISABLED (model outputs final code, we execute it locally)")
        print("-" * 50)
        
        results = []
        for task_id, task_data in tasks:
            result = self.run_task(task_id, task_data)
            results.append(result)
            
            # Save individual result
            self.save_result(result)
        
        # Save summary
        self.save_summary(results, subset_name, dataset)
        
        return results
    
    def save_result(self, result: Dict):
        """Save individual task result"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{result['task_id']}.json"
        filepath = self.logs_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(result, f, indent=2)
    
    def save_summary(self, results: List[Dict], subset_name: str, dataset: str):
        """Save summary of all results"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Calculate statistics
        total_tasks = len(results)
        correct_tasks = sum(1 for r in results if r.get('score', {}).get('correct', False))
        total_pixels = sum(r.get('score', {}).get('total_pixels', 0) for r in results)
        correct_pixels = sum(r.get('score', {}).get('correct_pixels', 0) for r in results)
        
        # Calculate MDL statistics
        mdl_scores = [r.get('mdl', {}).get('mdl_score') for r in results if r.get('mdl')]
        avg_mdl = sum(mdl_scores) / len(mdl_scores) if mdl_scores else 0
        avg_program_tokens = sum(r.get('mdl', {}).get('program_tokens', 0) for r in results if r.get('mdl')) / total_tasks if total_tasks > 0 else 0
        avg_residual_bytes = sum(r.get('mdl', {}).get('residual_bytes', 0) for r in results if r.get('mdl')) / total_tasks if total_tasks > 0 else 0
        
        # Calculate null program MDL statistics
        null_mdl_scores = [r.get('null_mdl', {}).get('null_mdl_score') for r in results if r.get('null_mdl')]
        avg_null_mdl = sum(null_mdl_scores) / len(null_mdl_scores) if null_mdl_scores else 0
        
        # Calculate how many solutions beat the null baseline
        mdl_improvements = []
        for r in results:
            if r.get('mdl') and r.get('null_mdl'):
                improvement = r['null_mdl']['null_mdl_score'] - r['mdl']['mdl_score']
                mdl_improvements.append(improvement)
        
        tasks_beating_null = sum(1 for imp in mdl_improvements if imp > 0)
        avg_mdl_improvement = sum(mdl_improvements) / len(mdl_improvements) if mdl_improvements else 0
        
        # Calculate tool usage statistics
        total_tool_calls = sum(r.get('tool_calls_count', 0) for r in results)
        avg_tool_calls = total_tool_calls / total_tasks if total_tasks > 0 else 0
        
        summary = {
            'timestamp': timestamp,
            'dataset': dataset,
            'subset': subset_name,
            'model': self.model,
            'use_tools': self.use_tools,
            'api_type': 'responses_api',
            'total_tasks': total_tasks,
            'correct_tasks': correct_tasks,
            'task_accuracy': correct_tasks / total_tasks if total_tasks > 0 else 0.0,
            'total_pixels': total_pixels,
            'correct_pixels': correct_pixels,
            'pixel_accuracy': correct_pixels / total_pixels if total_pixels > 0 else 0.0,
            'avg_mdl_score': avg_mdl,
            'avg_program_tokens': avg_program_tokens,
            'avg_residual_bytes': avg_residual_bytes,
            'avg_null_mdl_score': avg_null_mdl,
            'tasks_beating_null': tasks_beating_null,
            'avg_mdl_improvement': avg_mdl_improvement,
            'total_tool_calls': total_tool_calls,
            'avg_tool_calls': avg_tool_calls,
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
        print(f"API: Responses (single-shot)")
        print(f"Tools enabled: {self.use_tools}")
        print(f"Tasks solved correctly: {correct_tasks}/{total_tasks} ({summary['task_accuracy']:.1%})")
        print(f"Pixel accuracy: {correct_pixels}/{total_pixels} ({summary['pixel_accuracy']:.1%})")
        print(f"Average MDL score: {avg_mdl:.1f}")
        print(f"Average program tokens: {avg_program_tokens:.1f}")
        print(f"Average residual bytes: {avg_residual_bytes:.1f}")
        print(f"Average null program MDL: {avg_null_mdl:.1f}")
        print(f"Solutions beating null baseline: {tasks_beating_null}/{len(mdl_improvements)} ({tasks_beating_null/len(mdl_improvements)*100:.1f}%)" if mdl_improvements else "Solutions beating null baseline: N/A")
        print(f"Average MDL improvement over null: {avg_mdl_improvement:.1f}")
        print(f"Total tool calls made: {total_tool_calls}")
        print(f"Average tool calls per task: {avg_tool_calls:.1f}")
        print(f"Total tokens used: {self.total_tokens:,}")
        print(f"Total cost: ${self.total_cost:.6f}")
        print(f"\nResults saved to: {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Run ARC tasks with OpenAI o3/o4 models")
    parser.add_argument("--dataset", default="arc-agi-1", choices=["arc-agi-1", "arc-agi-2"],
                       help="Dataset to use")
    parser.add_argument("--subset", default="shortest_1",
                       help="Subset name (e.g., shortest_1, shortest_10, shortest_100)")
    parser.add_argument("--model", default="gpt-4o-mini",
                       help="OpenAI model to use")
    parser.add_argument("--tools", action="store_true",
                       help="Enable code interpreter tools")
    parser.add_argument("--limit", type=int,
                       help="Limit number of tasks to run")
    
    args = parser.parse_args()
    
    # Create runner and run tasks
    runner = ARCTaskRunner(model=args.model, use_tools=args.tools)
    runner.run_subset(args.subset, args.dataset, args.limit)


if __name__ == "__main__":
    main()