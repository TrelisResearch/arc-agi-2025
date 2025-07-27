#!/usr/bin/env python3
"""
Test with a real ARC task using the exact same setup as the main script
"""

import os
import sys
import json
from openai import OpenAI
from dotenv import load_dotenv

# Add the current directory to the path so we can import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from .utils.task_loader import TaskLoader
from prompt_loader import PromptLoader

# Load environment variables from .env file
load_dotenv()

def test_real_arc_scenario():
    """Test with the exact same setup as the main script"""
    
    print("=== Real ARC Scenario Test ===")
    print("Simulating: --dataset arc-agi-1 --subset all_training --limit 1")
    print("Model: qwen/qwen3-coder:free")
    print("Parameters: --max_turns 1 --independent-attempts --reasoning_effort medium")
    print()
    
    # Get API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ No OPENAI_API_KEY environment variable found")
        return
    
    print(f"✅ API Key found: {api_key[:8]}...")
    
    # Initialize the same components as the main script
    task_loader = TaskLoader()
    prompt_loader = PromptLoader()
    
    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1"
    )
    
    # Load a real ARC task from the all_training subset
    print("\n=== Loading Real ARC Task ===")
    
    try:
        tasks = task_loader.load_tasks_from_subset("all_training", "arc-agi-1")
        if not tasks:
            print("❌ No tasks loaded!")
            return
            
        # Take the first task (limit 1)
        task_id, task_data = tasks[0]
        print(f"✅ Loaded task: {task_id}")
        print(f"Train examples: {len(task_data['train'])}")
        print(f"Test examples: {len(task_data['test'])}")
        
    except Exception as e:
        print(f"❌ Failed to load tasks: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    # Create the exact same prompt as the main script would
    print("\n=== Creating Prompt ===")
    
    try:
        # Get system message
        system_msg = {"role": "system", "content": prompt_loader.get_system_message("v1")}
        print(f"✅ System message length: {len(system_msg['content'])}")
        
        # Create the initial prompt using the main script's method
        # This is a simplified version of what create_prompt does
        def create_simple_prompt(task_data, task_id):
            # Format training examples
            train_examples = []
            for i, example in enumerate(task_data['train']):
                input_grid = example['input']
                output_grid = example['output']
                train_examples.append(f"Training Example {i+1}:")
                train_examples.append(f"Input: {input_grid}")
                train_examples.append(f"Output: {output_grid}")
                train_examples.append("")
            
            # Format test input
            test_input = task_data['test'][0]['input']
            
            prompt = f"""You are an AI assistant specialized in solving Abstract Reasoning Corpus (ARC-AGI) tasks by generating Python code.

Task: {task_id}

{chr(10).join(train_examples)}

Test Input: {test_input}

Write a Python function called `transform` that takes the input grid and produces the correct output. The function should work for all the training examples and the test case.

```python
def transform(grid):
    # Your implementation here
    pass
```"""
            
            return prompt
        
        initial_prompt = create_simple_prompt(task_data, task_id)
        print(f"✅ Initial prompt length: {len(initial_prompt)}")
        print(f"First 500 chars of prompt: {initial_prompt[:500]}...")
        
    except Exception as e:
        print(f"❌ Failed to create prompt: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    # Make the API call with exact same parameters
    print("\n=== Making API Call ===")
    
    messages = [
        system_msg,
        {"role": "user", "content": initial_prompt}
    ]
    
    try:
        # Exact same parameters as the main script with reasoning_effort medium
        kwargs = {
            "model": "qwen/qwen3-coder:free",
            "messages": messages,
            "max_tokens": 8000,  # medium reasoning effort
        }
        
        print(f"Request parameters: {json.dumps({k: v if k != 'messages' else f'[{len(v)} messages]' for k, v in kwargs.items()}, indent=2)}")
        
        response = client.chat.completions.create(**kwargs)
        
        print("✅ API call successful!")
        print(f"Choice finish_reason: {getattr(response.choices[0], 'finish_reason', 'N/A')}")
        
        if response.choices and response.choices[0].message.content:
            content = response.choices[0].message.content
            print(f"Message content length: {len(content)}")
            
            # Show first and last parts of response
            if len(content) > 1000:
                print(f"\n=== First 500 chars of Response ===")
                print(content[:500])
                print(f"\n=== Last 500 chars of Response ===")
                print(content[-500:])
            else:
                print(f"\n=== Full Response Content ===")
                print(content)
            
            # Test code extraction like the main script does
            print(f"\n=== Testing Code Extraction ===")
            import re
            code_pattern = r'```(?:python)?\s*\n(.*?)```'
            matches = re.findall(code_pattern, content, re.DOTALL)
            
            if matches:
                print(f"Found {len(matches)} code blocks:")
                for i, match in enumerate(matches):
                    code = match.strip()
                    print(f"Code block {i+1} (length: {len(code)}):")
                    print(code[:200] + "..." if len(code) > 200 else code)
                    print()
                    
                    # Check if transform function exists
                    if 'def transform(' in code:
                        print("✅ Found 'def transform()' function in this block")
                    
                        # Try to execute the code to see if it's valid
                        try:
                            exec(code)
                            print("✅ Code compiles successfully")
                        except Exception as e:
                            print(f"❌ Code compilation failed: {type(e).__name__}: {str(e)}")
                        
            else:
                print("❌ No code blocks found in response!")
                print("This could be why you're getting 'no meaningful response'")
                
        else:
            print("❌ Message content is None/empty!")
            print("This explains the 'no meaningful response' issue!")
            
        # Check usage info
        if hasattr(response, 'usage'):
            print(f"\n=== Usage Information ===")
            print(f"Prompt tokens: {response.usage.prompt_tokens}")
            print(f"Completion tokens: {response.usage.completion_tokens}")
            print(f"Total tokens: {response.usage.total_tokens}")
        
    except Exception as e:
        print(f"❌ API call failed: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_real_arc_scenario() 