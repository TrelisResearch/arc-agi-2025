#!/usr/bin/env python3

import os
import json
import requests
import datetime
from pathlib import Path
from dotenv import load_dotenv

# Load .env from o3-tools folder
load_dotenv("o3-tools/.env")

class ReasoningPersistenceTest:
    """Test whether o4-mini can provide reasoning summaries across multiple tasks"""
    
    def __init__(self):
        self.model = "o4-mini"
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        # Create test results directory
        self.results_dir = Path("tests/results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def call_responses_api(self, messages, use_tools=False):
        """Call the OpenAI Responses API"""
        data = {
            "model": self.model,
            "input": messages,
            "reasoning": {"effort": "medium"}
        }
        
        if use_tools:
            data["tools"] = [{"type": "code_interpreter", "container": {"type": "auto"}}]
            data["include"] = ["code_interpreter_call.outputs"]
            data["max_tool_calls"] = 10
        
        response = requests.post(
            'https://api.openai.com/v1/responses',
            headers=self.headers,
            json=data
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"API error {response.status_code}: {response.json()}")
    
    def extract_response_content(self, response_data):
        """Extract content from Responses API output"""
        content = []
        
        for output_item in response_data.get('output', []):
            if output_item.get('type') == 'message':
                content_items = output_item.get('content', [])
                for content_item in content_items:
                    if content_item.get('type') == 'output_text':
                        content.append({
                            'type': 'text',
                            'content': content_item.get('text', '')
                        })
            elif output_item.get('type') == 'code_interpreter_call':
                content.append({
                    'type': 'code_interpreter_call',
                    'content': output_item
                })
        
        return content
    
    def run_two_task_test(self, use_tools=False):
        """Run test with two reasoning tasks"""
        
        prompt = """I'm going to give you two reasoning tasks. Please solve them one by one, and provide a brief summary of your reasoning approach before solving each task.

Task 1: Pattern Recognition
Look at this sequence: 2, 6, 12, 20, 30, ?
What comes next and why? Before calculating, briefly explain your reasoning approach.

Task 2: Logic Puzzle  
Three friends - Alice, Bob, and Carol - each have a different pet (cat, dog, bird) and live in different colored houses (red, blue, green).
- Alice doesn't live in the red house
- The person with the cat lives in the blue house
- Bob doesn't have the bird
- Carol lives in the green house

Who has which pet? Before solving, briefly explain your reasoning approach.

Please solve these step by step, providing reasoning summaries as requested."""
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant skilled at reasoning and problem-solving."},
            {"role": "user", "content": prompt}
        ]
        
        print(f"\n{'='*60}")
        print(f"TESTING: o4-mini {'WITH' if use_tools else 'WITHOUT'} tools")
        print(f"{'='*60}")
        
        try:
            response_data = self.call_responses_api(messages, use_tools)
            content = self.extract_response_content(response_data)
            
            # Print the trace
            print("\nRESPONSE TRACE:")
            print("-" * 40)
            
            for i, item in enumerate(content):
                print(f"\n[{i+1}] Type: {item['type']}")
                if item['type'] == 'text':
                    print(f"Content: {item['content'][:500]}...")
                elif item['type'] == 'code_interpreter_call':
                    call_data = item['content']
                    print(f"Tool Call ID: {call_data.get('id', 'N/A')}")
                    if 'code' in call_data:
                        print(f"Code: {call_data['code'][:200]}...")
                    if 'outputs' in call_data:
                        print(f"Outputs: {str(call_data['outputs'])[:200]}...")
            
            # Save full response
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"reasoning_test_{timestamp}_{'with_tools' if use_tools else 'no_tools'}.json"
            filepath = self.results_dir / filename
            
            with open(filepath, 'w') as f:
                json.dump({
                    'timestamp': timestamp,
                    'model': self.model,
                    'use_tools': use_tools,
                    'prompt': prompt,
                    'raw_response': response_data,
                    'extracted_content': content
                }, f, indent=2)
            
            print(f"\nFull response saved to: {filepath}")
            
            return response_data, content
            
        except Exception as e:
            print(f"Error: {e}")
            return None, None
    
    def run_tests(self):
        """Run tests both with and without tools"""
        print("Testing o4-mini reasoning summaries...")
        
        # Test without tools
        print("\n" + "="*60)
        print("TEST 1: WITHOUT TOOLS")
        self.run_two_task_test(use_tools=False)
        
        # Test with tools  
        print("\n" + "="*60)
        print("TEST 2: WITH TOOLS")
        self.run_two_task_test(use_tools=True)
        
        print(f"\n{'='*60}")
        print("TESTS COMPLETE")
        print(f"Results saved in: {self.results_dir}")
        print(f"{'='*60}")

def main():
    tester = ReasoningPersistenceTest()
    tester.run_tests()

if __name__ == "__main__":
    main() 