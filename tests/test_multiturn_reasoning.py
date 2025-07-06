#!/usr/bin/env python3

import os
import json
import requests
import datetime
from pathlib import Path
from dotenv import load_dotenv

# Load .env from o3-tools folder
load_dotenv("o3-tools/.env")

class MultiTurnReasoningTest:
    """Test whether o4-mini reasoning persists across multiple turns in a conversation"""
    
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
            "reasoning": {"effort": "low"}
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
    
    def add_response_to_messages(self, messages, response_data):
        """Add the assistant's response to the message history for the next turn"""
        # Extract assistant response content
        assistant_content = ""
        
        for output_item in response_data.get('output', []):
            if output_item.get('type') == 'message':
                content_items = output_item.get('content', [])
                for content_item in content_items:
                    if content_item.get('type') == 'output_text':
                        assistant_content += content_item.get('text', '')
        
        if assistant_content:
            messages.append({
                "role": "assistant", 
                "content": assistant_content
            })
        
        return messages
    
    def run_multiturn_test(self, use_tools=False):
        """Run test with dependent reasoning across multiple turns"""
        
        print(f"\n{'='*60}")
        print(f"MULTITURN TEST: o4-mini {'WITH' if use_tools else 'WITHOUT'} tools")
        print(f"{'='*60}")
        
        # Turn 1: Establish a reasoning method
        turn1_prompt = """I have a pattern recognition challenge for you. Please solve this step by step and clearly explain your reasoning method:

Pattern: 1, 4, 9, 16, 25, ?

What comes next? More importantly, please describe the specific reasoning method you used to solve this, as I want to understand your approach."""
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant skilled at pattern recognition and reasoning."},
            {"role": "user", "content": turn1_prompt}
        ]
        
        try:
            print("\n--- TURN 1 ---")
            print("Prompt: Solve pattern 1, 4, 9, 16, 25, ? and explain reasoning method")
            
            response1 = self.call_responses_api(messages, use_tools)
            content1 = self.extract_response_content(response1)
            
            # Show turn 1 response
            for item in content1:
                if item['type'] == 'text':
                    print(f"Response: {item['content'][:300]}...")
                    break
            
            # Add response to message history
            messages = self.add_response_to_messages(messages, response1)
            
            # Turn 2: Ask to apply the same reasoning method to a new pattern
            turn2_prompt = """Perfect! Now please apply the exact same reasoning method you just described to solve this new pattern:

Pattern: 2, 8, 18, 32, 50, ?

Use the same approach you explained in your previous response. Can you reference your previous reasoning method and show how it applies here?"""
            
            messages.append({
                "role": "user",
                "content": turn2_prompt
            })
            
            print("\n--- TURN 2 ---")
            print("Prompt: Apply the same reasoning method to pattern 2, 8, 18, 32, 50, ?")
            
            response2 = self.call_responses_api(messages, use_tools)
            content2 = self.extract_response_content(response2)
            
            # Show turn 2 response
            for item in content2:
                if item['type'] == 'text':
                    print(f"Response: {item['content'][:300]}...")
                    break
            
            # Save complete conversation
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"multiturn_test_{timestamp}_{'with_tools' if use_tools else 'no_tools'}.json"
            filepath = self.results_dir / filename
            
            conversation_data = {
                'timestamp': timestamp,
                'model': self.model,
                'use_tools': use_tools,
                'turn1_prompt': turn1_prompt,
                'turn1_response': response1,
                'turn1_content': content1,
                'turn2_prompt': turn2_prompt,
                'turn2_response': response2,
                'turn2_content': content2,
                'full_messages': messages
            }
            
            with open(filepath, 'w') as f:
                json.dump(conversation_data, f, indent=2)
            
            print(f"\nFull conversation saved to: {filepath}")
            
            # Analyze if reasoning was referenced
            turn2_text = ""
            for item in content2:
                if item['type'] == 'text':
                    turn2_text = item['content']
                    break
            
            # Look for indicators that previous reasoning was referenced
            reference_indicators = [
                "same method", "previous", "earlier", "as I mentioned",
                "same approach", "like before", "similarly", "same reasoning"
            ]
            
            found_references = [indicator for indicator in reference_indicators 
                              if indicator.lower() in turn2_text.lower()]
            
            print(f"\n--- ANALYSIS ---")
            print(f"Turn 2 references to previous reasoning: {len(found_references)}")
            if found_references:
                print(f"Found references: {found_references}")
            else:
                print("No clear references to previous reasoning found")
            
            return conversation_data
            
        except Exception as e:
            print(f"Error: {e}")
            return None
    
    def run_tests(self):
        """Run multiturn tests both with and without tools"""
        print("Testing o4-mini multiturn reasoning persistence...")
        
        # Test without tools
        print("\n" + "="*60)
        print("TEST 1: WITHOUT TOOLS")
        result1 = self.run_multiturn_test(use_tools=False)
        
        # Test with tools  
        print("\n" + "="*60)
        print("TEST 2: WITH TOOLS")
        result2 = self.run_multiturn_test(use_tools=True)
        
        print(f"\n{'='*60}")
        print("MULTITURN TESTS COMPLETE")
        print(f"Results saved in: {self.results_dir}")
        print(f"{'='*60}")
        
        return result1, result2

def main():
    tester = MultiTurnReasoningTest()
    tester.run_tests()

if __name__ == "__main__":
    main() 