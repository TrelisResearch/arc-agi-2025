#!/usr/bin/env python3

import os
import json
import requests
import datetime
from pathlib import Path
from dotenv import load_dotenv

# Load .env from o3-tools folder
load_dotenv("o3-tools/.env")

class HiddenReasoningPersistenceTest:
    """Test whether o4-mini's hidden reasoning tokens persist across turns"""
    
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
    
    def run_hidden_reasoning_test(self, use_tools=False):
        """Test with reasoning that stays hidden, then reference it in turn 2"""
        
        print(f"\n{'='*60}")
        print(f"HIDDEN REASONING TEST: o4-mini {'WITH' if use_tools else 'WITHOUT'} tools")
        print(f"{'='*60}")
        
        # Turn 1: Give it a task that requires reasoning but don't ask to explain
        turn1_prompt = """What's the next number in this sequence: 7, 26, 63, 124, 215, ?

Just give me the answer."""
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Solve problems accurately."},
            {"role": "user", "content": turn1_prompt}
        ]
        
        try:
            print("\n--- TURN 1 ---")
            print("Prompt: What's next in 7, 26, 63, 124, 215, ? (Just answer)")
            
            response1 = self.call_responses_api(messages, use_tools)
            content1 = self.extract_response_content(response1)
            
            # Show turn 1 response (should be brief since we didn't ask for explanation)
            for item in content1:
                if item['type'] == 'text':
                    print(f"Response: {item['content']}")
                    break
            
            # Add response to message history
            messages = self.add_response_to_messages(messages, response1)
            
            # Turn 2: Ask for something that would benefit from the same pattern insight
            turn2_prompt = """Now, can you tell me what 342 would be in that same sequence?
            
In other words, if the pattern continued, what position would 342 hold?"""
            
            messages.append({
                "role": "user",
                "content": turn2_prompt
            })
            
            print("\n--- TURN 2 ---")
            print("Prompt: What position would 342 hold in that sequence?")
            
            response2 = self.call_responses_api(messages, use_tools)
            content2 = self.extract_response_content(response2)
            
            # Show turn 2 response
            for item in content2:
                if item['type'] == 'text':
                    print(f"Response: {item['content']}")
                    break
            
            # Save complete conversation
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"hidden_reasoning_test_{timestamp}_{'with_tools' if use_tools else 'no_tools'}.json"
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
                'full_messages': messages,
                'reasoning_tokens_turn1': response1.get('usage', {}).get('output_tokens_details', {}).get('reasoning_tokens', 0),
                'reasoning_tokens_turn2': response2.get('usage', {}).get('output_tokens_details', {}).get('reasoning_tokens', 0)
            }
            
            with open(filepath, 'w') as f:
                json.dump(conversation_data, f, indent=2)
            
            print(f"\nFull conversation saved to: {filepath}")
            
            # Analyze reasoning token usage
            reasoning1 = response1.get('usage', {}).get('output_tokens_details', {}).get('reasoning_tokens', 0)
            reasoning2 = response2.get('usage', {}).get('output_tokens_details', {}).get('reasoning_tokens', 0)
            
            print(f"\n--- REASONING TOKEN ANALYSIS ---")
            print(f"Turn 1 reasoning tokens: {reasoning1}")
            print(f"Turn 2 reasoning tokens: {reasoning2}")
            
            # Check if Turn 2 shows evidence of using the pattern from Turn 1
            turn2_text = ""
            for item in content2:
                if item['type'] == 'text':
                    turn2_text = item['content']
                    break
            
            # The correct answer for position of 342 would be 7 (since the pattern is n³ + n² + n + 1)
            # If the model gets this right, it suggests it figured out the pattern in Turn 1
            # and applied it in Turn 2 (whether from visible text or hidden reasoning)
            
            pattern_indicators = ["342", "position", "7th", "n=7", "seventh"]
            found_indicators = [indicator for indicator in pattern_indicators 
                              if indicator.lower() in turn2_text.lower()]
            
            print(f"Turn 2 pattern application indicators: {found_indicators}")
            
            return conversation_data
            
        except Exception as e:
            print(f"Error: {e}")
            return None
    
    def run_tests(self):
        """Run hidden reasoning tests both with and without tools"""
        print("Testing o4-mini hidden reasoning persistence...")
        print("(The pattern: 7, 26, 63, 124, 215 follows n³ + n² + n + 1 for n=1,2,3,4,5)")
        print("Next would be 342 (n=6), and 342 itself would be at position 7")
        
        # Test without tools
        print("\n" + "="*60)
        print("TEST 1: WITHOUT TOOLS")
        result1 = self.run_hidden_reasoning_test(use_tools=False)
        
        # Test with tools  
        print("\n" + "="*60)
        print("TEST 2: WITH TOOLS")
        result2 = self.run_hidden_reasoning_test(use_tools=True)
        
        print(f"\n{'='*60}")
        print("HIDDEN REASONING TESTS COMPLETE")
        print(f"Results saved in: {self.results_dir}")
        print(f"{'='*60}")
        
        return result1, result2

def main():
    tester = HiddenReasoningPersistenceTest()
    tester.run_tests()

if __name__ == "__main__":
    main() 