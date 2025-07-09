#!/usr/bin/env python3

import os
import json
import base64
import datetime
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# Load .env from o3-tools folder
load_dotenv(Path(__file__).parent.parent / "o3-tools" / ".env")

class ARCVisualAPITest:
    """Test ARC visual representations with the OpenAI Responses API"""
    
    def __init__(self):
        self.model = "o4-mini"
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.client = OpenAI(api_key=self.api_key)
        
        # Create test results directory
        self.results_dir = Path("tests/results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def encode_image_to_base64(self, image_path):
        """Encode an image file to base64 string"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    
    def test_training_examples_understanding(self):
        """Test if the model can understand training examples from visual representation"""
        image_path = self.results_dir / "arc_demo_training.png"
        base64_image = self.encode_image_to_base64(image_path)
        
        prompt = """I'm showing you visual representations of ARC (Abstraction and Reasoning Corpus) training examples.

Each row shows: INPUT → OUTPUT (input grid, arrow, output grid)

Please analyze these training examples and:
1. Describe what you see in each training example
2. Try to identify the transformation pattern that maps inputs to outputs
3. Explain the rule in your own words

Take your time to carefully examine each example."""
        
        print(f"\n--- TESTING TRAINING EXAMPLES UNDERSTANDING ---")
        print(f"Image: {image_path}")
        print(f"Base64 length: {len(base64_image)} characters")
        
        # Create the input messages with image
        input_messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text", 
                        "text": prompt
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{base64_image}"
                    }
                ]
            }
        ]
        
        try:
            # Make the API call
            response = self.client.responses.create(
                model=self.model,
                input=input_messages,
                reasoning={"effort": "medium"}
            )
            
            # Extract response text
            response_text = ""
            for output_item in response.output:
                if output_item.type == 'message':
                    for content_item in output_item.content:
                        if hasattr(content_item, 'text'):
                            response_text += content_item.text
            
            print(f"\n--- RESPONSE ---")
            print(response_text)
            
            # Save result
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            result_data = {
                'timestamp': timestamp,
                'test_type': 'training_examples_understanding',
                'model': self.model,
                'image_path': str(image_path),
                'prompt': prompt,
                'response_text': response_text,
                'tokens_used': response.usage.total_tokens,
                'success': True
            }
            
            filename = f"arc_visual_test_training_{timestamp}.json"
            filepath = self.results_dir / filename
            
            with open(filepath, 'w') as f:
                json.dump(result_data, f, indent=2)
            
            print(f"\nResult saved to: {filepath}")
            print(f"Tokens used: {response.usage.total_tokens}")
            
            return result_data
            
        except Exception as e:
            print(f"Error: {e}")
            return {'error': str(e), 'success': False}
    
    def test_feedback_understanding(self):
        """Test if the model can understand feedback visual representation"""
        image_path = self.results_dir / "arc_demo_feedback.png"
        base64_image = self.encode_image_to_base64(image_path)
        
        prompt = """I'm showing you a visual representation of training feedback for an ARC task.

Each row shows: EXPECTED vs PREDICTED (with a status indicator)
- Green circle with ✓ means the prediction was correct
- Red circle with ✗ means the prediction was wrong

Please analyze this feedback and:
1. How many training examples were predicted correctly vs incorrectly?
2. For the incorrect predictions, can you see what went wrong?
3. What would you suggest to improve the transformation rule?"""
        
        print(f"\n--- TESTING FEEDBACK UNDERSTANDING ---")
        print(f"Image: {image_path}")
        print(f"Base64 length: {len(base64_image)} characters")
        
        # Create the input messages with image
        input_messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text", 
                        "text": prompt
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{base64_image}"
                    }
                ]
            }
        ]
        
        try:
            # Make the API call
            response = self.client.responses.create(
                model=self.model,
                input=input_messages,
                reasoning={"effort": "medium"}
            )
            
            # Extract response text
            response_text = ""
            for output_item in response.output:
                if output_item.type == 'message':
                    for content_item in output_item.content:
                        if hasattr(content_item, 'text'):
                            response_text += content_item.text
            
            print(f"\n--- RESPONSE ---")
            print(response_text)
            
            # Save result
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            result_data = {
                'timestamp': timestamp,
                'test_type': 'feedback_understanding',
                'model': self.model,
                'image_path': str(image_path),
                'prompt': prompt,
                'response_text': response_text,
                'tokens_used': response.usage.total_tokens,
                'success': True
            }
            
            filename = f"arc_visual_test_feedback_{timestamp}.json"
            filepath = self.results_dir / filename
            
            with open(filepath, 'w') as f:
                json.dump(result_data, f, indent=2)
            
            print(f"\nResult saved to: {filepath}")
            print(f"Tokens used: {response.usage.total_tokens}")
            
            return result_data
            
        except Exception as e:
            print(f"Error: {e}")
            return {'error': str(e), 'success': False}
    
    def run_all_tests(self):
        """Run all visual API tests"""
        print("Testing ARC Visual Representations with OpenAI Responses API...")
        
        print(f"\n{'='*60}")
        print("TEST 1: TRAINING EXAMPLES UNDERSTANDING")
        result1 = self.test_training_examples_understanding()
        
        print(f"\n{'='*60}")
        print("TEST 2: FEEDBACK UNDERSTANDING")
        result2 = self.test_feedback_understanding()
        
        print(f"\n{'='*60}")
        print("ARC VISUAL API TESTS COMPLETE")
        print(f"Results saved in: {self.results_dir}")
        print(f"{'='*60}")
        
        return [result1, result2]

def main():
    tester = ARCVisualAPITest()
    tester.run_all_tests()

if __name__ == "__main__":
    main() 