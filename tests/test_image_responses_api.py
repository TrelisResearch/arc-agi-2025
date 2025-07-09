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

class ImageResponsesAPITest:
    """Test sending images with the OpenAI Responses API for o3/o4-mini models"""
    
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
    
    def create_simple_test_image(self):
        """Create a simple test image using PIL"""
        try:
            from PIL import Image, ImageDraw
            import numpy as np
            
            # Create a simple 100x100 test image with some colored squares
            img = Image.new('RGB', (100, 100), color='white')
            draw = ImageDraw.Draw(img)
            
            # Draw some colored rectangles
            draw.rectangle([10, 10, 40, 40], fill='red')
            draw.rectangle([60, 10, 90, 40], fill='blue')
            draw.rectangle([10, 60, 40, 90], fill='green')
            draw.rectangle([60, 60, 90, 90], fill='yellow')
            
            # Save the test image
            test_image_path = self.results_dir / "test_image.png"
            img.save(test_image_path)
            print(f"Created test image: {test_image_path}")
            return test_image_path
            
        except ImportError:
            print("PIL not available, skipping image creation")
            return None
    
    def test_image_with_responses_api(self, image_path, prompt="What do you see in this image?"):
        """Test sending an image with the responses API"""
        try:
            # Encode image to base64
            base64_image = self.encode_image_to_base64(image_path)
            
            print(f"\n--- TESTING IMAGE WITH RESPONSES API ---")
            print(f"Model: {self.model}")
            print(f"Image: {image_path}")
            print(f"Base64 length: {len(base64_image)} characters")
            print(f"Prompt: {prompt}")
            
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
            
            # Make the API call
            response = self.client.responses.create(
                model=self.model,
                input=input_messages,
                reasoning={"effort": "low"}
            )
            
            # Extract response text
            response_text = ""
            for output_item in response.output:
                if output_item.type == 'message':
                    for content_item in output_item.content:
                        if hasattr(content_item, 'text'):
                            response_text += content_item.text
            
            print(f"\n--- RESPONSE ---")
            print(f"Response: {response_text}")
            
            # Save complete result
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"image_test_{timestamp}_{self.model}.json"
            filepath = self.results_dir / filename
            
            result_data = {
                'timestamp': timestamp,
                'model': self.model,
                'image_path': str(image_path),
                'base64_length': len(base64_image),
                'prompt': prompt,
                'response_text': response_text,
                'full_response': {
                    'id': response.id,
                    'model': response.model,
                    'usage': {
                        'input_tokens': response.usage.input_tokens,
                        'output_tokens': response.usage.output_tokens,
                        'total_tokens': response.usage.total_tokens
                    }
                },
                'success': True
            }
            
            with open(filepath, 'w') as f:
                json.dump(result_data, f, indent=2)
            
            print(f"\nResult saved to: {filepath}")
            print(f"Tokens used: {response.usage.total_tokens} (input: {response.usage.input_tokens}, output: {response.usage.output_tokens})")
            
            return result_data
            
        except Exception as e:
            print(f"Error: {e}")
            
            # Save error result
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"image_test_error_{timestamp}_{self.model}.json"
            filepath = self.results_dir / filename
            
            error_data = {
                'timestamp': timestamp,
                'model': self.model,
                'image_path': str(image_path) if image_path else None,
                'prompt': prompt,
                'error': str(e),
                'success': False
            }
            
            with open(filepath, 'w') as f:
                json.dump(error_data, f, indent=2)
            
            return error_data
    
    def run_tests(self):
        """Run all image tests"""
        print("Testing image sending with OpenAI Responses API...")
        
        # Test 1: Create and test with a simple generated image
        print(f"\n{'='*60}")
        print("TEST 1: GENERATED TEST IMAGE")
        test_image = self.create_simple_test_image()
        
        if test_image:
            result1 = self.test_image_with_responses_api(
                test_image, 
                "Describe what you see in this image. What colors and shapes are present?"
            )
        else:
            print("Skipping generated image test - PIL not available")
            result1 = None
        
        # Test 2: Test with both o4-mini and o3-mini if available
        print(f"\n{'='*60}")
        print("TEST 2: MODEL COMPARISON")
        
        models_to_test = ["o4-mini", "o3-mini"]
        results = []
        
        for model in models_to_test:
            if test_image:
                print(f"\n--- Testing {model} ---")
                self.model = model
                try:
                    result = self.test_image_with_responses_api(
                        test_image,
                        "What geometric shapes and colors do you see? Be specific."
                    )
                    results.append(result)
                except Exception as e:
                    print(f"Failed to test {model}: {e}")
                    results.append({'model': model, 'error': str(e), 'success': False})
        
        print(f"\n{'='*60}")
        print("IMAGE TESTS COMPLETE")
        print(f"Results saved in: {self.results_dir}")
        print(f"{'='*60}")
        
        return results

def main():
    tester = ImageResponsesAPITest()
    tester.run_tests()

if __name__ == "__main__":
    main() 