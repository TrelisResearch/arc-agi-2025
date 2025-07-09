#!/usr/bin/env python3

import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import base64
import io

class ARCGridVisualizer:
    """Create visual representations of ARC grids for use with vision models"""
    
    def __init__(self, pixel_size=20, grid_spacing=2, section_spacing=30):
        self.pixel_size = pixel_size  # Size of each grid cell in pixels
        self.grid_spacing = grid_spacing  # Spacing between individual grids
        self.section_spacing = section_spacing  # Spacing between major sections
        
        # ARC color palette (0-9 mapped to distinct colors)
        self.colors = {
            0: (0, 0, 0),         # Black
            1: (0, 116, 217),     # Blue  
            2: (255, 65, 54),     # Red
            3: (46, 204, 64),     # Green
            4: (255, 220, 0),     # Yellow
            5: (170, 170, 170),   # Gray
            6: (240, 18, 190),    # Magenta
            7: (255, 133, 27),    # Orange
            8: (127, 219, 255),   # Sky Blue
            9: (135, 12, 37)      # Dark Red
        }
    
    def grid_to_image(self, grid):
        """Convert a single grid (2D array) to PIL Image"""
        height, width = len(grid), len(grid[0])
        img_width = width * self.pixel_size
        img_height = height * self.pixel_size
        
        img = Image.new('RGB', (img_width, img_height), 'white')
        draw = ImageDraw.Draw(img)
        
        for row in range(height):
            for col in range(width):
                value = grid[row][col]
                color = self.colors.get(value, (128, 128, 128))  # Default gray for unknown values
                
                x1 = col * self.pixel_size
                y1 = row * self.pixel_size
                x2 = x1 + self.pixel_size
                y2 = y1 + self.pixel_size
                
                draw.rectangle([x1, y1, x2, y2], fill=color)
                
                # Add thin border for better visibility
                draw.rectangle([x1, y1, x2, y2], outline=(64, 64, 64), width=1)
        
        return img
    
    def create_training_examples_image(self, task_data):
        """Create an image showing all training examples with input->output pairs"""
        train_examples = task_data['train']
        
        # Calculate dimensions
        example_images = []
        max_width = 0
        total_height = 0
        
        for i, example in enumerate(train_examples):
            input_grid = example['input']
            output_grid = example['output']
            
            input_img = self.grid_to_image(input_grid)
            output_img = self.grid_to_image(output_grid)
            
            # Create arrow image
            arrow_width = 40
            arrow_height = max(input_img.height, output_img.height)
            arrow_img = Image.new('RGB', (arrow_width, arrow_height), 'white')
            draw = ImageDraw.Draw(arrow_img)
            
            # Draw arrow
            mid_y = arrow_height // 2
            draw.line([(10, mid_y), (30, mid_y)], fill='black', width=3)
            draw.polygon([(25, mid_y-5), (30, mid_y), (25, mid_y+5)], fill='black')
            
            # Combine input, arrow, output horizontally
            example_width = input_img.width + arrow_width + output_img.width + 2 * self.grid_spacing
            example_height = max(input_img.height, output_img.height)
            
            example_img = Image.new('RGB', (example_width, example_height), 'white')
            
            x_offset = 0
            example_img.paste(input_img, (x_offset, 0))
            x_offset += input_img.width + self.grid_spacing
            example_img.paste(arrow_img, (x_offset, 0))
            x_offset += arrow_width + self.grid_spacing
            example_img.paste(output_img, (x_offset, 0))
            
            example_images.append(example_img)
            max_width = max(max_width, example_width)
            total_height += example_height + self.section_spacing
        
        # Remove last spacing
        total_height -= self.section_spacing
        
        # Create final image
        final_img = Image.new('RGB', (max_width, total_height), 'white')
        y_offset = 0
        
        for example_img in example_images:
            final_img.paste(example_img, (0, y_offset))
            y_offset += example_img.height + self.section_spacing
        
        return final_img
    
    def create_test_input_image(self, task_data):
        """Create an image showing just the test input"""
        test_input = task_data['test'][0]['input']
        return self.grid_to_image(test_input)
    
    def create_feedback_demo_image(self, task_data, predicted_outputs, training_results):
        """Create a demo of what feedback might look like"""
        train_examples = task_data['train']
        
        # Calculate dimensions for feedback layout
        feedback_images = []
        max_width = 0
        total_height = 0
        
        for i, (example, predicted, result) in enumerate(zip(train_examples, predicted_outputs, training_results)):
            expected_grid = example['output']
            predicted_grid = predicted
            
            expected_img = self.grid_to_image(expected_grid)
            predicted_img = self.grid_to_image(predicted_grid)
            
            # Create status indicator
            status_width = 60
            status_height = max(expected_img.height, predicted_img.height)
            status_img = Image.new('RGB', (status_width, status_height), 'white')
            draw = ImageDraw.Draw(status_img)
            
            # Draw status (✓ or ✗)
            if result['correct']:
                # Green checkmark
                draw.ellipse([10, 10, 50, 50], fill='green')
                draw.text((25, 25), "✓", fill='white', anchor="mm")
            else:
                # Red X
                draw.ellipse([10, 10, 50, 50], fill='red')
                draw.text((25, 25), "✗", fill='white', anchor="mm")
            
            # Create vs image
            vs_width = 40
            vs_img = Image.new('RGB', (vs_width, status_height), 'white')
            draw = ImageDraw.Draw(vs_img)
            draw.text((vs_width//2, status_height//2), "vs", fill='black', anchor="mm")
            
            # Combine expected, vs, predicted, status horizontally
            example_width = expected_img.width + vs_width + predicted_img.width + status_width + 3 * self.grid_spacing
            example_height = max(expected_img.height, predicted_img.height, status_height)
            
            example_img = Image.new('RGB', (example_width, example_height), 'white')
            
            x_offset = 0
            example_img.paste(expected_img, (x_offset, 0))
            x_offset += expected_img.width + self.grid_spacing
            example_img.paste(vs_img, (x_offset, 0))
            x_offset += vs_width + self.grid_spacing
            example_img.paste(predicted_img, (x_offset, 0))
            x_offset += predicted_img.width + self.grid_spacing
            example_img.paste(status_img, (x_offset, 0))
            
            feedback_images.append(example_img)
            max_width = max(max_width, example_width)
            total_height += example_height + self.section_spacing
        
        # Remove last spacing
        total_height -= self.section_spacing
        
        # Create final image
        final_img = Image.new('RGB', (max_width, total_height), 'white')
        y_offset = 0
        
        for feedback_img in feedback_images:
            final_img.paste(feedback_img, (0, y_offset))
            y_offset += feedback_img.height + self.section_spacing
        
        return final_img
    
    def image_to_base64(self, img):
        """Convert PIL Image to base64 string"""
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return img_str
    
    def save_demo_images(self, task_data, output_dir="tests/results"):
        """Create and save all demo images for a task"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create training examples image
        training_img = self.create_training_examples_image(task_data)
        training_path = output_path / "arc_demo_training.png"
        training_img.save(training_path)
        print(f"Saved training examples image: {training_path}")
        
        # Create test input image
        test_img = self.create_test_input_image(task_data)
        test_path = output_path / "arc_demo_test_input.png"
        test_img.save(test_path)
        print(f"Saved test input image: {test_path}")
        
        # Create mock feedback image
        # Generate some mock predicted outputs and results for demo
        mock_predicted = []
        mock_results = []
        
        for i, example in enumerate(task_data['train']):
            expected = example['output']
            
            # Create mock prediction (sometimes correct, sometimes not)
            if i % 2 == 0:  # Make every other prediction "correct"
                predicted = expected
                mock_results.append({'correct': True, 'pixel_accuracy': 1.0})
            else:  # Make some predictions wrong
                # Create a slightly wrong prediction
                predicted = [row[:] for row in expected]  # Deep copy
                if len(predicted) > 0 and len(predicted[0]) > 0:
                    predicted[0][0] = (predicted[0][0] + 1) % 10  # Change one pixel
                mock_results.append({'correct': False, 'pixel_accuracy': 0.95})
            
            mock_predicted.append(predicted)
        
        feedback_img = self.create_feedback_demo_image(task_data, mock_predicted, mock_results)
        feedback_path = output_path / "arc_demo_feedback.png"
        feedback_img.save(feedback_path)
        print(f"Saved feedback demo image: {feedback_path}")
        
        return {
            'training': training_path,
            'test_input': test_path, 
            'feedback': feedback_path
        }

def load_arc_task(task_file="data/arc-agi-1/training/007bbfb7.json"):
    """Load an ARC task for demo purposes"""
    with open(task_file, 'r') as f:
        return json.load(f)

def main():
    print("Creating ARC Grid Visualization Demo...")
    
    # Load a sample task
    task_data = load_arc_task()
    print(f"Loaded task with {len(task_data['train'])} training examples")
    
    # Create visualizer
    visualizer = ARCGridVisualizer(pixel_size=25, grid_spacing=5, section_spacing=40)
    
    # Generate demo images
    demo_paths = visualizer.save_demo_images(task_data)
    
    print("\n" + "="*60)
    print("DEMO IMAGES CREATED")
    print("="*60)
    
    for demo_type, path in demo_paths.items():
        img = Image.open(path)
        print(f"{demo_type.replace('_', ' ').title()}: {path}")
        print(f"  Size: {img.size[0]}x{img.size[1]} pixels")
        
        # Convert to base64 for API usage demo
        base64_str = visualizer.image_to_base64(img)
        print(f"  Base64 length: {len(base64_str)} characters")
        print()
    
    print("These images can now be sent to the responses API along with text descriptions!")
    
    # Print sample usage
    print("\nSample API Usage:")
    print("```python")
    print("input_messages = [")
    print("    {")
    print("        'role': 'user',")
    print("        'content': [")
    print("            {")
    print("                'type': 'input_text',")
    print("                'text': 'Analyze these training examples and find the pattern...'")
    print("            },")
    print("            {")
    print("                'type': 'input_image',")
    print("                'image_url': f'data:image/png;base64,{training_image_base64}'")
    print("            }")
    print("        ]")
    print("    }")
    print("]")
    print("```")

if __name__ == "__main__":
    main() 