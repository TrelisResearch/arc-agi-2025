#!/usr/bin/env python3

import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import base64
import io

class ARCVisualDemo:
    """Demo visual representations for ARC task integration"""
    
    def __init__(self, pixel_size=20, grid_spacing=5, section_spacing=30):
        self.pixel_size = pixel_size
        self.grid_spacing = grid_spacing
        self.section_spacing = section_spacing
        
        # ARC color palette
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
        """Convert a single grid to PIL Image"""
        if not grid or not grid[0]:
            return Image.new('RGB', (20, 20), 'white')
            
        height, width = len(grid), len(grid[0])
        img_width = width * self.pixel_size
        img_height = height * self.pixel_size
        
        img = Image.new('RGB', (img_width, img_height), 'white')
        draw = ImageDraw.Draw(img)
        
        for row in range(height):
            for col in range(width):
                value = grid[row][col]
                color = self.colors.get(value, (128, 128, 128))
                
                x1 = col * self.pixel_size
                y1 = row * self.pixel_size
                x2 = x1 + self.pixel_size
                y2 = y1 + self.pixel_size
                
                draw.rectangle([x1, y1, x2, y2], fill=color)
                draw.rectangle([x1, y1, x2, y2], outline=(80, 80, 80), width=1)
        
        return img
    
    def add_label(self, img, text, position='top'):
        """Add a text label to an image"""
        # Create new image with space for label
        label_height = 25
        if position == 'top':
            new_img = Image.new('RGB', (img.width, img.height + label_height), 'white')
            new_img.paste(img, (0, label_height))
            text_y = 12
        else:  # bottom
            new_img = Image.new('RGB', (img.width, img.height + label_height), 'white')
            new_img.paste(img, (0, 0))
            text_y = img.height + 12
        
        draw = ImageDraw.Draw(new_img)
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 14)
        except:
            font = ImageFont.load_default()
        
        # Center the text
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_x = (new_img.width - text_width) // 2
        
        draw.text((text_x, text_y), text, fill='black', font=font)
        return new_img
    
    def create_turn1_input_demo(self, task_data):
        """Create demo of what Turn 1 input would look like"""
        
        # Training examples section
        train_examples = task_data['train']
        example_pairs = []
        
        for i, example in enumerate(train_examples):
            input_grid = example['input']
            output_grid = example['output']
            
            input_img = self.grid_to_image(input_grid)
            output_img = self.grid_to_image(output_grid)
            
            # Add labels
            input_img = self.add_label(input_img, f"Input {i+1}")
            output_img = self.add_label(output_img, f"Output {i+1}")
            
            # Create arrow
            arrow_width = 50
            arrow_height = max(input_img.height, output_img.height)
            arrow_img = Image.new('RGB', (arrow_width, arrow_height), 'white')
            draw = ImageDraw.Draw(arrow_img)
            
            mid_y = arrow_height // 2
            draw.line([(15, mid_y), (35, mid_y)], fill='black', width=3)
            draw.polygon([(30, mid_y-6), (35, mid_y), (30, mid_y+6)], fill='black')
            
            # Combine into pair
            pair_width = input_img.width + arrow_width + output_img.width + 2 * self.grid_spacing
            pair_height = max(input_img.height, output_img.height)
            pair_img = Image.new('RGB', (pair_width, pair_height), 'white')
            
            x_offset = 0
            pair_img.paste(input_img, (x_offset, 0))
            x_offset += input_img.width + self.grid_spacing
            pair_img.paste(arrow_img, (x_offset, 0))
            x_offset += arrow_width + self.grid_spacing  
            pair_img.paste(output_img, (x_offset, 0))
            
            example_pairs.append(pair_img)
        
        # Test input section
        test_input = task_data['test'][0]['input']
        test_img = self.grid_to_image(test_input)
        test_img = self.add_label(test_img, "Test Input")
        
        # Layout everything vertically with section headers
        max_width = max([pair.width for pair in example_pairs] + [test_img.width])
        
        # Calculate total height
        total_height = 0
        # Training section header
        total_height += 40
        # Training examples
        for pair in example_pairs:
            total_height += pair.height + self.section_spacing
        # Test section header  
        total_height += 40
        # Test input
        total_height += test_img.height
        
        # Create final image
        final_img = Image.new('RGB', (max_width + 40, total_height), 'white')
        draw = ImageDraw.Draw(final_img)
        
        try:
            title_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 18)
        except:
            title_font = ImageFont.load_default()
        
        y_offset = 10
        
        # Training section
        draw.text((20, y_offset), "TRAINING EXAMPLES:", fill='black', font=title_font)
        y_offset += 40
        
        for pair in example_pairs:
            x_center = (final_img.width - pair.width) // 2
            final_img.paste(pair, (x_center, y_offset))
            y_offset += pair.height + self.section_spacing
        
        # Test section
        draw.text((20, y_offset), "TEST INPUT:", fill='black', font=title_font)
        y_offset += 40
        
        x_center = (final_img.width - test_img.width) // 2
        final_img.paste(test_img, (x_center, y_offset))
        
        return final_img
    
    def create_feedback_demo(self, task_data):
        """Create demo of what feedback would look like"""
        train_examples = task_data['train']
        
        # Generate mock predictions with some errors
        mock_predictions = []
        mock_results = []
        
        for i, example in enumerate(train_examples):
            expected = example['output']
            
            if i % 3 == 0:  # Every 3rd prediction is correct
                predicted = expected
                correct = True
                pixel_acc = 1.0
            else:  # Others have errors
                # Create a slightly wrong prediction
                predicted = [row[:] for row in expected]  # Deep copy
                if len(predicted) > 1 and len(predicted[0]) > 1:
                    # Change a few pixels randomly
                    import random
                    for _ in range(random.randint(1, 3)):
                        r = random.randint(0, len(predicted)-1)
                        c = random.randint(0, len(predicted[0])-1)
                        predicted[r][c] = (predicted[r][c] + random.randint(1, 9)) % 10
                
                correct = False
                pixel_acc = random.uniform(0.7, 0.95)
            
            mock_predictions.append(predicted)
            mock_results.append({
                'correct': correct,
                'pixel_accuracy': pixel_acc,
                'solved_count': 1 if correct else 0
            })
        
        # Create feedback visualization
        feedback_rows = []
        
        for i, (example, predicted, result) in enumerate(zip(train_examples, mock_predictions, mock_results)):
            expected_grid = example['output']
            predicted_grid = predicted
            
            expected_img = self.grid_to_image(expected_grid)
            predicted_img = self.grid_to_image(predicted_grid)
            
            # Add labels
            expected_img = self.add_label(expected_img, f"Expected {i+1}")
            predicted_img = self.add_label(predicted_img, f"Predicted {i+1}")
            
            # Create status indicator
            status_size = 50
            status_img = Image.new('RGB', (status_size, status_size), 'white')
            draw = ImageDraw.Draw(status_img)
            
            if result['correct']:
                # Green circle with checkmark
                draw.ellipse([5, 5, 45, 45], fill='green')
                draw.text((25, 25), "✓", fill='white', anchor="mm")
            else:
                # Red circle with X
                draw.ellipse([5, 5, 45, 45], fill='red')
                draw.text((25, 25), "✗", fill='white', anchor="mm")
            
            # Add accuracy text below status
            status_with_text = Image.new('RGB', (status_size, status_size + 30), 'white')
            status_with_text.paste(status_img, (0, 0))
            draw_text = ImageDraw.Draw(status_with_text)
            acc_text = f"{result['pixel_accuracy']:.1%}"
            draw_text.text((status_size//2, status_size + 15), acc_text, fill='black', anchor="mm")
            
            # Create "vs" separator
            vs_width = 30
            vs_height = max(expected_img.height, predicted_img.height)
            vs_img = Image.new('RGB', (vs_width, vs_height), 'white')
            draw_vs = ImageDraw.Draw(vs_img)
            draw_vs.text((vs_width//2, vs_height//2), "vs", fill='black', anchor="mm")
            
            # Combine into row
            row_width = expected_img.width + vs_width + predicted_img.width + status_with_text.width + 3 * self.grid_spacing
            row_height = max(expected_img.height, predicted_img.height, status_with_text.height)
            row_img = Image.new('RGB', (row_width, row_height), 'white')
            
            x_offset = 0
            row_img.paste(expected_img, (x_offset, 0))
            x_offset += expected_img.width + self.grid_spacing
            row_img.paste(vs_img, (x_offset, (row_height - vs_height) // 2))
            x_offset += vs_width + self.grid_spacing
            row_img.paste(predicted_img, (x_offset, 0))
            x_offset += predicted_img.width + self.grid_spacing
            row_img.paste(status_with_text, (x_offset, (row_height - status_with_text.height) // 2))
            
            feedback_rows.append(row_img)
        
        # Combine all rows with title
        max_width = max(row.width for row in feedback_rows)
        total_height = 40  # Title space
        for row in feedback_rows:
            total_height += row.height + self.section_spacing
        
        final_img = Image.new('RGB', (max_width + 40, total_height), 'white')
        draw = ImageDraw.Draw(final_img)
        
        try:
            title_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 18)
        except:
            title_font = ImageFont.load_default()
        
        # Title
        y_offset = 10
        draw.text((20, y_offset), "TRAINING FEEDBACK:", fill='black', font=title_font)
        y_offset += 40
        
        # Feedback rows
        for row in feedback_rows:
            x_center = (final_img.width - row.width) // 2
            final_img.paste(row, (x_center, y_offset))
            y_offset += row.height + self.section_spacing
        
        return final_img
    
    def save_demo_images(self, task_data, output_dir="tests/results"):
        """Save demo images for review"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create Turn 1 input demo
        turn1_img = self.create_turn1_input_demo(task_data)
        turn1_path = output_path / "demo_turn1_input.png"
        turn1_img.save(turn1_path)
        
        # Create feedback demo
        feedback_img = self.create_feedback_demo(task_data)
        feedback_path = output_path / "demo_feedback.png"
        feedback_img.save(feedback_path)
        
        return {
            'turn1_input': turn1_path,
            'feedback': feedback_path
        }

def main():
    print("Creating ARC Visual Integration Demo...")
    
    # Load sample task
    task_file = "data/arc-agi-1/training/007bbfb7.json"
    with open(task_file, 'r') as f:
        task_data = json.load(f)
    
    print(f"Loaded task: {task_file}")
    print(f"Training examples: {len(task_data['train'])}")
    
    # Create demo
    demo = ARCVisualDemo(pixel_size=25, grid_spacing=8, section_spacing=30)
    demo_paths = demo.save_demo_images(task_data)
    
    print("\n" + "="*60)
    print("INTEGRATION DEMO IMAGES CREATED")
    print("="*60)
    
    for demo_type, path in demo_paths.items():
        img = Image.open(path)
        print(f"{demo_type.replace('_', ' ').title()}: {path}")
        print(f"  Size: {img.size[0]}x{img.size[1]} pixels")
        
        # Estimate base64 size
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        base64_size = len(base64.b64encode(buffer.getvalue()).decode())
        print(f"  Base64 size: {base64_size} characters")
        print()
    
    print("WHAT THIS SHOWS:")
    print("================")
    print()
    print("1. TURN 1 INPUT DEMO (demo_turn1_input.png):")
    print("   - How all training examples will be shown together")
    print("   - Clear input→output relationships")
    print("   - Test input clearly separated")
    print("   - This replaces the current text-only format")
    print()
    print("2. FEEDBACK DEMO (demo_feedback.png):")
    print("   - How training feedback will be visualized")
    print("   - Expected vs Predicted grids side-by-side")
    print("   - Clear success/failure indicators")
    print("   - Pixel accuracy percentages")
    print("   - This supplements the current text feedback")
    print()
    print("NEXT STEPS:")
    print("- Review these images")
    print("- Approve the visual format")
    print("- Integrate into run_arc_tasks.py")

if __name__ == "__main__":
    main() 