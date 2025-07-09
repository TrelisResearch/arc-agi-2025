#!/usr/bin/env python3

import os
import json
import argparse
import datetime
import time
import threading
import base64
import io
from pathlib import Path
from typing import Dict, List, Optional
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

from task_loader import TaskLoader
from scoring import GridScorer, ProgramExecutor

load_dotenv()

class ARCGridVisualizer:
    """Create visual representations of ARC grids for use with vision models"""
    
    def __init__(self, pixel_size=20, grid_spacing=5, section_spacing=30, debug_save=False):
        self.pixel_size = pixel_size
        self.grid_spacing = grid_spacing
        self.section_spacing = section_spacing
        self.debug_save = debug_save
        
        # Create debug directory if debug mode is enabled
        if self.debug_save:
            self.debug_dir = Path("debug_images")
            self.debug_dir.mkdir(exist_ok=True)
        
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
    
    def create_training_examples_image(self, task_data, task_id=None, turn=None):
        """Create image showing all training examples with input->output pairs"""
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
        total_height = 40  # Training section header
        for pair in example_pairs:
            total_height += pair.height + self.section_spacing
        total_height += 40  # Test section header
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
        
        # Debug: Save image if debug mode is enabled
        if self.debug_save and task_id:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            turn_info = f"_turn{turn}" if turn is not None else ""
            filename = f"{timestamp}_{task_id}{turn_info}_training.png"
            debug_path = self.debug_dir / filename
            final_img.save(debug_path)
            print(f"🐛 Debug: Saved training image to {debug_path}")
        
        return final_img
    
    def create_feedback_image(self, training_examples, predicted_outputs, training_results, task_id=None, turn=None):
        """Create visual feedback showing expected vs predicted outputs"""
        feedback_rows = []
        
        for i, (example, predicted, result) in enumerate(zip(training_examples, predicted_outputs, training_results)):
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
            
            if result['solved']:
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
        max_width = max(row.width for row in feedback_rows) if feedback_rows else 400
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
        
        # Debug: Save image if debug mode is enabled
        if self.debug_save and task_id:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            turn_info = f"_turn{turn}" if turn is not None else ""
            filename = f"{timestamp}_{task_id}{turn_info}_feedback.png"
            debug_path = self.debug_dir / filename
            final_img.save(debug_path)
            print(f"🐛 Debug: Saved feedback image to {debug_path}")
        
        return final_img
    
    def image_to_base64(self, img):
        """Convert PIL Image to base64 string"""
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return img_str

def execute_with_timeout(func, *args, timeout=300, **kwargs):
    """Execute a function with timeout using ThreadPoolExecutor"""
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout)
        except Exception as e:
            # Cancel the future if it's still running
            future.cancel()
            raise e

class ARCTaskRunner:
    """Run ARC tasks using the OpenAI Responses API (single-shot with tool execution)"""
    
    def __init__(self, model: str = "gpt-4.1-nano", reasoning_effort: str = "low", max_workers: int = 1, rate_limit_delay: float = 0.0, max_turns: int = 3, debug_images: bool = False, enable_images: bool = True):
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.max_workers = max_workers
        self.rate_limit_delay = rate_limit_delay
        self.max_turns = max_turns
        self.debug_images = debug_images
        self.enable_images = enable_images
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.client = OpenAI()
        self.task_loader = TaskLoader()
        self.scorer = GridScorer()
        self.executor = ProgramExecutor(timeout=0.5)
        
        # Initialize visual capabilities
        self.use_visuals = self.enable_images and PIL_AVAILABLE and self.is_vision_model()
        if self.use_visuals:
            self.visualizer = ARCGridVisualizer(debug_save=self.debug_images)
            debug_msg = " with debug image saving" if self.debug_images else ""
            print(f"🖼️  Visual mode enabled for {self.model} (PIL available: {PIL_AVAILABLE}){debug_msg}")
        else:
            self.visualizer = None
            if not self.enable_images:
                print(f"📝 Text-only mode for {self.model} (images disabled by user)")
            elif not PIL_AVAILABLE:
                print("⚠️  PIL not available - install pillow for visual features")
            elif not self.is_vision_model():
                print(f"📝 Text-only mode for {self.model} (no vision support)")
        
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
    
    def is_vision_model(self, model: str = None) -> bool:
        """Check if the model supports vision/image inputs"""
        model_to_check = model or self.model
        model_lower = model_to_check.lower()
        # o4-mini and o3 (but not o3-mini) support vision
        # Also gpt-4o and gpt-4.1 series support vision
        return (model_lower.startswith('o4-mini') or 
                (model_lower.startswith('o3') and not model_lower.startswith('o3-mini')) or
                model_lower.startswith('gpt-4o') or 
                model_lower.startswith('gpt-4.1') or
                model_lower.startswith('gpt-image'))
    
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
        
        # Default fallback (gpt-4o-mini pricing)
        else:
            return (0.15, 0.60)
    
    def create_prompt(self, task_data: Dict, is_first_turn: bool = True, task_id: str = None, turn: int = None) -> List:
        """Create a prompt for the model to solve an ARC task (returns messages for responses API)"""
        if not is_first_turn:
            # For subsequent turns, encourage partial solutions
            return [{"type": "input_text", "text": """Please analyze the training feedback and programs from all prior turns, and write an improved transformation.

**Important**:
1. Always start by attempting to find a complete rule that solves all training examples and should generalize to the test input.
2. If you cannot find a perfect solution, provide your best attempt at a transformation that solves as many training examples as possible - ideally more than in the best previous turn.
3. Even if your solution doesn't solve all examples perfectly, ensure it demonstrates meaningful pattern recognition and provides reasonable outputs that minimize pixel-level errors.

Final answer:
```python
def transform(grid):
    # Your improved transformation logic here (even if partial)
    return transformed_grid
```"""}]
        
        # Create the text content
        task_str = self.task_loader.format_task_for_prompt(task_data, include_test=True)
        
        text_content = f"""You are solving an ARC (Abstraction and Reasoning Corpus) task. 
I will show you training examples with input and output grids, plus a test input grid. Your task is to:

1. **Analyze the training examples** to discover patterns that map input grids to output grids
2. **Write a Python program** that implements your best understanding of the transformation  
3. **DO NOT predict or generate the test output** - your job is only to write the transformation program
4. **Attempt a solution** - even if the pattern isn't completely clear, provide your best hypothesis
5. **Do not repeat the same transformation** - if you have already tried a transformation, do not repeat it.

The test input is shown for context so you understand what type of grid your program will eventually process. Focus on learning patterns from training examples and writing code that captures your understanding.

{task_str}

Analyze the patterns in the training examples and write a Python function that performs this transformation.

**Approach Guidelines:**
- Look for patterns in shapes, colors, positions, sizes, rotations, reflections, etc.
- Even if you can't solve all training examples perfectly, implement what patterns you do observe
- A partial solution that captures some aspects is better than returning the input unchanged
- If the pattern is unclear, make your best educated guess based on what you can see

Requirements:
- The function takes a 2D list (grid) where grid[row][col] gives the value at that position
- Values are integers from 0-9
- Return a new grid (2D list) with the transformation applied
- You can use numpy if needed - just add 'import numpy as np' at the start of your function
- Aim to handle the training examples as well as possible, even if not perfectly
- Your function should attempt some meaningful transformation based on the patterns you observe

You MUST end your response with the following exact format:

Final answer:
```python
def transform(grid):
    # Your transformation logic here (implement your best understanding)
    return transformed_grid
```
"""
        
        # Create message content
        messages = []
        messages.append({"type": "input_text", "text": text_content})
        
        # Add visual representation if available
        if self.use_visuals and self.visualizer:
            try:
                training_image = self.visualizer.create_training_examples_image(task_data, task_id, turn)
                image_base64 = self.visualizer.image_to_base64(training_image)
                
                messages.append({
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{image_base64}"
                })
                
                # Add explanation of the visual
                messages.append({
                    "type": "input_text", 
                    "text": "Above is a visual representation of all the training examples and the test input. Each grid cell is colored according to its value (0-9), with clear labels and arrows showing input→output transformations. Use both the textual and visual information to understand the pattern."
                })
                
            except Exception as e:
                print(f"⚠️  Failed to create visual representation: {e}")
        
        return messages
    
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
                kwargs["include"] = ["reasoning.encrypted_content"]
                kwargs["store"] = False  # Enable stateless mode for encrypted content
            
            # Make the API call
            response = self.client.responses.create(**kwargs)
            
            return response
            
        except Exception as e:
            raise Exception(f"API call failed: {e}")
    
    def create_training_feedback(self, program: str, training_examples: List[Dict], test_correct: bool = None, task_id: str = None, turn: int = None) -> tuple[List, int, float]:
        """Generate detailed training feedback with stats, actual outputs, and optional visuals for LLM"""
        results = []
        predicted_outputs = []
        total_pixels = 0
        correct_pixels = 0
        solved_count = 0
        
        for i, example in enumerate(training_examples):
            predicted_output, error, timed_out = self.executor.execute_program(program, example['input'])
            expected_output = example['output']
            predicted_outputs.append(predicted_output)
            
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
        
        # Format feedback for LLM with encouraging language
        feedback_text = f"Training results: {solved_count}/{len(training_examples)} examples solved, {overall_accuracy:.1%} pixel accuracy"
        
        # Add encouraging context for partial solutions
        if solved_count == len(training_examples) and test_correct is False:
            feedback_text += " - Perfect training accuracy but test failed! Your transformation is overfitting to the training examples. Try to generalize your approach - look for broader patterns that work across all cases, not just the specific training examples.\n\n"
        elif solved_count == 0 and overall_accuracy > 0:
            feedback_text += " - Good partial progress! Your approach is capturing some patterns. Next, try to identify a transformation that solves at least one of the training examples.\n\n"
        elif solved_count == 0 and overall_accuracy == 0:
            feedback_text += " - Keep experimenting! Try a different approach to the transformation.\n\n"
        elif solved_count > 0:
            feedback_text += f" - Great progress! You're on the right track. Next, try to identify a transformation that solves more than {solved_count} of the training examples.\n\n"
        else:
            feedback_text += "\n\n"
        
        for result in results:
            status = "✓" if result['solved'] else "✗"
            feedback_text += f"Training Example {result['index']} {status}:\n"
            feedback_text += f"Expected: {result['expected']}\n"
            
            if result['error']:
                feedback_text += f"Error: {result['error']}\n"
            elif result['timed_out']:
                feedback_text += f"Error: Code execution timed out\n"
            else:
                feedback_text += f"Your output: {result['predicted']}\n"
                if not result['solved']:
                    feedback_text += f"Pixel accuracy: {result['pixel_accuracy']:.1%}\n"
            feedback_text += "\n"
        
        # Create message list with text and optional visual feedback
        feedback_messages = []
        feedback_messages.append({"type": "input_text", "text": feedback_text})
        
        # Add visual feedback if available
        if self.use_visuals and self.visualizer:
            try:
                # Create visual feedback showing expected vs predicted
                feedback_image = self.visualizer.create_feedback_image(training_examples, predicted_outputs, results, task_id, turn)
                image_base64 = self.visualizer.image_to_base64(feedback_image)
                
                feedback_messages.append({
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{image_base64}"
                })
                
                feedback_messages.append({
                    "type": "input_text",
                    "text": "Above is a visual comparison of expected vs predicted outputs for each training example. Green circles indicate correct solutions, red circles show errors. Use this visual feedback along with the textual analysis to improve your approach."
                })
                
            except Exception as e:
                print(f"⚠️  Failed to create visual feedback: {e}")
        
        return feedback_messages, solved_count, overall_accuracy
    
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
    
    def create_success_result(self, task_id: str, program: str, response, test_score: Dict, total_cost: float, total_tokens: int, turns_used: int, task_data: Dict, conversation_history: List = None, all_responses: List = None, turn_details: List = None) -> Dict:
        """Create a successful task result with complete multi-turn data"""

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

        # Convert all responses to JSON-serializable format
        all_responses_dict = []
        if all_responses:
            for resp in all_responses:
                try:
                    output_items = []
                    for item in resp.output:
                        if item.type == 'message':
                            content_texts = []
                            if hasattr(item, 'content'):
                                for content_item in item.content:
                                    if hasattr(content_item, 'text'):
                                        content_texts.append(content_item.text)
                            output_items.append({'type': item.type, 'content': content_texts})
                        else:
                            output_items.append({'type': item.type, 'content': str(getattr(item, 'content', ''))})
                    
                    all_responses_dict.append({
                        'id': resp.id,
                        'model': resp.model,
                        'usage': {
                            'input_tokens': resp.usage.input_tokens,
                            'output_tokens': resp.usage.output_tokens,
                            'total_tokens': resp.usage.total_tokens
                        },
                        'output': output_items
                    })
                except Exception as e:
                    all_responses_dict.append({'error': f'Failed to serialize response: {str(e)}'})

        # Serialize conversation history (handle response objects)
        conversation_history_dict = []
        if conversation_history:
            for item in conversation_history:
                try:
                    if hasattr(item, 'type'):  # Response object from API
                        if item.type == 'message':
                            content_texts = []
                            if hasattr(item, 'content'):
                                for content_item in item.content:
                                    if hasattr(content_item, 'text'):
                                        content_texts.append(content_item.text)
                            conversation_history_dict.append({
                                'type': item.type,
                                'role': getattr(item, 'role', 'assistant'),
                                'content': content_texts
                            })
                        else:
                            conversation_history_dict.append({
                                'type': item.type,
                                'content': str(getattr(item, 'content', ''))
                            })
                    else:  # Regular dict message
                        conversation_history_dict.append(item)
                except Exception as e:
                    conversation_history_dict.append({'error': f'Failed to serialize conversation item: {str(e)}'})

        return {
            'task_id': task_id,
            'model': self.model,
            'reasoning_effort': self.reasoning_effort if self.is_reasoning_model() else "N/A",
            'api_type': 'responses_api_multiturn',
            'program': program,
            'execution_error': '',
            'timed_out': False,
            'tokens_used': total_tokens,
            'request_cost': total_cost,
            'turns_used': turns_used,
            'raw_response': response_dict,  # Final response only (for backward compatibility)
            'score': test_score,
            'predicted_output': test_score.get('predicted_output'),
            'actual_output': test_score.get('actual_output'),
            'api_success': True,
            # NEW: Complete multi-turn conversation data
            'multiturn_data': {
                'conversation_history': conversation_history_dict,
                'all_responses': all_responses_dict,
                'turn_details': turn_details or [],
                'total_turns': turns_used
            }
        }
    
    def create_failure_result(self, task_id: str, program: str, all_responses: List, total_cost: float, total_tokens: int, turns_used: int, task_data: Dict, error_msg: str, conversation_history: List = None, turn_details: List = None) -> Dict:
        """Create a failed task result with complete multi-turn data"""
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

        # Convert all responses to JSON-serializable format
        all_responses_dict = []
        if all_responses:
            for resp in all_responses:
                try:
                    output_items = []
                    for item in resp.output:
                        if item.type == 'message':
                            content_texts = []
                            if hasattr(item, 'content'):
                                for content_item in item.content:
                                    if hasattr(content_item, 'text'):
                                        content_texts.append(content_item.text)
                            output_items.append({'type': item.type, 'content': content_texts})
                        else:
                            output_items.append({'type': item.type, 'content': str(getattr(item, 'content', ''))})
                    
                    all_responses_dict.append({
                        'id': resp.id,
                        'model': resp.model,
                        'usage': {
                            'input_tokens': resp.usage.input_tokens,
                            'output_tokens': resp.usage.output_tokens,
                            'total_tokens': resp.usage.total_tokens
                        },
                        'output': output_items
                    })
                except Exception as e:
                    all_responses_dict.append({'error': f'Failed to serialize response: {str(e)}'})

        # Serialize conversation history (handle response objects)
        conversation_history_dict = []
        if conversation_history:
            for item in conversation_history:
                try:
                    if hasattr(item, 'type'):  # Response object from API
                        if item.type == 'message':
                            content_texts = []
                            if hasattr(item, 'content'):
                                for content_item in item.content:
                                    if hasattr(content_item, 'text'):
                                        content_texts.append(content_item.text)
                            conversation_history_dict.append({
                                'type': item.type,
                                'role': getattr(item, 'role', 'assistant'),
                                'content': content_texts
                            })
                        else:
                            conversation_history_dict.append({
                                'type': item.type,
                                'content': str(getattr(item, 'content', ''))
                            })
                    else:  # Regular dict message
                        conversation_history_dict.append(item)
                except Exception as e:
                    conversation_history_dict.append({'error': f'Failed to serialize conversation item: {str(e)}'})
        
        return {
            'task_id': task_id,
            'model': self.model,
            'reasoning_effort': self.reasoning_effort if self.is_reasoning_model() else "N/A",
            'api_type': 'responses_api_multiturn',
            'program': program,
            'execution_error': error_msg,
            'timed_out': False,
            'tokens_used': total_tokens,
            'request_cost': total_cost,
            'turns_used': turns_used,
            'raw_response': response_dict,  # Final response only (for backward compatibility)
            'score': {
                'correct': False,
                'pixel_accuracy': 0.0,
                'total_pixels': total_pixels,
                'correct_pixels': 0,
                'error': error_msg
            },
            'actual_output': actual_output,
            'api_success': True,
            # NEW: Complete multi-turn conversation data
            'multiturn_data': {
                'conversation_history': conversation_history_dict,
                'all_responses': all_responses_dict,
                'turn_details': turn_details or [],
                'total_turns': turns_used
            }
        }
    
    def create_timeout_failure_result(self, task_id: str, total_cost: float, total_tokens: int, turns_completed: int, task_data: Dict, conversation_history: List = None, all_responses: List = None, turn_details: List = None) -> Dict:
        """Create a timeout failure result - separate from regular task failures"""
        # Get test output for pixel counting
        actual_output = task_data['test'][0]['output']
        total_pixels = len(actual_output) * len(actual_output[0]) if actual_output else 0
        
        # Convert all responses to JSON-serializable format
        all_responses_dict = []
        if all_responses:
            for resp in all_responses:
                try:
                    output_items = []
                    for item in resp.output:
                        if item.type == 'message':
                            content_texts = []
                            if hasattr(item, 'content'):
                                for content_item in item.content:
                                    if hasattr(content_item, 'text'):
                                        content_texts.append(content_item.text)
                            output_items.append({'type': item.type, 'content': content_texts})
                        else:
                            output_items.append({'type': item.type, 'content': str(getattr(item, 'content', ''))})
                    
                    all_responses_dict.append({
                        'id': resp.id,
                        'model': resp.model,
                        'usage': {
                            'input_tokens': resp.usage.input_tokens,
                            'output_tokens': resp.usage.output_tokens,
                            'total_tokens': resp.usage.total_tokens
                        },
                        'output': output_items
                    })
                except Exception as e:
                    all_responses_dict.append({'error': f'Failed to serialize response: {str(e)}'})

        # Serialize conversation history (handle response objects)
        conversation_history_dict = []
        if conversation_history:
            for item in conversation_history:
                try:
                    if hasattr(item, 'type'):  # Response object from API
                        if item.type == 'message':
                            content_texts = []
                            if hasattr(item, 'content'):
                                for content_item in item.content:
                                    if hasattr(content_item, 'text'):
                                        content_texts.append(content_item.text)
                            conversation_history_dict.append({
                                'type': item.type,
                                'role': getattr(item, 'role', 'assistant'),
                                'content': content_texts
                            })
                        else:
                            conversation_history_dict.append({
                                'type': item.type,
                                'content': str(getattr(item, 'content', ''))
                            })
                    else:  # Regular dict message
                        conversation_history_dict.append(item)
                except Exception as e:
                    conversation_history_dict.append({'error': f'Failed to serialize conversation item: {str(e)}'})
        
        return {
            'task_id': task_id,
            'model': self.model,
            'reasoning_effort': self.reasoning_effort if self.is_reasoning_model() else "N/A",
            'api_type': 'responses_api_multiturn',
            'program': '',
            'execution_error': 'API timeout after retries',
            'timed_out': True,
            'tokens_used': total_tokens,
            'request_cost': total_cost,
            'turns_used': turns_completed,
            'raw_response': None,
            'score': {
                'correct': False,
                'pixel_accuracy': 0.0,
                'total_pixels': total_pixels,
                'correct_pixels': 0,
                'error': 'API timeout after retries'
            },
            'actual_output': actual_output,
            'api_success': False,  # This is a timeout failure, not a regular failure
            'timeout_failure': True,  # NEW: Mark as timeout failure
            # NEW: Complete multi-turn conversation data
            'multiturn_data': {
                'conversation_history': conversation_history_dict,
                'all_responses': all_responses_dict,
                'turn_details': turn_details or [],
                'total_turns': turns_completed
            }
        }

    def run_task(self, task_id: str, task_data: Dict, total_tasks: int = 1) -> Dict:
        """Run a single ARC task using multi-turn local code execution"""
        if self.max_workers == 1:  # Only print for sequential execution
            print(f"\nProcessing task: {task_id}")
        
        # Apply rate limiting if configured
        if self.rate_limit_delay > 0:
            time.sleep(self.rate_limit_delay)
        
        return self.run_task_multiturn(task_id, task_data, total_tasks)
    
    def run_task_multiturn(self, task_id: str, task_data: Dict, total_tasks: int = 1) -> Dict:
        """Run multi-turn conversation with local code execution"""
        conversation_history = []
        total_cost = 0.0
        total_tokens = 0
        all_responses = []
        turn_details = []  # NEW: Collect detailed turn-by-turn data
        
        # Start conversation
        system_msg = {"role": "system", "content": "You are an expert at solving abstract reasoning puzzles. Write clean, efficient Python code."}
        initial_prompt_messages = self.create_prompt(task_data, is_first_turn=True, task_id=task_id, turn=1)
        conversation_history = [system_msg, {"role": "user", "content": initial_prompt_messages}]
        
        try:
            for turn in range(self.max_turns):
                turn_start_time = datetime.datetime.now()
                if self.max_workers == 1:  # Only print detailed logs for sequential execution
                    print(f"  🔄 Turn {turn + 1}/{self.max_turns}...")
                
                # Make API call with timeout and retry logic
                response = None
                api_call_successful = False
                
                for attempt in range(3):  # 3 attempts total (initial + 2 retries)
                    try:
                        if self.max_workers == 1 and attempt > 0:
                            print(f"     🔄 Turn {turn + 1} attempt {attempt + 1}/3...")
                        
                        response = execute_with_timeout(self.call_responses_api, conversation_history, timeout=150)
                        api_call_successful = True
                        break  # Success!
                        
                    except Exception as e:
                        if attempt < 2:  # Can still retry
                            if self.max_workers == 1:
                                print(f"     ⏰ Turn {turn + 1} attempt {attempt + 1} timed out, retrying in 2s...")
                            time.sleep(2)  # Brief backoff
                            continue
                        else:  # All retries exhausted
                            if self.max_workers == 1:
                                print(f"     ❌ Turn {turn + 1} failed after 3 attempts: {str(e)}")
                            # Return timeout failure result
                            self._update_costs(total_cost, total_tokens)
                            return self.create_timeout_failure_result(task_id, total_cost, total_tokens, turn, task_data, conversation_history, all_responses, turn_details)
                
                if not api_call_successful or response is None:
                    # This shouldn't happen, but just in case
                    self._update_costs(total_cost, total_tokens)
                    return self.create_timeout_failure_result(task_id, total_cost, total_tokens, turn, task_data, conversation_history, all_responses, turn_details)
                
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
                
                # Initialize turn detail
                turn_detail = {
                    'turn_number': turn + 1,
                    'timestamp': turn_start_time.isoformat(),
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'turn_cost': turn_cost,
                    'program_extracted': bool(program),
                    'program': program,
                    'training_feedback': None,
                    'test_result': None,
                    'status': 'in_progress'
                }
                
                if not program:
                    if self.max_workers == 1:
                        print(f"     ❌ No code found in response")
                    
                    turn_detail['status'] = 'no_code_found'
                    turn_details.append(turn_detail)
                    
                    # Stop if this is the last turn
                    if turn == self.max_turns - 1:
                        if self.max_workers == 1:
                            print(f"     ⏰ Max turns ({self.max_turns}) reached")
                        
                        # BUGFIX: Add final response to conversation history before breaking
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
                    code_request = """I need you to provide Python code to attempt this task. Even if you're not completely certain about the pattern, please provide your best hypothesis as a working transformation function.

**Remember**: A partial solution that captures some observed patterns is much more valuable than refusing to attempt the task. Your goal is to implement whatever understanding you have, even if incomplete.

You MUST end your response with the following exact format:

Final answer:
```python
def transform(grid):
    # Your transformation logic here (implement your best understanding)
    return transformed_grid
```

Make sure to include the function definition inside a proper code block."""
                    
                    conversation_history.append({"role": "user", "content": code_request})
                    turn_detail['code_request_sent'] = True
                    if self.max_workers == 1:
                        print(f"     💬 Requesting code from model...")
                    continue
                
                # Test on test input first
                test_input = task_data['test'][0]['input']
                test_expected = task_data['test'][0]['output']
                predicted_output, error, timed_out = self.executor.execute_program(program, test_input)
                
                # Determine test correctness for training feedback
                test_correct = None
                if predicted_output is not None and not error and not timed_out:
                    test_score = self.scorer.score_grid(predicted_output, test_expected)
                    test_correct = test_score['correct']
                else:
                    test_correct = False
                
                # Get training feedback for this turn (including test result context)
                training_feedback_messages, solved_count, pixel_acc = self.create_training_feedback(program, task_data['train'], test_correct, task_id, turn + 1)
                turn_detail['training_feedback'] = {
                    'feedback_messages': training_feedback_messages,
                    'solved_count': solved_count,
                    'total_training_examples': len(task_data['train']),
                    'pixel_accuracy': pixel_acc
                }
                
                if predicted_output is not None and not error and not timed_out:
                    # Check if test is correct
                    test_score = self.scorer.score_grid(predicted_output, test_expected)
                    turn_detail['test_result'] = test_score
                    
                    if test_score['correct']:
                        # SUCCESS! 
                        turn_detail['status'] = 'success'
                        turn_details.append(turn_detail)
                        
                        if self.max_workers == 1:
                            print(f"     ✅ Perfect solution found! Training: {solved_count}/{len(task_data['train'])} solved, {pixel_acc:.1%} accuracy")
                        
                        # Update costs only (progress handled by parallel executor)
                        self._update_costs(total_cost, total_tokens)
                        
                        # Add actual outputs to test_score for logging
                        test_score['predicted_output'] = predicted_output
                        test_score['actual_output'] = test_expected
                        
                        return self.create_success_result(task_id, program, response, test_score, total_cost, total_tokens, turn + 1, task_data, conversation_history, all_responses, turn_details)
                else:
                    # Execution failed
                    turn_detail['test_result'] = {
                        'execution_error': error,
                        'timed_out': timed_out,
                        'predicted_output': predicted_output
                    }
                
                turn_detail['status'] = 'failed_test'
                turn_details.append(turn_detail)
                
                if self.max_workers == 1:
                    print(f"     📊 Training: {solved_count}/{len(task_data['train'])} solved, {pixel_acc:.1%} accuracy")
                
                # Stop if this is the last turn
                if turn == self.max_turns - 1:
                    if self.max_workers == 1:
                        print(f"     ⏰ Max turns ({self.max_turns}) reached")
                    
                    # BUGFIX: Add final response to conversation history before breaking
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
                
                # Add training feedback - combine prompt messages with feedback messages
                subsequent_prompt_messages = self.create_prompt(task_data, is_first_turn=False, task_id=task_id, turn=turn + 2)
                combined_feedback_content = subsequent_prompt_messages + training_feedback_messages
                conversation_history.append({"role": "user", "content": combined_feedback_content})
                turn_detail['feedback_sent'] = True
            
            # Failed after max turns
            self._update_costs(total_cost, total_tokens)
            
            return self.create_failure_result(task_id, program if 'program' in locals() else "", all_responses, total_cost, total_tokens, self.max_turns, task_data, "Max turns reached", conversation_history, turn_details)
        
        except Exception as e:
            print(f"     ❌ Multi-turn execution failed: {e}")
            self._update_costs(total_cost, total_tokens)
            
            return self.create_failure_result(task_id, "", all_responses, total_cost, total_tokens, len(turn_details), task_data, str(e), conversation_history, turn_details)

    def run_subset(self, subset_name: str, dataset: str = "arc-agi-1", limit: Optional[int] = None) -> List[Dict]:
        """Run all tasks in a subset with optional parallelization"""
        tasks = self.task_loader.load_tasks_from_subset(subset_name, dataset)
        
        if limit:
            tasks = tasks[:limit]
        
        total_tasks = len(tasks)
        
        # Print configuration info
        print(f"\nRunning {total_tasks} tasks from {dataset}/{subset_name}")
        print(f"Model: {self.model}")
        print(f"API: Responses API (multi-turn, max {self.max_turns} turns)")
        print("Tools: ENABLED (multi-turn local execution with training examples feedback)")
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
                    turns = result.get('turns_used', 1)
                    print(f"[{task_id}] {status} (${cost:.6f}, {turns} turns)")
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
                        turns = result.get('turns_used', 1)
                        status_with_turns = f"{status} ({turns} turns)"
                    except Exception as e:
                        print(f"[{task_id}] ❌ SAVE FAILED: {e}")
                        # Create a minimal error result
                        error_result = {
                            'task_id': task_id,
                            'error': str(e),
                            'score': {'correct': False, 'pixel_accuracy': 0.0, 'total_pixels': 0, 'correct_pixels': 0}
                        }
                        results.append(error_result)
                        status_with_turns = "❌ SAVE FAILED"
                    
                    # Show overall progress
                    progress_pct = (completed / total_tasks) * 100
                    print(f"Progress: {completed}/{total_tasks} tasks processed ({progress_pct:.1f}%) - {status_with_turns}")
            
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
        
        # Separate successful API calls from complete failures and timeout failures
        api_successes = [r for r in results if r.get('api_success', True)]  # Default True for backward compatibility
        api_failures = [r for r in results if not r.get('api_success', True) and not r.get('timeout_failure', False)]
        timeout_failures = [r for r in results if r.get('timeout_failure', False)]
        failed_tasks = len(api_failures)
        timeout_tasks = len(timeout_failures)
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
            'api_type': 'responses_api',
            'total_tasks': total_tasks,
            'successful_api_calls': successful_api_calls,
            'failed_api_calls': failed_tasks,
            'timeout_failed_tasks': timeout_tasks,
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
        print(f"API: Responses (multi-turn, max {self.max_turns} turns)")
        print(f"Total tasks attempted: {total_tasks}")
        print(f"Successful API calls: {successful_api_calls}/{total_tasks} ({summary['success_rate']:.1%})")
        if failed_tasks > 0:
            print(f"Failed API calls: {failed_tasks}/{total_tasks} ({failed_tasks/total_tasks:.1%}) ❌")
        if timeout_tasks > 0:
            print(f"Timeout failures: {timeout_tasks}/{total_tasks} ({timeout_tasks/total_tasks:.1%}) ⏰")
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
        
        if timeout_tasks > 0:
            print(f"\n⏰ TIMEOUT FAILURES ({timeout_tasks}):")
            for result in timeout_failures:
                task_id = result.get('task_id', 'unknown')
                turns_completed = result.get('turns_used', 0)
                print(f"  - {task_id}: API timeout after {turns_completed} turns and 3 retries")
        
        print(f"\nResults saved to: {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Run ARC tasks with OpenAI o3/o4 models")
    parser.add_argument("--dataset", default="arc-agi-1", choices=["arc-agi-1", "arc-agi-2"],
                       help="Dataset to use")
    parser.add_argument("--subset", default="shortest_1",
                       help="Subset name (e.g., shortest_1, shortest_10, shortest_100)")
    parser.add_argument("--model", default="gpt-4.1-mini",
                       help="OpenAI model to use")
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
    parser.add_argument("--debug_images", action="store_true",
                       help="Save debug images to debug_images/ directory")
    parser.add_argument("--enable_images", action="store_true", default=True,
                       help="Enable visual image generation for vision models (default: True)")
    parser.add_argument("--disable_images", action="store_true",
                       help="Disable visual image generation (text-only mode)")
    
    args = parser.parse_args()
    
    # Validate max_workers
    if args.max_workers < 1:
        parser.error("--max_workers must be at least 1")
    if args.max_workers > 30:
        parser.error("--max_workers cannot exceed 30 (OpenAI rate limits)")
    
    # Validate rate_limit_delay
    if args.rate_limit_delay < 0:
        parser.error("--rate_limit_delay cannot be negative")
    
    # Handle image enable/disable flags
    enable_images = args.enable_images and not args.disable_images
    
    # Create runner and run tasks
    runner = ARCTaskRunner(model=args.model, reasoning_effort=args.reasoning_effort, max_workers=args.max_workers, rate_limit_delay=args.rate_limit_delay, max_turns=args.max_turns, debug_images=args.debug_images, enable_images=enable_images)
    runner.run_subset(args.subset, args.dataset, args.limit)


if __name__ == "__main__":
    main()