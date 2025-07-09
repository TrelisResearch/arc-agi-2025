#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'o3-tools'))

from run_arc_tasks import ARCTaskRunner
import json

def test_visual_integration():
    """Test that visual integration works with o4-mini model"""
    print("🧪 Testing visual integration...")
    
    # Test with o4-mini (supports vision)
    print("\n1. Testing with o4-mini (vision-capable)...")
    runner_visual = ARCTaskRunner(model="o4-mini", max_turns=1)
    print(f"   Visual mode enabled: {runner_visual.use_visuals}")
    print(f"   Has visualizer: {runner_visual.visualizer is not None}")
    
    # Test with o3 (supports vision)
    print("\n2. Testing with o3 (vision-capable)...")
    runner_o3 = ARCTaskRunner(model="o3", max_turns=1)
    print(f"   Visual mode enabled: {runner_o3.use_visuals}")
    print(f"   Has visualizer: {runner_o3.visualizer is not None}")
    
    # Test with o3-mini (no vision)
    print("\n3. Testing with o3-mini (no vision support)...")
    runner_text = ARCTaskRunner(model="o3-mini", max_turns=1)
    print(f"   Visual mode enabled: {runner_text.use_visuals}")
    print(f"   Has visualizer: {runner_text.visualizer is not None}")
    
    # Test prompt creation with sample task data
    print("\n4. Testing prompt creation...")
    sample_task = {
        'train': [
            {
                'input': [[1, 0], [0, 1]],
                'output': [[0, 1], [1, 0]]
            }
        ],
        'test': [
            {
                'input': [[1, 1], [0, 0]],
                'output': [[0, 0], [1, 1]]
            }
        ]
    }
    
    # Test visual prompt (o4-mini)
    if runner_visual.use_visuals:
        visual_prompt = runner_visual.create_prompt(sample_task, is_first_turn=True)
        print(f"   o4-mini visual prompt message count: {len(visual_prompt)}")
        print(f"   o4-mini message types: {[msg['type'] for msg in visual_prompt]}")
    
    # Test visual prompt (o3)
    if runner_o3.use_visuals:
        o3_prompt = runner_o3.create_prompt(sample_task, is_first_turn=True)
        print(f"   o3 visual prompt message count: {len(o3_prompt)}")
        print(f"   o3 message types: {[msg['type'] for msg in o3_prompt]}")
    
    # Test text-only prompt (o3-mini)
    text_prompt = runner_text.create_prompt(sample_task, is_first_turn=True)
    print(f"   o3-mini text prompt message count: {len(text_prompt)}")
    print(f"   o3-mini message types: {[msg['type'] for msg in text_prompt]}")
    
    print("\n✅ Visual integration test completed successfully!")

if __name__ == "__main__":
    test_visual_integration() 