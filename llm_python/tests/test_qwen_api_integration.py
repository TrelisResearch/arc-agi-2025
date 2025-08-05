#!/usr/bin/env python3
"""
Test script for Qwen API integration - tests both reasoning and non-reasoning modes
and measures token usage in each case.
"""

import os
import sys
import json
import time
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def make_single_api_call(client, question, test_id, thinking_budget=2000):
    """Make a single API call for concurrent testing"""
    try:
        start_time = time.time()
        
        kwargs = {
            "model": "qwen3-235b-a22b-thinking-2507",
            "messages": [{"role": "user", "content": question}],
            "temperature": 0.7,
            "max_tokens": 1000,  # Increased for harder problems
            "extra_body": {"thinking_budget": thinking_budget}
        }
        
        response = client.chat.completions.create(**kwargs)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Extract response details
        content = ""
        reasoning_present = False
        reasoning_length = 0
        
        if response.choices and len(response.choices) > 0:
            message = response.choices[0].message
            content = message.content or ""
            
            # Check for reasoning content
            if hasattr(message, 'reasoning_content') and message.reasoning_content:
                reasoning_present = True
                reasoning_length = len(message.reasoning_content)
            elif hasattr(message, 'reasoning') and message.reasoning:
                reasoning_present = True
                reasoning_length = len(message.reasoning)
        
        tokens_per_second = (response.usage.completion_tokens / duration) if response.usage and duration > 0 else 0
        
        return {
            'test_id': test_id,
            'success': True,
            'duration': duration,
            'usage': {
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens
            } if response.usage else None,
            'tokens_per_second': tokens_per_second,
            'response_length': len(content),
            'reasoning_present': reasoning_present,
            'reasoning_length': reasoning_length,
            'thinking_budget': thinking_budget
        }
        
    except Exception as e:
        return {
            'test_id': test_id,
            'success': False,
            'error': str(e),
            'thinking_budget': thinking_budget
        }

def test_concurrent_api_calls(client, question, concurrency=8):
    """Test concurrent API calls with different thinking budgets"""
    print(f"\n🚀 CONCURRENT TEST: {concurrency} parallel requests")
    print("-" * 50)
    
    # Test different thinking budgets
    thinking_budgets = {'low': 1000, 'medium': 4000, 'high': 8000}
    results = {}
    
    for budget_name, budget_value in thinking_budgets.items():
        print(f"📡 Testing {concurrency} concurrent requests with {budget_name} thinking budget ({budget_value} tokens)...")
        start_time = time.time()
        
        budget_results = []
        
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [
                executor.submit(make_single_api_call, client, question, i+1, budget_value)
                for i in range(concurrency)
            ]
            
            for future in as_completed(futures):
                result = future.result()
                budget_results.append(result)
                if result['success']:
                    reasoning_info = f"reasoning: {result['reasoning_length']} chars" if result['reasoning_present'] else "no reasoning"
                    print(f"  ✅ Request {result['test_id']}: {result['tokens_per_second']:.1f} tok/s in {result['duration']:.2f}s, {reasoning_info}")
                else:
                    print(f"  ❌ Request {result['test_id']}: {result['error']}")
        
        total_time = time.time() - start_time
        results[budget_name] = {
            'results': budget_results,
            'total_time': total_time
        }
    
    # Calculate aggregate statistics
    print(f"\n📊 CONCURRENT PERFORMANCE ANALYSIS")
    print("=" * 60)
    
    for budget_name, budget_data in results.items():
        mode_results = budget_data['results']
        wall_clock_time = budget_data['total_time']
        
        successful_results = [r for r in mode_results if r['success']]
        if not successful_results:
            print(f"\n❌ {budget_name.upper()}: No successful requests")
            continue
            
        total_tokens = sum(r['usage']['completion_tokens'] for r in successful_results if r['usage'])
        total_individual_time = sum(r['duration'] for r in successful_results)
        avg_individual_time = total_individual_time / len(successful_results)
        
        # Calculate reasoning statistics
        reasoning_lengths = [r['reasoning_length'] for r in successful_results if r['reasoning_present']]
        avg_reasoning_length = sum(reasoning_lengths) / len(reasoning_lengths) if reasoning_lengths else 0
        
        # Calculate different token/sec metrics
        individual_avg_tps = sum(r['tokens_per_second'] for r in successful_results) / len(successful_results)
        aggregate_tps = total_tokens / wall_clock_time if wall_clock_time > 0 else 0
        theoretical_sequential_time = sum(r['duration'] for r in successful_results)
        
        budget_value = successful_results[0]['thinking_budget'] if successful_results else 0
        
        print(f"\n🧠 {budget_name.upper()} BUDGET ({budget_value} tokens) - ({len(successful_results)}/{concurrency} successful):")
        print(f"  Total output tokens: {total_tokens:,}")
        print(f"  Wall clock time: {wall_clock_time:.2f}s")
        print(f"  Average individual duration: {avg_individual_time:.2f}s")
        print(f"  Average reasoning length: {avg_reasoning_length:.0f} characters")
        print(f"  Total individual time: {total_individual_time:.2f}s")
        print(f"  Sequential time (theoretical): {theoretical_sequential_time:.2f}s")
        print(f"  Speedup vs sequential: {theoretical_sequential_time/wall_clock_time:.2f}x")
        print(f"  Average individual tokens/sec: {individual_avg_tps:.2f}")
        print(f"  Aggregate tokens/sec: {aggregate_tps:.2f}")
        print(f"  Parallelization efficiency: {(theoretical_sequential_time/wall_clock_time)/concurrency:.1%}")
    
    # Add wall clock times to results for final summary
    wall_clock_times = {budget_name: budget_data['total_time'] for budget_name, budget_data in results.items()}
    results['wall_clock_times'] = wall_clock_times
    
    return results

def test_qwen_api_integration():
    """Test Qwen DashScope API with different thinking budgets, measuring tokens"""
    
    # Check if API key is available
    api_key = os.getenv('DASHSCOPE_API_KEY')
    if not api_key:
        print("❌ DASHSCOPE_API_KEY environment variable not found")
        print("Please set your DashScope API key:")
        print("export DASHSCOPE_API_KEY='your-api-key-here'")
        return False
    
    print("🔑 Found DASHSCOPE_API_KEY")
    print("🚀 Testing Qwen API Integration...")
    print("=" * 60)
    
    # Initialize client
    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        timeout=120
    )
    
    # Test question - harder problem requiring more reasoning
    test_question = """
You have a 3x3 grid of lights, initially all OFF. Each light can be toggled ON/OFF. 
When you press a light, it toggles itself AND all adjacent lights (up, down, left, right - not diagonals).

Starting pattern (all OFF):
OFF OFF OFF
OFF OFF OFF  
OFF OFF OFF

If you press the center light, what will the final pattern be? 
Then, if you press the top-left corner light, what will the pattern become?
Explain your reasoning step by step, considering each toggle operation carefully.
""".strip()
    
    results = {}
    
    # Note: enable_thinking=False is not supported by commercial DashScope models
    # The parameter is restricted to True only, so we test different thinking budgets instead
    print("\n📝 NOTE: Commercial DashScope models do not support enable_thinking=False")
    print("Testing different thinking_budget values instead...")
    
    # Test different thinking budgets
    thinking_budgets = {'low': 1000, 'medium': 4000, 'high': 8000}
    
    for budget_name, budget_value in thinking_budgets.items():
        print(f"\n🧠 TEST: {budget_name.upper()} THINKING BUDGET ({budget_value} tokens)")
        print("-" * 50)
        
        try:
            start_time = time.time()
            
            response = client.chat.completions.create(
                model="qwen3-235b-a22b-thinking-2507",
                messages=[
                    {"role": "user", "content": test_question}
                ],
                temperature=0.7,
                max_tokens=1000,  # Increased for harder problem
                extra_body={
                    "thinking_budget": budget_value
                }
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            print("✅ API call successful")
            print(f"📊 Model: {response.model}")
            print(f"⏱️  Duration: {duration:.2f} seconds")
            
            if response.usage:
                usage = response.usage
                tokens_per_second = usage.completion_tokens / duration if duration > 0 else 0
                print(f"🔢 Tokens:")
                print(f"   Input tokens: {usage.prompt_tokens}")
                print(f"   Output tokens: {usage.completion_tokens}")
                print(f"   Total tokens: {usage.total_tokens}")
                print(f"   Tokens/second: {tokens_per_second:.2f}")
            
            if response.choices and len(response.choices) > 0:
                content = response.choices[0].message.content
                print(f"💬 Response length: {len(content)} characters")
                print(f"📝 Response preview: {content[:200]}...")
                
                # Check reasoning content
                message = response.choices[0].message
                reasoning_present = False
                reasoning_length = 0
                
                if hasattr(message, 'reasoning_content') and message.reasoning_content:
                    reasoning_present = True
                    reasoning_length = len(message.reasoning_content)
                    print(f"🤔 Reasoning content found: {reasoning_length} chars")
                    print(f"🔍 Reasoning preview: {message.reasoning_content[:200]}...")
                elif hasattr(message, 'reasoning') and message.reasoning:
                    reasoning_present = True
                    reasoning_length = len(message.reasoning)
                    print(f"🤔 Reasoning found: {reasoning_length} chars")
                    print(f"🔍 Reasoning preview: {message.reasoning[:200]}...")
                else:
                    print("⚠️  No reasoning content found (unexpected)")
            
            tokens_per_second_calc = (response.usage.completion_tokens / duration) if response.usage and duration > 0 else 0
            
            results[budget_name] = {
                'success': True,
                'duration': duration,
                'usage': {
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens
                } if response.usage else None,
                'tokens_per_second': tokens_per_second_calc,
                'response_length': len(content) if 'content' in locals() else 0,
                'reasoning_present': reasoning_present,
                'reasoning_length': reasoning_length,
                'thinking_budget': budget_value
            }
            
        except Exception as e:
            print(f"❌ Error in {budget_name} budget test: {e}")
            results[budget_name] = {'success': False, 'error': str(e), 'thinking_budget': budget_value}
    
    # Compare results across different thinking budgets
    print("\n📊 THINKING BUDGET COMPARISON")
    print("=" * 60)
    
    successful_results = {k: v for k, v in results.items() if v.get('success', False)}
    
    if len(successful_results) >= 2:
        print("Budget Performance Comparison:")
        print(f"{'Budget':<8} {'Tokens':<8} {'Duration':<10} {'Tok/Sec':<10} {'Reasoning':<12}")
        print("-" * 55)
        
        for budget_name, result in successful_results.items():
            if result['usage']:
                tokens = result['usage']['total_tokens']
                duration = result['duration']
                tps = result['tokens_per_second']
                reasoning = f"{result['reasoning_length']} chars" if result['reasoning_present'] else "None"
                budget_val = result['thinking_budget']
                
                print(f"{budget_name:<8} {tokens:<8d} {duration:<10.2f} {tps:<10.2f} {reasoning:<12}")
        
        # Analysis
        if 'low' in successful_results and 'high' in successful_results:
            low_reasoning = successful_results['low']['reasoning_length']
            high_reasoning = successful_results['high']['reasoning_length']
            reasoning_diff = high_reasoning - low_reasoning
            
            print(f"\n🔍 Analysis:")
            print(f"  Reasoning length difference (High vs Low): {reasoning_diff:+d} characters")
            
            if reasoning_diff > 0:
                print(f"  ✅ Higher thinking budget produces more reasoning content")
            elif reasoning_diff == 0:
                print(f"  ⚠️  No difference in reasoning length detected")
            else:
                print(f"  ❓ Lower thinking budget produced more reasoning (unexpected)")
    else:
        print("⚠️  Not enough successful results for comparison")
    
    # Run concurrent tests - COMMENTED OUT FOR NOW
    # concurrent_results = test_concurrent_api_calls(client, test_question, concurrency=8)
    concurrent_results = None
    
    # Save detailed results including concurrent tests
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = Path(__file__).parent / "results" / f"qwen_api_test_{timestamp}.json"
    results_file.parent.mkdir(exist_ok=True)
    
    detailed_results = {
        'timestamp': timestamp,
        'test_question': test_question,
        'thinking_budget_results': results,
        'concurrent_results': concurrent_results
    }
    
    with open(results_file, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: {results_file}")
    print("\n✅ Qwen API integration test completed!")
    
    return results, concurrent_results

def main():
    """Main function to run the test"""
    sequential_results, concurrent_results = test_qwen_api_integration()
    
    print("\n🎯 FINAL SUMMARY")
    print("=" * 60)
    
    successful_sequential = {k: v for k, v in sequential_results.items() if v.get('success', False)}
    if successful_sequential:
        print(f"Sequential Performance:")
        for budget_name, result in successful_sequential.items():
            if result.get('tokens_per_second'):
                print(f"  {budget_name.title()} Budget: {result['tokens_per_second']:.2f} tokens/sec")
    
    # Show concurrent performance if available
    if concurrent_results:
        wall_clock_times = concurrent_results.get('wall_clock_times', {})
        print(f"\nConcurrent Performance (8 parallel requests):")
        
        for budget_name in ['low', 'medium', 'high']:
            if budget_name in concurrent_results and wall_clock_times.get(budget_name):
                budget_data = concurrent_results[budget_name]
                successful = [r for r in budget_data['results'] if r['success']]
                if successful:
                    total_tokens = sum(r['usage']['completion_tokens'] for r in successful if r['usage'])
                    wall_time = wall_clock_times[budget_name]
                    aggregate_tps = total_tokens / wall_time if wall_time > 0 else 0
                    budget_value = successful[0]['thinking_budget']
                    print(f"  {budget_name.title()} Budget ({budget_value} tokens): {aggregate_tps:.2f} tokens/sec (aggregate)")
        
        print(f"\n🔍 Detailed concurrent analysis available in the results file.")
    else:
        print(f"\n🚫 Concurrent testing disabled for this run.")
    
    print("\n🚀 Test completed! Check the detailed results file for full analysis.")

if __name__ == "__main__":
    main()