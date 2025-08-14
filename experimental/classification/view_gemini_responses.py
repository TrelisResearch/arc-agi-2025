#!/usr/bin/env python3

import json
from pathlib import Path

def view_gemini_responses():
    """Display Gemini's full responses with reasoning for each program."""
    
    # Load the dataset
    data_file = Path(__file__).parent / "data" / "program_analysis_dataset.json"
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    programs = data['programs']
    
    print("🤖 GEMINI CLASSIFICATION RESPONSES WITH REASONING")
    print("=" * 80)
    print(f"Total programs: {len(programs)}")
    print()
    
    # Show summary first
    human_general = [i for i, p in enumerate(programs) if p['human_classification'] == 'general']
    gemini_general = [i for i, p in enumerate(programs) if p['gemini_classification'] == 'general']
    
    print(f"Human 'general' classifications: {human_general}")
    print(f"Gemini 'general' classifications: {gemini_general}")
    print(f"Agreement: {'✅ Perfect!' if human_general == gemini_general else '❌ Disagreement'}")
    print()
    
    # Display each program's reasoning
    for i, program in enumerate(programs):
        print(f"📋 PROGRAM {i} (1-based: {i+1}) - {program['model']}")
        print(f"   Human: {program['human_classification']}")
        print(f"   Gemini: {program['gemini_classification']}")
        
        # Show agreement/disagreement
        agreement = "✅" if program['human_classification'] == program['gemini_classification'] else "❌"
        print(f"   Agreement: {agreement}")
        
        print("\n🤖 Gemini's Full Response:")
        print("-" * 40)
        print(program['gemini_full_response'])
        print("-" * 40)
        
        # Show code preview
        code_preview = program['code'][:200] + "..." if len(program['code']) > 200 else program['code']
        print(f"\n💻 Code Preview:\n{code_preview}")
        
        print("\n" + "=" * 80 + "\n")

if __name__ == "__main__":
    view_gemini_responses()