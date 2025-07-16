#!/usr/bin/env python3
"""
Verify what happens with max_turns=1 when code extraction fails
"""

print("=== Verifying max_turns=1 Logic ===")

# Simulate the exact condition from run_arc_tasks.py
max_turns = 1

for turn in range(max_turns):
    print(f"Turn {turn} (of max_turns={max_turns})")
    print(f"Condition: turn == self.max_turns - 1")
    print(f"Evaluation: {turn} == {max_turns} - 1")
    print(f"Result: {turn} == {max_turns - 1} → {turn == max_turns - 1}")
    
    if turn == max_turns - 1:
        print("✅ Would break (last turn) - NO more API calls")
        break
    else:
        print("❌ Would continue to next turn - ANOTHER API call")

print("\n=== Conclusion ===")
print("With --max-turns 1, code extraction failure should NOT trigger duplicate calls")
print("The issue must be elsewhere...") 