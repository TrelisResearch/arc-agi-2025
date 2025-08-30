from datasets import load_dataset
import json

print("Loading Trelis/arc-agi-2-partial-100 dataset...")
dataset = load_dataset("Trelis/arc-agi-2-partial-100", split="train", streaming=True)

# Get first sample to understand structure
sample = next(iter(dataset))

print("\n=== Dataset Column Names ===")
print(list(sample.keys()))

print("\n=== Column Types and Previews ===")
for key, value in sample.items():
    if isinstance(value, str):
        print(f"\n{key}: string (length {len(value)})")
        if len(value) < 500:
            print(f"  Full content: {value}")
        else:
            print(f"  Preview: {value[:500]}...")
    elif isinstance(value, list):
        print(f"\n{key}: list (length {len(value)})")
        if value and len(str(value[0])) < 200:
            print(f"  First item: {value[0]}")
    elif isinstance(value, dict):
        print(f"\n{key}: dict")
        print(f"  Keys: {list(value.keys())}")
    else:
        print(f"\n{key}: {type(value).__name__} = {value}")

# Check if there's programs field
if 'programs' in sample:
    print("\n=== Programs Field Analysis ===")
    programs = sample['programs']
    if isinstance(programs, str):
        try:
            programs_data = json.loads(programs)
            print(f"Programs is JSON string, parsed to: {type(programs_data)}")
            if isinstance(programs_data, list):
                print(f"Number of programs: {len(programs_data)}")
                if programs_data:
                    print(f"First program structure: {list(programs_data[0].keys()) if isinstance(programs_data[0], dict) else type(programs_data[0])}")
                    print(f"Sample program: {programs_data[0]}")
        except:
            print("Programs is a string but not JSON")
    elif isinstance(programs, list):
        print(f"Programs is a list with {len(programs)} items")
        if programs:
            print(f"First program: {programs[0]}")

# Check for correctness/evaluation fields
correctness_fields = ['train_correct', 'test_correct', 'correct', 'evaluation', 'results', 'outputs']
print("\n=== Checking for Correctness/Evaluation Fields ===")
for field in correctness_fields:
    if field in sample:
        print(f"Found field: {field}")
        print(f"  Type: {type(sample[field])}")
        print(f"  Value preview: {str(sample[field])[:200]}")