from datasets import load_dataset
import json

print("Loading dataset sample to inspect structure...")
dataset = load_dataset("Trelis/arc-agi-2-partial-100", split="train", streaming=True)

sample = next(iter(dataset))
print("\nDataset sample keys:")
print(list(sample.keys()))

print("\nFirst sample structure:")
for key, value in sample.items():
    if isinstance(value, str):
        print(f"  {key}: string (length {len(value)})")
        if len(value) < 200:
            print(f"    Preview: {value[:100]}...")
        else:
            print(f"    Preview: {value[:200]}...")
    elif isinstance(value, (list, dict)):
        print(f"  {key}: {type(value).__name__}")
        if isinstance(value, dict):
            print(f"    Keys: {list(value.keys())}")
    else:
        print(f"  {key}: {type(value).__name__}")

print("\nAttempting to parse 'train' field if it exists...")
if 'train' in sample:
    try:
        if isinstance(sample['train'], str):
            train_data = json.loads(sample['train'])
        else:
            train_data = sample['train']
        print(f"Train data type: {type(train_data)}")
        if isinstance(train_data, list):
            print(f"Number of train examples: {len(train_data)}")
            if train_data:
                print(f"First train example keys: {list(train_data[0].keys())}")
    except Exception as e:
        print(f"Error parsing train: {e}")

print("\nAttempting to parse 'test' field if it exists...")
if 'test' in sample:
    try:
        if isinstance(sample['test'], str):
            test_data = json.loads(sample['test'])
        else:
            test_data = sample['test']
        print(f"Test data type: {type(test_data)}")
        if isinstance(test_data, list):
            print(f"Number of test examples: {len(test_data)}")
            if test_data:
                print(f"First test example keys: {list(test_data[0].keys())}")
    except Exception as e:
        print(f"Error parsing test: {e}")