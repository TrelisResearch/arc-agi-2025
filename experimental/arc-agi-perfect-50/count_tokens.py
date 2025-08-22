from transformers import AutoTokenizer
import pandas as pd

print("Loading qwen3-4b tokenizer...")
try:
    # Try the exact model name as requested
    tokenizer = AutoTokenizer.from_pretrained("qwen/qwen3-4b")
except:
    try:
        # Try with capital Q
        tokenizer = AutoTokenizer.from_pretrained("Qwen/qwen3-4b")
    except:
        try:
            # Try alternative naming conventions
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
        except:
            try:
                tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-3-4B")
            except:
                # If specific model not available, note it
                print("Note: qwen3-4b not found, trying Qwen2.5-3B as closest match")
                tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")

# Load the longest program
with open('longest_program.py', 'r') as f:
    longest_code = f.read()

# Remove the comment header to get just the code
code_lines = longest_code.split('\n')
# Skip comment lines at the top
code_start = 0
for i, line in enumerate(code_lines):
    if not line.startswith('#') and line.strip():
        code_start = i
        break
just_code = '\n'.join(code_lines[code_start:])

# Tokenize
tokens = tokenizer.encode(just_code)

print(f"\n" + "="*50)
print(f"LONGEST PROGRAM TOKEN COUNT")
print(f"="*50)
print(f"Tokenizer used: {tokenizer.name_or_path}")
print(f"Character count: {len(just_code):,}")
print(f"Token count: {len(tokens):,}")
print(f"Avg chars per token: {len(just_code)/len(tokens):.2f}")

# Also check a few other programs for comparison
df = pd.read_parquet('dataset.parquet')
df['code_length'] = df['code'].str.len()
top_5 = df.nlargest(5, 'code_length')

print(f"\n" + "="*50)
print(f"TOKEN COUNTS FOR TOP 5 LONGEST PROGRAMS")
print(f"="*50)
for idx, row in top_5.iterrows():
    tokens = tokenizer.encode(row['code'])
    print(f"Task {row['task_id']}: {row['code_length']:,} chars → {len(tokens):,} tokens (ratio: {row['code_length']/len(tokens):.2f})")

# Find the program with most tokens (might be different from longest by chars)
print(f"\n" + "="*50)
print(f"FINDING PROGRAM WITH MOST TOKENS...")
print(f"="*50)
df['token_count'] = df['code'].apply(lambda x: len(tokenizer.encode(x)))
most_tokens = df.loc[df['token_count'].idxmax()]
print(f"Task with most tokens: {most_tokens['task_id']}")
print(f"  Characters: {most_tokens['code_length']:,}")
print(f"  Tokens: {most_tokens['token_count']:,}")
print(f"  Model: {most_tokens['model']}")

if most_tokens['task_id'] != top_5.iloc[0]['task_id']:
    print(f"\nNote: The program with most tokens is different from the longest by character count!")