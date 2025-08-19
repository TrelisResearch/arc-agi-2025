import pandas as pd
from datasets import load_dataset
import json
import ast
from typing import List, Dict, Any, Tuple

def download_and_filter_dataset():
    print("Downloading dataset from Hugging Face...")
    dataset = load_dataset("Trelis/arc-programs-correct-50", split="train")
    
    df = dataset.to_pandas()
    
    models_to_keep = [
        "Qwen2.5-72B-Instruct",
        "Mistral-Large-Instruct-2407", 
        "Qwen2.5-Coder-32B-Instruct"
    ]
    
    print(f"Original dataset size: {len(df)}")
    filtered_df = df[df['model'].isin(models_to_keep)]
    print(f"Filtered dataset size: {len(filtered_df)}")
    
    model_counts = filtered_df['model'].value_counts()
    print("\nRows per model:")
    for model, count in model_counts.items():
        print(f"  {model}: {count}")
    
    return filtered_df

def parse_grid(grid_str):
    if isinstance(grid_str, str):
        try:
            return ast.literal_eval(grid_str)
        except:
            return json.loads(grid_str)
    return grid_str

def detect_transduction_method1(row):
    try:
        predicted_test = parse_grid(row['predicted_test_output'])
        correct_train = parse_grid(row['correct_train_input'])
        
        if not predicted_test or not correct_train:
            return False
            
        if isinstance(predicted_test[0], list):
            predicted_test = predicted_test[0]
        if isinstance(correct_train[0], list):
            correct_train = correct_train[0]
            
        pred_flat = [item for sublist in predicted_test for item in sublist] if predicted_test else []
        train_flat = [item for sublist in correct_train for item in sublist] if correct_train else []
        
        if not pred_flat or not train_flat:
            return False
            
        return pred_flat == train_flat
    except Exception as e:
        return False

def detect_transduction_method2(row):
    try:
        predicted_test = parse_grid(row['predicted_test_output'])
        correct_test = parse_grid(row['correct_test_input'])
        
        if not predicted_test or not correct_test:
            return False
            
        if isinstance(predicted_test[0], list):
            predicted_test = predicted_test[0]
        if isinstance(correct_test[0], list):
            correct_test = correct_test[0]
            
        pred_flat = [item for sublist in predicted_test for item in sublist] if predicted_test else []
        test_flat = [item for sublist in correct_test for item in sublist] if correct_test else []
        
        if not pred_flat or not test_flat:
            return False
            
        return pred_flat == test_flat
    except Exception as e:
        return False

def main():
    df = download_and_filter_dataset()
    
    print("\n" + "="*60)
    print("Applying transduction detection methods...")
    print("="*60)
    
    df['transductive_method1'] = df.apply(detect_transduction_method1, axis=1)
    df['transductive_method2'] = df.apply(detect_transduction_method2, axis=1)
    
    print("\n--- METHOD 1: Predicted test output == Correct train input ---")
    method1_count = df['transductive_method1'].sum()
    print(f"Total transductive cases: {method1_count}/{len(df)} ({method1_count/len(df)*100:.2f}%)")
    
    print("\nBreakdown by model:")
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        trans_count = model_df['transductive_method1'].sum()
        print(f"  {model}: {trans_count}/{len(model_df)} ({trans_count/len(model_df)*100:.2f}%)")
    
    print("\n--- METHOD 2: Predicted test output == Correct test input ---")
    method2_count = df['transductive_method2'].sum()
    print(f"Total transductive cases: {method2_count}/{len(df)} ({method2_count/len(df)*100:.2f}%)")
    
    print("\nBreakdown by model:")
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        trans_count = model_df['transductive_method2'].sum()
        print(f"  {model}: {trans_count}/{len(model_df)} ({trans_count/len(model_df)*100:.2f}%)")
    
    both_transductive = df['transductive_method1'] & df['transductive_method2']
    print(f"\n--- BOTH METHODS ---")
    print(f"Cases transductive by both methods: {both_transductive.sum()}/{len(df)} ({both_transductive.sum()/len(df)*100:.2f}%)")
    
    either_transductive = df['transductive_method1'] | df['transductive_method2']
    print(f"Cases transductive by either method: {either_transductive.sum()}/{len(df)} ({either_transductive.sum()/len(df)*100:.2f}%)")
    
    df.to_csv('experimental/transduction_hf_dataset/results.csv', index=False)
    print(f"\nResults saved to experimental/transduction_hf_dataset/results.csv")

if __name__ == "__main__":
    main()