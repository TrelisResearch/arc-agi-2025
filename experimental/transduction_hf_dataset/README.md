# Transduction Detection Analysis - Hugging Face Datasets

Analysis of transduction detection using two methods on Hugging Face ARC datasets.

## Datasets Analyzed

1. **Trelis/arc-programs-correct-50**: 9,866 programs
2. **Trelis/arc-programs-50-full-200-partial**: 50,930 programs

Filtered for models: Qwen2.5-72B-Instruct, Mistral-Large-Instruct-2407, Qwen2.5-Coder-32B-Instruct

## Methods

1. **Augmentation Method**: Tests program invariance under transformations (flip, rotation)
2. **Code-based Method**: ML classifier on program features

## Results

### Dataset 1 (arc-programs-correct-50)
- **Augmentation**: 1,398/9,866 (**14.17%** transductive)
- **Code-based**: 486/9,866 (**4.93%** transductive, confidence: 0.910)
- **Agreement**: 85.5%
- **Either method**: 1,655/9,866 (**16.77%**)

### Dataset 2 (arc-programs-50-full-200-partial)  
- **Augmentation**: 23,559/50,930 (**46.26%** transductive)
- **Code-based**: 13,150/50,930 (**25.82%** transductive, confidence: 0.882)
- **Agreement**: 65.0%
- **Either method**: 27,271/50,930 (**53.55%**)

## Key Findings

- Dataset 2 shows much higher transduction rates than Dataset 1
- Augmentation method consistently detects more cases than code-based method
- Mistral-Large-Instruct-2407 shows highest transduction rates
- Methods are complementary with significant unique detections

## Files

- `analyze_optimized.py`: Main analysis script with parallel processing
- `Trelis_arc-programs-correct-50_results.csv`: Results for dataset 1
- `Trelis_arc-programs-50-full-200-partial_results.csv`: Results for dataset 2