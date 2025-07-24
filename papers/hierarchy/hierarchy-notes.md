# Hierarchical Reasoning Model (HRM) Explanation

## Ronan's Comments

- This is a **pure neural transductive approach**.
- **BUT**, it trains a planning type module that has **no direct access to the problem input-pairs**.
- The net is trained only on **one input-output pair at a time**, but also takes in positional embeddings for the grids AND an embedding of the task id (to link examples for the same task).
- Training data is made up of training examples from training and evaluation datasets, **augmented by a factor of 1000 through simple transformations** (flips, rotations, recolours).
- The test output is evaluated by a **majority vote over augmentations**.
- Basically it performs like Greenblatt and like TTFT on ARC-AGI-1 even though the two neural nets used are **only <50M params**.
- There is **feedback in latent space** but there is **no execution feedback**, as there is no DSL. Possibly a weakness for generalisation.
- Also, seems limited on ARC AGI II suggesting that there is **no ability to smartly search/adapt to new operations**.

---

## How HRM Works

### Core Architecture: Hierarchical Neural Network

HRM is a **hierarchical transformer-based architecture** with two key innovations:

1. **Two-Level Hierarchy**: Fast detailed processing (L-level) + slow abstract planning (H-level)
2. **Adaptive Computation**: Model decides when to stop "thinking" using Q-learning

### The Two-Level System

Both levels are **identical transformer architectures** but serve different roles:

**L-Level (Low-Level, Fast Processing):**
- Processes raw input grids + receives guidance from H-level
- Does detailed pattern recognition and computation
- Updates frequently (multiple times per reasoning step)
- Like visual cortex: "I see red squares in corners"

**H-Level (High-Level, Abstract Planning):**
- **No direct access to raw input** - only sees L-level's processed information
- Maintains strategic plans and high-level patterns
- Updates less frequently
- Like prefrontal cortex: "The pattern might be rotation"

### Input Processing

Each example gets:
1. **Token embeddings**: Flattened input/output grids (30x30 → 900 tokens)
2. **Puzzle embeddings**: Shared embedding for all examples from same puzzle type
3. **Position embeddings**: Spatial relationships in the grid

### The Reasoning Process

```
Input: Puzzle embedding + input grid + position info
       ↓
   Initial state (z_H, z_L)
       ↓
   Reasoning cycles:
   for H_step in range(2):
       for L_step in range(2):
           z_L = L_level(z_L, z_H + input)  # L gets H guidance + raw input
       z_H = H_level(z_H, z_L)              # H gets only L's analysis
       ↓
   Decision: Should I stop thinking?
   halt = (q_halt_logits > q_continue_logits)
       ↓
   Output: Predicted grid + confidence
```

### Working Memory (Carry State)

The model maintains **persistent thought vectors** across reasoning steps:
- `z_H`: High-level strategic thoughts
- `z_L`: Low-level detailed analysis

These carry forward between iterations, allowing the model to "change its mind" and refine solutions.

### Adaptive Computation Time (ACT)

The model learns **when to stop reasoning** through Q-learning:
- **q_halt_logits**: "How confident am I that I should stop?"
- **q_continue_logits**: "How confident am I that I should keep thinking?"
- Can reason for 1-16 steps depending on problem difficulty

### Training Approach

**Data:**
- Individual input-output pairs (not all examples from a puzzle at once)
- 960 ARC-AGI-1 puzzles → 960,000 training examples (1000x augmentation)
- Augmentations: rotations, flips, color permutations

**Multi-Objective Loss:**
1. **Language modeling loss**: Predict correct output grid
2. **Q-halt loss**: Learn when current answer is correct
3. **Q-continue loss**: Learn long-term reasoning value (bootstrapping)

**Key Insight:** Model processes examples individually but shares knowledge through puzzle embeddings.

### Evaluation

**Test-time inference:**
1. Generate predictions on multiple augmented versions
2. Apply inverse transformations to get back to original orientation
3. **Majority vote** across augmentations for final answer

---

## Why This Works

1. **Hierarchical Separation**: H-level forced to think abstractly (no raw input access)
2. **Iterative Refinement**: Can "think step by step" like humans
3. **Adaptive Depth**: Spends more time on harder problems
4. **Shared Knowledge**: Puzzle embeddings link examples from same task
5. **Data Efficiency**: Only 27M parameters, ~1000 training examples per task

---

## Limitations

1. **No execution feedback**: Pure latent space reasoning, no DSL interaction
2. **Limited adaptability**: Struggles with completely novel operations (ARC-AGI-2)
3. **No explicit search**: Can't systematically explore solution space
4. **Transductive only**: Trained on specific task distribution

---

## Concise Q&A

**Q: Why batch dimension?**
A: Pure parallelization - process multiple puzzle examples simultaneously on GPU.

**Q: Why both halt AND continue losses?**
A: Q-learning with bootstrapping. Halt = immediate reward, Continue = future value estimation.

**Q: How do the two transformer levels differ?**
A: Same architecture, different inputs. L-level gets raw input + H guidance. H-level gets only L's analysis (forced abstraction).

**Q: How do iterations connect?**
A: Carry state (z_H, z_L) maintains "working memory" between reasoning steps. Model refines thoughts over multiple iterations.

**Q: Are all puzzle examples processed together?**
A: No - each input-output pair processed individually, but shares puzzle embedding to link examples from same task.

**Q: What does the model output?**
A: Three things: (1) Predicted output grid, (2) Halt confidence, (3) Continue confidence.

**Q: How is this different from typical transformers?**
A: (1) Hierarchical (two levels), (2) Adaptive computation (variable reasoning steps), (3) Persistent working memory across iterations.

**Q: Why does it work with so few parameters?**
A: Efficient architecture + massive augmentation + transductive learning on specific task distribution. 