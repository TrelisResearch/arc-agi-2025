# ARC-AGI Program Style Guide

*A prescriptive guide for writing effective ARC-AGI transformation programs*

## Core Principles

**Prioritize clarity over cleverness.** Your code should make the transformation logic immediately obvious to a reader.

**Use numpy for array operations.** Python's numpy is more concise and readable for grid manipulations than nested loops.

**Keep functions focused.** Each function should do one clear transformation.

## Function Signature

**Always use this signature:**
```python
def transform(grid_lst: list[list[int]]) -> list[list[int]]:
```

**Convert to numpy immediately:**
```python
def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    grid = np.array(grid_lst)
    # ... your logic here
    return result.tolist()
```

## Import Strategy

**Import numpy as np:**
```python
import numpy as np
```

**Import specific functions when needed:**
```python
from collections import Counter
from scipy.ndimage import label
```

**Avoid star imports.** Never use `from numpy import *`.

## Variable Naming

**Use descriptive, consistent names:**

- `grid` for the numpy array version of input
- `result` or `output` for the final grid
- `rows`, `cols` for dimensions (not `h`, `w`)
- `r`, `c` for row, column indices in loops
- Color variables: `red_color = 2` (not just `red = 2`)

**Example:**
```python
def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    grid = np.array(grid_lst)
    rows, cols = grid.shape
    result = np.zeros_like(grid)
    
    for r in range(rows):
        for c in range(cols):
            # transformation logic
    
    return result.tolist()
```

## Array Operations

**Prefer numpy operations over loops when possible:**

```python
# Good: Use numpy operations
result = np.where(grid == 0, 5, grid)

# Avoid: Nested loops for simple operations  
for r in range(rows):
    for c in range(cols):
        if grid[r][c] == 0:
            result[r][c] = 5
```

**Use list comprehensions for grid copying:**
```python
# For simple copying without numpy
result = [row[:] for row in grid_lst]
```

## Logic Structure

**Handle edge cases first:**
```python
def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst or not grid_lst[0]:
        return grid_lst
    
    grid = np.array(grid_lst)
    # main logic
```

**Use helper functions for complex repeated operations:**
```python
def find_bounding_box(grid, color):
    """Find bounding box of all cells with given color."""
    rows, cols = np.where(grid == color)
    if len(rows) == 0:
        return None
    return rows.min(), rows.max(), cols.min(), cols.max()

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    grid = np.array(grid_lst)
    bbox = find_bounding_box(grid, 5)
    # use bbox
```

**Always check bounds when accessing arrays:**
```python
# Good
if 0 <= new_r < rows and 0 <= new_c < cols:
    result[new_r][new_c] = value

# Avoid
result[new_r][new_c] = value  # might crash
```

## Common Patterns

**For coordinate transformations:**
```python
# Reflection across center
center_r, center_c = rows // 2, cols // 2
new_r = center_r + (center_r - r)
new_c = center_c + (center_c - c)
```

**For finding patterns:**
```python
# Use np.argwhere for finding positions
red_positions = np.argwhere(grid == 2)
for r, c in red_positions:
    # process each red cell
```

**For filling regions:**
```python
# Create masks for conditions
mask = (grid > 0) & (grid < 5)
result[mask] = 9
```

## What NOT to Do

❌ **Don't use magic numbers without explanation:**
```python
# Bad
if grid[i][j] == 7:
    grid[i][j] = 3

# Good  
BACKGROUND = 7
TARGET = 3
if grid[i][j] == BACKGROUND:
    grid[i][j] = TARGET
```

❌ **Don't modify the input grid unless you mean to:**
```python
# Bad - modifies input
def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    for i in range(len(grid_lst)):
        grid_lst[i][0] = 5  # Oops! Modified input
    return grid_lst

# Good - work on copy
def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    grid = np.array(grid_lst)  # Creates copy
    grid[:, 0] = 5
    return grid.tolist()
```

❌ **Don't write overly complex nested loops:**
```python
# Bad - hard to understand
for i in range(len(grid)):
    for j in range(len(grid[0])):
        for k in range(len(grid)):
            for l in range(len(grid[0])):
                if some_complex_condition(i,j,k,l):
                    # nested logic

# Good - break into helper functions
def find_matching_pairs(grid):
    # clear single purpose
    
def transform_pairs(grid, pairs):
    # clear single purpose
```

## Summary

Write code that clearly expresses the transformation you're implementing. Use numpy for array operations, check bounds, use descriptive names, and break complex logic into focused helper functions. The goal is code that another person can read and immediately understand what transformation it performs.