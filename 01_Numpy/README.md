# Numpy - Numerical Computing in Python

## Overview
NumPy (Numerical Python) is the fundamental package for numerical computing in Python. It provides support for large, multi-dimensional arrays and matrices, along with a collection of mathematical functions to operate on these arrays efficiently.

## Key Concepts

### 1. Arrays
- **ndarray**: The core data structure in NumPy
- Multi-dimensional, homogeneous array
- Fixed size at creation
- Efficient element-wise operations

### 2. Array Creation
- `np.array()`: Create from lists
- `np.zeros()`, `np.ones()`: Arrays of zeros/ones
- `np.arange()`: Range of values
- `np.linspace()`: Evenly spaced values
- `np.random`: Random number generation

### 3. Array Operations
- Element-wise arithmetic operations
- Broadcasting rules
- Universal functions (ufuncs)
- Aggregation functions (sum, mean, std, etc.)

### 4. Indexing and Slicing
- Basic indexing: `arr[i]`, `arr[i:j]`
- Boolean indexing
- Fancy indexing
- Multi-dimensional indexing

### 5. Array Manipulation
- Reshaping: `reshape()`, `flatten()`, `ravel()`
- Stacking: `hstack()`, `vstack()`, `concatenate()`
- Splitting: `split()`, `hsplit()`, `vsplit()`
- Transposing: `transpose()`, `T`

### 6. Linear Algebra
- Matrix multiplication: `@` operator, `np.dot()`
- Matrix decomposition
- Eigenvalues and eigenvectors
- Solving linear systems

## Files in This Directory

- **numpy_basics.py**: Basic array operations and creation
- **numpy_advanced.py**: Advanced operations and techniques
- **numpy_examples.ipynb**: Interactive Jupyter notebook with examples

## Quick Reference

```python
import numpy as np

# Array creation
arr = np.array([1, 2, 3, 4, 5])
zeros = np.zeros((3, 4))
ones = np.ones((2, 3))
range_arr = np.arange(0, 10, 2)

# Array operations
arr + 5              # Add scalar
arr * 2              # Multiply
arr.mean()           # Mean
arr.std()            # Standard deviation

# Indexing
arr[0]               # First element
arr[1:4]             # Slice
arr[arr > 3]         # Boolean indexing

# Reshaping
arr.reshape(5, 1)    # Reshape to column vector
arr.T                # Transpose

# Linear algebra
A @ B                # Matrix multiplication
np.linalg.inv(A)     # Matrix inverse
np.linalg.eig(A)     # Eigenvalues
```

## Resources

- [Official NumPy Documentation](https://numpy.org/doc/)
- [NumPy User Guide](https://numpy.org/doc/stable/user/index.html)
- [NumPy Cheat Sheet](https://numpy.org/doc/stable/user/absolute_beginners.html)

## Practice Tips

1. Start with basic array creation and manipulation
2. Practice broadcasting to avoid loops
3. Use vectorized operations for better performance
4. Master indexing techniques
5. Understand memory layout (C vs Fortran order)

---
Next Topic: [Pandas →](../02_Pandas/)
