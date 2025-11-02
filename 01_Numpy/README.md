# Numpy - Numerical Python 🔢

NumPy is the fundamental package for scientific computing in Python. It provides support for large, multi-dimensional arrays and matrices, along with mathematical functions to operate on these arrays.

## 📚 Topics Covered

### 1. Array Basics
- Creating arrays (np.array, np.zeros, np.ones, np.arange, np.linspace)
- Array attributes (shape, size, dtype, ndim)
- Array indexing and slicing
- Reshaping arrays

### 2. Array Operations
- Element-wise operations
- Mathematical functions (sum, mean, std, min, max)
- Universal functions (ufuncs)
- Broadcasting rules

### 3. Linear Algebra
- Matrix multiplication
- Transpose
- Inverse
- Determinant
- Eigenvalues and eigenvectors

### 4. Random Module
- Random number generation
- Random distributions
- Random sampling

### 5. Advanced Topics
- Vectorization
- Memory optimization
- Structured arrays
- Masked arrays

## 🎯 Learning Objectives

After completing this section, you will be able to:
- Create and manipulate NumPy arrays efficiently
- Perform mathematical and statistical operations
- Understand broadcasting and vectorization
- Apply linear algebra operations
- Generate random data for simulations

## 📖 Resources

- **Official Documentation:** [numpy.org](https://numpy.org/)
- **Notebooks in this folder:**
  - `numpy_basics.ipynb` - Comprehensive NumPy tutorial

## 💡 Quick Examples

```python
import numpy as np

# Create array
arr = np.array([1, 2, 3, 4, 5])

# Array operations
print(arr * 2)  # [2, 4, 6, 8, 10]
print(np.mean(arr))  # 3.0

# Matrix operations
matrix = np.array([[1, 2], [3, 4]])
print(np.linalg.det(matrix))  # -2.0
```

## 🔗 Next Steps

After mastering NumPy, move on to **Pandas** for data manipulation!
