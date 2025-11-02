"""
Numpy Advanced - Advanced Operations and Techniques
===================================================

This module demonstrates advanced NumPy operations including:
- Array reshaping and manipulation
- Linear algebra operations
- Advanced indexing
- Vectorization
"""

import numpy as np


def reshaping_examples():
    """Demonstrate array reshaping operations."""
    print("=" * 60)
    print("ARRAY RESHAPING")
    print("=" * 60)
    
    arr = np.arange(12)
    print(f"\nOriginal array: {arr}")
    print(f"Shape: {arr.shape}")
    
    # Reshape
    reshaped = arr.reshape(3, 4)
    print(f"\nReshaped to (3, 4):\n{reshaped}")
    
    reshaped_2 = arr.reshape(2, 6)
    print(f"\nReshaped to (2, 6):\n{reshaped_2}")
    
    # Flatten
    flattened = reshaped.flatten()
    print(f"\nFlattened: {flattened}")
    
    # Ravel (returns view when possible)
    raveled = reshaped.ravel()
    print(f"Raveled: {raveled}")
    
    # Transpose
    print(f"\nTransposed:\n{reshaped.T}")
    
    # Add dimension
    expanded = np.expand_dims(arr, axis=0)
    print(f"\nExpanded dims: {expanded.shape}")
    
    # Squeeze
    squeezed = np.squeeze(expanded)
    print(f"Squeezed: {squeezed.shape}")


def stacking_splitting_examples():
    """Demonstrate stacking and splitting arrays."""
    print("\n" + "=" * 60)
    print("STACKING AND SPLITTING")
    print("=" * 60)
    
    arr1 = np.array([[1, 2], [3, 4]])
    arr2 = np.array([[5, 6], [7, 8]])
    
    print(f"\nArray 1:\n{arr1}")
    print(f"\nArray 2:\n{arr2}")
    
    # Vertical stack
    vstack = np.vstack((arr1, arr2))
    print(f"\nVertical stack:\n{vstack}")
    
    # Horizontal stack
    hstack = np.hstack((arr1, arr2))
    print(f"\nHorizontal stack:\n{hstack}")
    
    # Concatenate
    concat_axis0 = np.concatenate((arr1, arr2), axis=0)
    print(f"\nConcatenate (axis=0):\n{concat_axis0}")
    
    concat_axis1 = np.concatenate((arr1, arr2), axis=1)
    print(f"\nConcatenate (axis=1):\n{concat_axis1}")
    
    # Split
    arr = np.arange(12).reshape(3, 4)
    print(f"\nArray to split:\n{arr}")
    
    split_result = np.split(arr, 3, axis=0)
    print(f"\nSplit into 3 parts (axis=0):")
    for i, part in enumerate(split_result):
        print(f"Part {i}:\n{part}")


def linear_algebra_examples():
    """Demonstrate linear algebra operations."""
    print("\n" + "=" * 60)
    print("LINEAR ALGEBRA")
    print("=" * 60)
    
    # Matrix multiplication
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    
    print(f"\nMatrix A:\n{A}")
    print(f"\nMatrix B:\n{B}")
    
    # Dot product
    dot_product = np.dot(A, B)
    print(f"\nDot product (A · B):\n{dot_product}")
    
    # Matrix multiplication using @ operator
    matmul = A @ B
    print(f"\nMatrix multiplication (A @ B):\n{matmul}")
    
    # Matrix properties
    print(f"\nDeterminant of A: {np.linalg.det(A):.2f}")
    print(f"Trace of A: {np.trace(A)}")
    print(f"Rank of A: {np.linalg.matrix_rank(A)}")
    
    # Inverse
    try:
        inv_A = np.linalg.inv(A)
        print(f"\nInverse of A:\n{inv_A}")
        print(f"\nVerify: A @ inv(A):\n{A @ inv_A}")
    except np.linalg.LinAlgError:
        print("\nMatrix is singular, cannot compute inverse")
    
    # Eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(A)
    print(f"\nEigenvalues: {eigenvalues}")
    print(f"Eigenvectors:\n{eigenvectors}")
    
    # Solving linear systems: Ax = b
    b = np.array([1, 2])
    x = np.linalg.solve(A, b)
    print(f"\nSolving Ax = b where b = {b}")
    print(f"Solution x: {x}")
    print(f"Verification (Ax): {A @ x}")


def fancy_indexing_examples():
    """Demonstrate fancy indexing."""
    print("\n" + "=" * 60)
    print("FANCY INDEXING")
    print("=" * 60)
    
    arr = np.arange(20).reshape(4, 5)
    print(f"\nArray:\n{arr}")
    
    # Index with array of indices
    rows = np.array([0, 2, 3])
    print(f"\nRows [0, 2, 3]:\n{arr[rows]}")
    
    # Multiple arrays for indexing
    rows = np.array([0, 1, 2])
    cols = np.array([1, 3, 4])
    print(f"\nElements at (0,1), (1,3), (2,4): {arr[rows, cols]}")
    
    # Boolean mask
    mask = arr > 10
    print(f"\nBoolean mask (> 10):\n{mask}")
    print(f"Elements > 10: {arr[mask]}")
    
    # Combining boolean and fancy indexing
    even_rows = arr[::2]
    print(f"\nEven rows:\n{even_rows}")


def vectorization_examples():
    """Demonstrate vectorization benefits."""
    print("\n" + "=" * 60)
    print("VECTORIZATION")
    print("=" * 60)
    
    # Example: Compute distance between points
    points1 = np.array([[1, 2], [3, 4], [5, 6]])
    points2 = np.array([[7, 8], [9, 10], [11, 12]])
    
    print(f"\nPoints 1:\n{points1}")
    print(f"\nPoints 2:\n{points2}")
    
    # Vectorized distance calculation
    distances = np.sqrt(np.sum((points1 - points2) ** 2, axis=1))
    print(f"\nDistances: {distances}")
    
    # Universal functions
    arr = np.array([1, 4, 9, 16, 25])
    print(f"\nArray: {arr}")
    print(f"Square root: {np.sqrt(arr)}")
    print(f"Exponential: {np.exp(arr[:3])}")  # Limited for readability
    print(f"Log: {np.log(arr)}")
    print(f"Sin: {np.sin(arr)}")
    
    # Where function
    arr = np.array([1, 2, 3, 4, 5, 6])
    result = np.where(arr > 3, arr * 2, arr)
    print(f"\nArray: {arr}")
    print(f"Where > 3, multiply by 2, else keep: {result}")


def random_examples():
    """Demonstrate random number generation."""
    print("\n" + "=" * 60)
    print("RANDOM NUMBER GENERATION")
    print("=" * 60)
    
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Uniform distribution
    uniform = np.random.rand(5)
    print(f"\nUniform [0, 1): {uniform}")
    
    # Normal distribution
    normal = np.random.randn(5)
    print(f"Normal (mean=0, std=1): {normal}")
    
    # Random integers
    integers = np.random.randint(1, 100, size=10)
    print(f"Random integers [1, 100): {integers}")
    
    # Random choice
    arr = np.array([1, 2, 3, 4, 5])
    choice = np.random.choice(arr, size=3, replace=False)
    print(f"\nRandom choice (no replacement): {choice}")
    
    # Shuffle
    arr = np.array([1, 2, 3, 4, 5])
    np.random.shuffle(arr)
    print(f"Shuffled: {arr}")
    
    # Random permutation
    perm = np.random.permutation(10)
    print(f"Permutation: {perm}")


def main():
    """Run all examples."""
    reshaping_examples()
    stacking_splitting_examples()
    linear_algebra_examples()
    fancy_indexing_examples()
    vectorization_examples()
    random_examples()


if __name__ == "__main__":
    main()
