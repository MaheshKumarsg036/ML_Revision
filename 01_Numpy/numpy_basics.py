"""
Numpy Basics - Array Creation and Operations
=============================================

This module demonstrates fundamental NumPy operations including:
- Array creation methods
- Basic array operations
- Array attributes
- Indexing and slicing
"""

import numpy as np


def array_creation_examples():
    """Demonstrate various ways to create NumPy arrays."""
    print("=" * 60)
    print("ARRAY CREATION EXAMPLES")
    print("=" * 60)
    
    # From Python lists
    arr_from_list = np.array([1, 2, 3, 4, 5])
    print(f"\nArray from list: {arr_from_list}")
    
    # 2D array
    arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
    print(f"\n2D Array:\n{arr_2d}")
    
    # Arrays of zeros and ones
    zeros = np.zeros((3, 4))
    print(f"\nArray of zeros (3x4):\n{zeros}")
    
    ones = np.ones((2, 3))
    print(f"\nArray of ones (2x3):\n{ones}")
    
    # Range arrays
    range_arr = np.arange(0, 10, 2)
    print(f"\nRange array (0 to 10, step 2): {range_arr}")
    
    # Evenly spaced values
    linspace_arr = np.linspace(0, 1, 5)
    print(f"\nLinspace (0 to 1, 5 values): {linspace_arr}")
    
    # Identity matrix
    identity = np.eye(3)
    print(f"\nIdentity matrix (3x3):\n{identity}")
    
    # Random arrays
    random_arr = np.random.rand(3, 3)
    print(f"\nRandom array (3x3):\n{random_arr}")
    
    random_int = np.random.randint(0, 100, size=(3, 3))
    print(f"\nRandom integers (0-100, 3x3):\n{random_int}")
    

def array_attributes_examples():
    """Demonstrate array attributes."""
    print("\n" + "=" * 60)
    print("ARRAY ATTRIBUTES")
    print("=" * 60)
    
    arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    
    print(f"\nArray:\n{arr}")
    print(f"Shape: {arr.shape}")
    print(f"Number of dimensions: {arr.ndim}")
    print(f"Size (total elements): {arr.size}")
    print(f"Data type: {arr.dtype}")
    print(f"Item size (bytes): {arr.itemsize}")
    print(f"Total bytes: {arr.nbytes}")


def array_operations_examples():
    """Demonstrate basic array operations."""
    print("\n" + "=" * 60)
    print("ARRAY OPERATIONS")
    print("=" * 60)
    
    arr = np.array([1, 2, 3, 4, 5])
    
    print(f"\nOriginal array: {arr}")
    print(f"Add 10: {arr + 10}")
    print(f"Multiply by 2: {arr * 2}")
    print(f"Square: {arr ** 2}")
    print(f"Square root: {np.sqrt(arr)}")
    
    # Array arithmetic
    arr1 = np.array([1, 2, 3])
    arr2 = np.array([4, 5, 6])
    
    print(f"\nArray 1: {arr1}")
    print(f"Array 2: {arr2}")
    print(f"Addition: {arr1 + arr2}")
    print(f"Multiplication: {arr1 * arr2}")
    print(f"Dot product: {np.dot(arr1, arr2)}")


def indexing_slicing_examples():
    """Demonstrate indexing and slicing."""
    print("\n" + "=" * 60)
    print("INDEXING AND SLICING")
    print("=" * 60)
    
    arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    
    print(f"\nOriginal array: {arr}")
    print(f"First element: {arr[0]}")
    print(f"Last element: {arr[-1]}")
    print(f"Elements 2 to 5: {arr[2:6]}")
    print(f"Every 2nd element: {arr[::2]}")
    print(f"Reversed: {arr[::-1]}")
    
    # 2D indexing
    arr_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(f"\n2D Array:\n{arr_2d}")
    print(f"Element at [1, 2]: {arr_2d[1, 2]}")
    print(f"First row: {arr_2d[0, :]}")
    print(f"Second column: {arr_2d[:, 1]}")
    print(f"Subarray:\n{arr_2d[0:2, 1:3]}")
    
    # Boolean indexing
    print(f"\nBoolean indexing (elements > 5): {arr[arr > 5]}")
    print(f"Even numbers: {arr[arr % 2 == 0]}")


def aggregation_examples():
    """Demonstrate aggregation functions."""
    print("\n" + "=" * 60)
    print("AGGREGATION FUNCTIONS")
    print("=" * 60)
    
    arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    
    print(f"\nArray: {arr}")
    print(f"Sum: {np.sum(arr)}")
    print(f"Mean: {np.mean(arr)}")
    print(f"Median: {np.median(arr)}")
    print(f"Standard deviation: {np.std(arr)}")
    print(f"Variance: {np.var(arr)}")
    print(f"Min: {np.min(arr)}")
    print(f"Max: {np.max(arr)}")
    print(f"Argmin (index of min): {np.argmin(arr)}")
    print(f"Argmax (index of max): {np.argmax(arr)}")
    
    # 2D aggregations
    arr_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(f"\n2D Array:\n{arr_2d}")
    print(f"Sum of all elements: {np.sum(arr_2d)}")
    print(f"Sum along axis 0 (columns): {np.sum(arr_2d, axis=0)}")
    print(f"Sum along axis 1 (rows): {np.sum(arr_2d, axis=1)}")


def broadcasting_examples():
    """Demonstrate broadcasting."""
    print("\n" + "=" * 60)
    print("BROADCASTING")
    print("=" * 60)
    
    # Scalar broadcasting
    arr = np.array([[1, 2, 3], [4, 5, 6]])
    print(f"\nArray:\n{arr}")
    print(f"Add 10 (broadcasting):\n{arr + 10}")
    
    # 1D to 2D broadcasting
    arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
    arr_1d = np.array([10, 20, 30])
    print(f"\n2D Array:\n{arr_2d}")
    print(f"1D Array: {arr_1d}")
    print(f"Broadcasting addition:\n{arr_2d + arr_1d}")


def main():
    """Run all examples."""
    array_creation_examples()
    array_attributes_examples()
    array_operations_examples()
    indexing_slicing_examples()
    aggregation_examples()
    broadcasting_examples()


if __name__ == "__main__":
    main()
