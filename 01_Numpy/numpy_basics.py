"""
Numpy Basics - Array Creation and Operations
"""

import numpy as np

# ============================================================================
# ARRAY CREATION
# ============================================================================

def array_creation_examples():
    """Demonstrate different ways to create NumPy arrays"""
    print("=" * 60)
    print("ARRAY CREATION")
    print("=" * 60)
    
    # From Python lists
    arr1 = np.array([1, 2, 3, 4, 5])
    print(f"1D Array from list: {arr1}")
    
    arr2 = np.array([[1, 2, 3], [4, 5, 6]])
    print(f"\n2D Array from nested list:\n{arr2}")
    
    # Using zeros, ones, and empty
    zeros = np.zeros((3, 4))
    print(f"\nZeros array (3x4):\n{zeros}")
    
    ones = np.ones((2, 3))
    print(f"\nOnes array (2x3):\n{ones}")
    
    # Using arange and linspace
    range_arr = np.arange(0, 10, 2)
    print(f"\nArray using arange(0, 10, 2): {range_arr}")
    
    linspace_arr = np.linspace(0, 1, 5)
    print(f"\nArray using linspace(0, 1, 5): {linspace_arr}")
    
    # Identity matrix
    identity = np.eye(4)
    print(f"\nIdentity matrix (4x4):\n{identity}")
    
    # Random arrays
    random_arr = np.random.rand(3, 3)
    print(f"\nRandom array (3x3):\n{random_arr}")


# ============================================================================
# ARRAY ATTRIBUTES
# ============================================================================

def array_attributes():
    """Demonstrate NumPy array attributes"""
    print("\n" + "=" * 60)
    print("ARRAY ATTRIBUTES")
    print("=" * 60)
    
    arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    
    print(f"Array:\n{arr}")
    print(f"\nShape: {arr.shape}")
    print(f"Dimensions: {arr.ndim}")
    print(f"Size (total elements): {arr.size}")
    print(f"Data type: {arr.dtype}")
    print(f"Item size (bytes): {arr.itemsize}")


# ============================================================================
# ARRAY OPERATIONS
# ============================================================================

def array_operations():
    """Demonstrate basic array operations"""
    print("\n" + "=" * 60)
    print("ARRAY OPERATIONS")
    print("=" * 60)
    
    arr1 = np.array([1, 2, 3, 4])
    arr2 = np.array([10, 20, 30, 40])
    
    print(f"Array 1: {arr1}")
    print(f"Array 2: {arr2}")
    
    # Arithmetic operations
    print(f"\nAddition: {arr1 + arr2}")
    print(f"Subtraction: {arr1 - arr2}")
    print(f"Multiplication: {arr1 * arr2}")
    print(f"Division: {arr2 / arr1}")
    print(f"Power: {arr1 ** 2}")
    
    # Scalar operations
    print(f"\nScalar multiplication: {arr1 * 5}")
    print(f"Scalar addition: {arr1 + 10}")


# ============================================================================
# INDEXING AND SLICING
# ============================================================================

def indexing_slicing():
    """Demonstrate indexing and slicing"""
    print("\n" + "=" * 60)
    print("INDEXING AND SLICING")
    print("=" * 60)
    
    arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    print(f"Array:\n{arr}")
    
    # Basic indexing
    print(f"\nElement at [0, 0]: {arr[0, 0]}")
    print(f"Element at [1, 2]: {arr[1, 2]}")
    
    # Slicing
    print(f"\nFirst row: {arr[0, :]}")
    print(f"First column: {arr[:, 0]}")
    print(f"First 2 rows, first 3 columns:\n{arr[:2, :3]}")
    
    # Boolean indexing
    print(f"\nElements greater than 5: {arr[arr > 5]}")
    
    # Fancy indexing
    indices = [0, 2]
    print(f"\nRows 0 and 2:\n{arr[indices, :]}")


# ============================================================================
# ARRAY MANIPULATION
# ============================================================================

def array_manipulation():
    """Demonstrate array manipulation"""
    print("\n" + "=" * 60)
    print("ARRAY MANIPULATION")
    print("=" * 60)
    
    arr = np.arange(12)
    print(f"Original array: {arr}")
    
    # Reshaping
    reshaped = arr.reshape(3, 4)
    print(f"\nReshaped to 3x4:\n{reshaped}")
    
    # Flattening
    flattened = reshaped.flatten()
    print(f"\nFlattened: {flattened}")
    
    # Transpose
    print(f"\nTransposed:\n{reshaped.T}")
    
    # Stacking
    arr1 = np.array([[1, 2], [3, 4]])
    arr2 = np.array([[5, 6], [7, 8]])
    
    vstacked = np.vstack((arr1, arr2))
    print(f"\nVertically stacked:\n{vstacked}")
    
    hstacked = np.hstack((arr1, arr2))
    print(f"\nHorizontally stacked:\n{hstacked}")


# ============================================================================
# MATHEMATICAL FUNCTIONS
# ============================================================================

def mathematical_functions():
    """Demonstrate mathematical functions"""
    print("\n" + "=" * 60)
    print("MATHEMATICAL FUNCTIONS")
    print("=" * 60)
    
    arr = np.array([1, 2, 3, 4, 5])
    print(f"Array: {arr}")
    
    # Statistical functions
    print(f"\nSum: {np.sum(arr)}")
    print(f"Mean: {np.mean(arr)}")
    print(f"Median: {np.median(arr)}")
    print(f"Standard deviation: {np.std(arr)}")
    print(f"Variance: {np.var(arr)}")
    print(f"Min: {np.min(arr)}")
    print(f"Max: {np.max(arr)}")
    
    # Trigonometric functions
    angles = np.array([0, np.pi/2, np.pi])
    print(f"\nAngles: {angles}")
    print(f"Sin: {np.sin(angles)}")
    print(f"Cos: {np.cos(angles)}")
    
    # Exponential and logarithmic
    print(f"\nExponential: {np.exp([1, 2, 3])}")
    print(f"Log: {np.log([1, 2.71828, 7.389])}")


# ============================================================================
# BROADCASTING
# ============================================================================

def broadcasting_examples():
    """Demonstrate broadcasting"""
    print("\n" + "=" * 60)
    print("BROADCASTING")
    print("=" * 60)
    
    # Broadcasting with scalar
    arr = np.array([[1, 2, 3], [4, 5, 6]])
    print(f"Array:\n{arr}")
    print(f"\nArray + 10:\n{arr + 10}")
    
    # Broadcasting with 1D array
    row = np.array([10, 20, 30])
    print(f"\nBroadcasting row {row}:")
    print(f"Result:\n{arr + row}")
    
    # Broadcasting with column
    col = np.array([[100], [200]])
    print(f"\nBroadcasting column:\n{col}")
    print(f"Result:\n{arr + col}")


# ============================================================================
# LINEAR ALGEBRA
# ============================================================================

def linear_algebra():
    """Demonstrate linear algebra operations"""
    print("\n" + "=" * 60)
    print("LINEAR ALGEBRA")
    print("=" * 60)
    
    # Matrix multiplication
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    
    print(f"Matrix A:\n{A}")
    print(f"\nMatrix B:\n{B}")
    
    # Dot product
    print(f"\nA @ B (matrix multiplication):\n{A @ B}")
    print(f"np.dot(A, B):\n{np.dot(A, B)}")
    
    # Transpose
    print(f"\nA transpose:\n{A.T}")
    
    # Determinant
    det_A = np.linalg.det(A)
    print(f"\nDeterminant of A: {det_A}")
    
    # Inverse
    inv_A = np.linalg.inv(A)
    print(f"\nInverse of A:\n{inv_A}")
    
    # Eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(A)
    print(f"\nEigenvalues: {eigenvalues}")
    print(f"Eigenvectors:\n{eigenvectors}")


# ============================================================================
# RANDOM MODULE
# ============================================================================

def random_module():
    """Demonstrate random number generation"""
    print("\n" + "=" * 60)
    print("RANDOM MODULE")
    print("=" * 60)
    
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Random floats
    print(f"Random floats (0 to 1): {np.random.rand(5)}")
    
    # Random integers
    print(f"\nRandom integers (1 to 10): {np.random.randint(1, 10, 5)}")
    
    # Random from normal distribution
    print(f"\nRandom from normal distribution: {np.random.randn(5)}")
    
    # Random choice
    arr = np.array([1, 2, 3, 4, 5])
    print(f"\nRandom choice from {arr}: {np.random.choice(arr, 3)}")
    
    # Shuffle
    shuffle_arr = np.array([1, 2, 3, 4, 5])
    np.random.shuffle(shuffle_arr)
    print(f"\nShuffled array: {shuffle_arr}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    array_creation_examples()
    array_attributes()
    array_operations()
    indexing_slicing()
    array_manipulation()
    mathematical_functions()
    broadcasting_examples()
    linear_algebra()
    random_module()
    
    print("\n" + "=" * 60)
    print("Numpy basics demonstration complete!")
    print("=" * 60)
