"""
Pandas Basics - DataFrame and Series Operations
===============================================

This module demonstrates fundamental Pandas operations including:
- Creating DataFrames and Series
- Data inspection and selection
- Basic data manipulation
"""

import pandas as pd
import numpy as np


def series_examples():
    """Demonstrate Pandas Series operations."""
    print("=" * 60)
    print("SERIES EXAMPLES")
    print("=" * 60)
    
    # Create Series from list
    s = pd.Series([1, 2, 3, 4, 5])
    print("\nSeries from list:")
    print(s)
    
    # Series with custom index
    s_indexed = pd.Series([10, 20, 30], index=['a', 'b', 'c'])
    print("\nSeries with custom index:")
    print(s_indexed)
    
    # From dictionary
    s_dict = pd.Series({'A': 100, 'B': 200, 'C': 300})
    print("\nSeries from dictionary:")
    print(s_dict)
    
    # Accessing elements
    print(f"\nFirst element: {s[0]}")
    print(f"Element at 'a': {s_indexed['a']}")
    
    # Series operations
    print(f"\nMultiply by 2:\n{s * 2}")
    print(f"\nSum: {s.sum()}")
    print(f"Mean: {s.mean()}")
    print(f"Max: {s.max()}")


def dataframe_creation():
    """Demonstrate DataFrame creation methods."""
    print("\n" + "=" * 60)
    print("DATAFRAME CREATION")
    print("=" * 60)
    
    # From dictionary
    df_dict = pd.DataFrame({
        'Name': ['Alice', 'Bob', 'Charlie', 'David'],
        'Age': [25, 30, 35, 40],
        'City': ['New York', 'London', 'Paris', 'Tokyo']
    })
    print("\nDataFrame from dictionary:")
    print(df_dict)
    
    # From list of lists
    df_list = pd.DataFrame(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        columns=['A', 'B', 'C']
    )
    print("\nDataFrame from list of lists:")
    print(df_list)
    
    # From numpy array
    df_numpy = pd.DataFrame(
        np.random.randn(4, 3),
        columns=['X', 'Y', 'Z']
    )
    print("\nDataFrame from numpy array:")
    print(df_numpy)
    
    # With custom index
    df_indexed = pd.DataFrame(
        {'A': [1, 2, 3], 'B': [4, 5, 6]},
        index=['row1', 'row2', 'row3']
    )
    print("\nDataFrame with custom index:")
    print(df_indexed)


def data_inspection():
    """Demonstrate data inspection methods."""
    print("\n" + "=" * 60)
    print("DATA INSPECTION")
    print("=" * 60)
    
    df = pd.DataFrame({
        'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
        'Age': [25, 30, 35, 40, 28],
        'Salary': [50000, 60000, 70000, 80000, 55000],
        'Department': ['HR', 'IT', 'Finance', 'IT', 'HR']
    })
    
    print("\nDataFrame:")
    print(df)
    
    print("\n--- Basic Info ---")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Index: {df.index.tolist()}")
    print(f"Data types:\n{df.dtypes}")
    
    print("\n--- Head and Tail ---")
    print("First 3 rows:")
    print(df.head(3))
    print("\nLast 2 rows:")
    print(df.tail(2))
    
    print("\n--- Info ---")
    df.info()
    
    print("\n--- Descriptive Statistics ---")
    print(df.describe())
    
    print("\n--- Value Counts ---")
    print("Department counts:")
    print(df['Department'].value_counts())


def data_selection():
    """Demonstrate data selection methods."""
    print("\n" + "=" * 60)
    print("DATA SELECTION")
    print("=" * 60)
    
    df = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [10, 20, 30, 40, 50],
        'C': [100, 200, 300, 400, 500]
    })
    
    print("\nDataFrame:")
    print(df)
    
    # Column selection
    print("\n--- Column Selection ---")
    print("Column 'A':")
    print(df['A'])
    
    print("\nMultiple columns:")
    print(df[['A', 'C']])
    
    # Row selection with loc (label-based)
    print("\n--- Row Selection (loc) ---")
    print("Row 0:")
    print(df.loc[0])
    
    print("\nRows 0 to 2:")
    print(df.loc[0:2])
    
    # Row selection with iloc (position-based)
    print("\n--- Row Selection (iloc) ---")
    print("First row:")
    print(df.iloc[0])
    
    print("\nFirst 3 rows:")
    print(df.iloc[0:3])
    
    # Combined selection
    print("\n--- Combined Selection ---")
    print("Element at row 1, column 'B':")
    print(df.loc[1, 'B'])
    
    print("\nSubset:")
    print(df.loc[0:2, ['A', 'C']])
    
    # Boolean indexing
    print("\n--- Boolean Indexing ---")
    print("Rows where A > 2:")
    print(df[df['A'] > 2])
    
    print("\nRows where B >= 30:")
    print(df[df['B'] >= 30])
    
    print("\nMultiple conditions:")
    print(df[(df['A'] > 2) & (df['B'] < 50)])


def basic_operations():
    """Demonstrate basic DataFrame operations."""
    print("\n" + "=" * 60)
    print("BASIC OPERATIONS")
    print("=" * 60)
    
    df = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [10, 20, 30, 40, 50],
        'C': [100, 200, 300, 400, 500]
    })
    
    print("\nOriginal DataFrame:")
    print(df)
    
    # Adding columns
    print("\n--- Adding Columns ---")
    df['D'] = df['A'] + df['B']
    print("Added column D (A + B):")
    print(df)
    
    # Column arithmetic
    print("\n--- Column Arithmetic ---")
    df['E'] = df['A'] * 2
    print("Added column E (A * 2):")
    print(df)
    
    # Dropping columns
    print("\n--- Dropping Columns ---")
    df_dropped = df.drop(['D', 'E'], axis=1)
    print("After dropping D and E:")
    print(df_dropped)
    
    # Sorting
    print("\n--- Sorting ---")
    df_sorted = df.sort_values('B', ascending=False)
    print("Sorted by B (descending):")
    print(df_sorted)
    
    # Aggregations
    print("\n--- Aggregations ---")
    print(f"Sum of column A: {df['A'].sum()}")
    print(f"Mean of column B: {df['B'].mean()}")
    print(f"Max of column C: {df['C'].max()}")
    print(f"\nSum by column:\n{df.sum()}")


def main():
    """Run all examples."""
    series_examples()
    dataframe_creation()
    data_inspection()
    data_selection()
    basic_operations()


if __name__ == "__main__":
    main()
