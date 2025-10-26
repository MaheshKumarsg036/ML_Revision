"""
Pandas Data Manipulation - Cleaning and Transformation
======================================================

This module demonstrates data manipulation operations including:
- Handling missing values
- Data cleaning
- Grouping and aggregation
- Merging and joining
"""

import pandas as pd
import numpy as np


def missing_data_examples():
    """Demonstrate handling missing data."""
    print("=" * 60)
    print("HANDLING MISSING DATA")
    print("=" * 60)
    
    df = pd.DataFrame({
        'A': [1, 2, np.nan, 4, 5],
        'B': [10, np.nan, 30, np.nan, 50],
        'C': [100, 200, 300, 400, 500]
    })
    
    print("\nDataFrame with missing values:")
    print(df)
    
    print("\n--- Detecting Missing Values ---")
    print("Is null:")
    print(df.isnull())
    
    print("\nMissing count per column:")
    print(df.isnull().sum())
    
    print("\n--- Filling Missing Values ---")
    print("Fill with 0:")
    print(df.fillna(0))
    
    print("\nForward fill:")
    print(df.fillna(method='ffill'))
    
    print("\nFill with column mean:")
    print(df.fillna(df.mean()))
    
    print("\n--- Dropping Missing Values ---")
    print("Drop rows with any NaN:")
    print(df.dropna())
    
    print("\nDrop columns with any NaN:")
    print(df.dropna(axis=1))


def groupby_examples():
    """Demonstrate groupby operations."""
    print("\n" + "=" * 60)
    print("GROUPBY OPERATIONS")
    print("=" * 60)
    
    df = pd.DataFrame({
        'Category': ['A', 'B', 'A', 'B', 'A', 'B'],
        'Value1': [10, 20, 30, 40, 50, 60],
        'Value2': [1, 2, 3, 4, 5, 6]
    })
    
    print("\nDataFrame:")
    print(df)
    
    print("\n--- Basic Grouping ---")
    print("Group by Category and sum:")
    print(df.groupby('Category').sum())
    
    print("\nGroup by Category and mean:")
    print(df.groupby('Category').mean())
    
    print("\n--- Multiple Aggregations ---")
    print("Multiple agg functions:")
    print(df.groupby('Category').agg({
        'Value1': ['sum', 'mean', 'max'],
        'Value2': ['min', 'max']
    }))


def merging_joining_examples():
    """Demonstrate merging and joining DataFrames."""
    print("\n" + "=" * 60)
    print("MERGING AND JOINING")
    print("=" * 60)
    
    df1 = pd.DataFrame({
        'key': ['A', 'B', 'C'],
        'value1': [1, 2, 3]
    })
    
    df2 = pd.DataFrame({
        'key': ['B', 'C', 'D'],
        'value2': [4, 5, 6]
    })
    
    print("\nDataFrame 1:")
    print(df1)
    print("\nDataFrame 2:")
    print(df2)
    
    print("\n--- Inner Join ---")
    print(pd.merge(df1, df2, on='key', how='inner'))
    
    print("\n--- Left Join ---")
    print(pd.merge(df1, df2, on='key', how='left'))
    
    print("\n--- Right Join ---")
    print(pd.merge(df1, df2, on='key', how='right'))
    
    print("\n--- Outer Join ---")
    print(pd.merge(df1, df2, on='key', how='outer'))


def main():
    """Run all examples."""
    missing_data_examples()
    groupby_examples()
    merging_joining_examples()


if __name__ == "__main__":
    main()
