# Pandas - Data Manipulation and Analysis

## Overview
Pandas is a powerful data manipulation and analysis library for Python. It provides data structures like DataFrame and Series that make it easy to work with structured data.

## Key Concepts

### 1. Data Structures
- **Series**: One-dimensional labeled array
- **DataFrame**: Two-dimensional labeled data structure (like a table)
- Index and columns for labeling

### 2. Data Loading
- Read from various formats: CSV, Excel, JSON, SQL, etc.
- `pd.read_csv()`, `pd.read_excel()`, `pd.read_json()`
- Write data back: `to_csv()`, `to_excel()`, etc.

### 3. Data Selection
- Column selection: `df['column']`, `df[['col1', 'col2']]`
- Row selection: `df.loc[]` (label-based), `df.iloc[]` (position-based)
- Boolean indexing: `df[df['column'] > value]`

### 4. Data Cleaning
- Handling missing values: `isnull()`, `fillna()`, `dropna()`
- Removing duplicates: `drop_duplicates()`
- Data type conversion: `astype()`
- String operations: `str` accessor

### 5. Data Transformation
- Adding/removing columns
- Sorting: `sort_values()`, `sort_index()`
- Filtering and querying
- Apply functions: `apply()`, `map()`, `applymap()`

### 6. Grouping and Aggregation
- GroupBy operations: `groupby()`
- Aggregation functions: `sum()`, `mean()`, `count()`, etc.
- Multi-level grouping
- Pivot tables: `pivot_table()`

### 7. Merging and Joining
- Concatenation: `pd.concat()`
- Merging: `pd.merge()`
- Joining: `df.join()`
- Different join types: inner, outer, left, right

### 8. Time Series
- DateTime index
- Resampling and frequency conversion
- Time-based indexing
- Rolling windows

## Files in This Directory

- **pandas_basics.py**: Basic DataFrame and Series operations
- **pandas_data_manipulation.py**: Data cleaning and transformation
- **pandas_advanced.py**: Advanced operations and techniques

## Quick Reference

```python
import pandas as pd

# Create DataFrame
df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6]
})

# Read/Write data
df = pd.read_csv('file.csv')
df.to_csv('output.csv', index=False)

# Data inspection
df.head()            # First 5 rows
df.info()            # Column info
df.describe()        # Statistics
df.shape             # Dimensions

# Selection
df['column']         # Select column
df[['col1', 'col2']] # Multiple columns
df.loc[0]            # Row by label
df.iloc[0]           # Row by position

# Filtering
df[df['A'] > 2]      # Boolean indexing
df.query('A > 2')    # Query method

# Grouping
df.groupby('A').mean()
df.groupby(['A', 'B']).agg({'C': 'sum', 'D': 'mean'})

# Missing values
df.isnull().sum()    # Count nulls
df.fillna(0)         # Fill with value
df.dropna()          # Drop rows with nulls
```

## Resources

- [Official Pandas Documentation](https://pandas.pydata.org/docs/)
- [Pandas User Guide](https://pandas.pydata.org/docs/user_guide/index.html)
- [10 Minutes to Pandas](https://pandas.pydata.org/docs/user_guide/10min.html)

## Practice Tips

1. Master DataFrame creation from different sources
2. Practice selection and filtering techniques
3. Understand the difference between `loc` and `iloc`
4. Learn to handle missing data effectively
5. Master groupby operations for aggregations
6. Practice merging and joining datasets

---
← [Numpy](../01_Numpy/) | [Matplotlib →](../03_Matplotlib/)
