"""
Pandas Basics - DataFrames and Series Operations
"""

import pandas as pd
import numpy as np

# ============================================================================
# DATA STRUCTURES
# ============================================================================

def data_structures():
    """Demonstrate Pandas data structures"""
    print("=" * 60)
    print("DATA STRUCTURES")
    print("=" * 60)
    
    # Series
    series = pd.Series([1, 2, 3, 4, 5], index=['a', 'b', 'c', 'd', 'e'])
    print("Series:")
    print(series)
    print(f"\nSeries dtype: {series.dtype}")
    
    # DataFrame
    data = {
        'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
        'Age': [25, 30, 35, 28, 32],
        'City': ['New York', 'London', 'Paris', 'Tokyo', 'Sydney'],
        'Salary': [50000, 60000, 75000, 55000, 70000]
    }
    df = pd.DataFrame(data)
    print("\n\nDataFrame:")
    print(df)
    
    return df


# ============================================================================
# DATA INSPECTION
# ============================================================================

def data_inspection(df):
    """Demonstrate data inspection methods"""
    print("\n" + "=" * 60)
    print("DATA INSPECTION")
    print("=" * 60)
    
    print("\nFirst 3 rows:")
    print(df.head(3))
    
    print("\n\nLast 2 rows:")
    print(df.tail(2))
    
    print("\n\nDataFrame Info:")
    df.info()
    
    print("\n\nStatistical Summary:")
    print(df.describe())
    
    print(f"\n\nShape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Data types:\n{df.dtypes}")


# ============================================================================
# DATA SELECTION
# ============================================================================

def data_selection(df):
    """Demonstrate data selection methods"""
    print("\n" + "=" * 60)
    print("DATA SELECTION")
    print("=" * 60)
    
    # Select single column
    print("\nSelect 'Name' column:")
    print(df['Name'])
    
    # Select multiple columns
    print("\n\nSelect 'Name' and 'Age' columns:")
    print(df[['Name', 'Age']])
    
    # Using loc (label-based)
    print("\n\nUsing loc - row 0:")
    print(df.loc[0])
    
    print("\n\nUsing loc - rows 0 to 2, columns 'Name' and 'Age':")
    print(df.loc[0:2, ['Name', 'Age']])
    
    # Using iloc (position-based)
    print("\n\nUsing iloc - first row:")
    print(df.iloc[0])
    
    print("\n\nUsing iloc - first 3 rows, first 2 columns:")
    print(df.iloc[:3, :2])
    
    # Boolean indexing
    print("\n\nPeople with Age > 30:")
    print(df[df['Age'] > 30])
    
    print("\n\nPeople with Age > 28 AND Salary > 60000:")
    print(df[(df['Age'] > 28) & (df['Salary'] > 60000)])


# ============================================================================
# DATA CLEANING
# ============================================================================

def data_cleaning():
    """Demonstrate data cleaning operations"""
    print("\n" + "=" * 60)
    print("DATA CLEANING")
    print("=" * 60)
    
    # Create DataFrame with missing values
    data = {
        'A': [1, 2, np.nan, 4, 5],
        'B': [5, np.nan, np.nan, 8, 9],
        'C': [10, 11, 12, 13, 14]
    }
    df_missing = pd.DataFrame(data)
    print("\nDataFrame with missing values:")
    print(df_missing)
    
    # Check for missing values
    print("\n\nMissing values:")
    print(df_missing.isna())
    
    print("\n\nCount of missing values per column:")
    print(df_missing.isna().sum())
    
    # Fill missing values
    print("\n\nFill missing values with 0:")
    print(df_missing.fillna(0))
    
    print("\n\nFill missing values with mean:")
    print(df_missing.fillna(df_missing.mean()))
    
    # Drop missing values
    print("\n\nDrop rows with any missing values:")
    print(df_missing.dropna())
    
    # Duplicates
    data_dup = {
        'Name': ['Alice', 'Bob', 'Alice', 'Charlie', 'Bob'],
        'Age': [25, 30, 25, 35, 30]
    }
    df_dup = pd.DataFrame(data_dup)
    print("\n\nDataFrame with duplicates:")
    print(df_dup)
    
    print("\n\nRemove duplicates:")
    print(df_dup.drop_duplicates())


# ============================================================================
# DATA TRANSFORMATION
# ============================================================================

def data_transformation(df):
    """Demonstrate data transformation operations"""
    print("\n" + "=" * 60)
    print("DATA TRANSFORMATION")
    print("=" * 60)
    
    df_copy = df.copy()
    
    # Creating new columns
    df_copy['Age_in_10_years'] = df_copy['Age'] + 10
    print("\nAdding new column:")
    print(df_copy[['Name', 'Age', 'Age_in_10_years']])
    
    # Apply function
    df_copy['Salary_Tax'] = df_copy['Salary'].apply(lambda x: x * 0.2)
    print("\n\nApply function to calculate tax:")
    print(df_copy[['Name', 'Salary', 'Salary_Tax']])
    
    # Map function
    city_country = {
        'New York': 'USA',
        'London': 'UK',
        'Paris': 'France',
        'Tokyo': 'Japan',
        'Sydney': 'Australia'
    }
    df_copy['Country'] = df_copy['City'].map(city_country)
    print("\n\nMap cities to countries:")
    print(df_copy[['Name', 'City', 'Country']])
    
    # Sorting
    print("\n\nSort by Age (ascending):")
    print(df_copy[['Name', 'Age']].sort_values('Age'))
    
    print("\n\nSort by Salary (descending):")
    print(df_copy[['Name', 'Salary']].sort_values('Salary', ascending=False))


# ============================================================================
# GROUPING AND AGGREGATION
# ============================================================================

def grouping_aggregation():
    """Demonstrate grouping and aggregation operations"""
    print("\n" + "=" * 60)
    print("GROUPING AND AGGREGATION")
    print("=" * 60)
    
    # Create sample data
    data = {
        'Department': ['IT', 'HR', 'IT', 'HR', 'IT', 'Finance', 'Finance'],
        'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank', 'Grace'],
        'Age': [25, 30, 35, 28, 32, 40, 38],
        'Salary': [50000, 45000, 75000, 48000, 70000, 80000, 75000]
    }
    df = pd.DataFrame(data)
    print("\nSample data:")
    print(df)
    
    # Group by and aggregate
    print("\n\nAverage salary by department:")
    print(df.groupby('Department')['Salary'].mean())
    
    print("\n\nMultiple aggregations:")
    print(df.groupby('Department')['Salary'].agg(['mean', 'min', 'max', 'count']))
    
    print("\n\nGroup by with multiple columns:")
    print(df.groupby('Department').agg({
        'Age': 'mean',
        'Salary': ['mean', 'sum']
    }))


# ============================================================================
# MERGING AND JOINING
# ============================================================================

def merging_joining():
    """Demonstrate merging and joining operations"""
    print("\n" + "=" * 60)
    print("MERGING AND JOINING")
    print("=" * 60)
    
    # Create sample DataFrames
    df1 = pd.DataFrame({
        'ID': [1, 2, 3, 4],
        'Name': ['Alice', 'Bob', 'Charlie', 'David']
    })
    
    df2 = pd.DataFrame({
        'ID': [1, 2, 3, 5],
        'Salary': [50000, 60000, 70000, 55000]
    })
    
    print("\nDataFrame 1:")
    print(df1)
    print("\nDataFrame 2:")
    print(df2)
    
    # Inner join
    print("\n\nInner join:")
    print(pd.merge(df1, df2, on='ID', how='inner'))
    
    # Left join
    print("\n\nLeft join:")
    print(pd.merge(df1, df2, on='ID', how='left'))
    
    # Right join
    print("\n\nRight join:")
    print(pd.merge(df1, df2, on='ID', how='right'))
    
    # Outer join
    print("\n\nOuter join:")
    print(pd.merge(df1, df2, on='ID', how='outer'))
    
    # Concatenation
    df3 = pd.DataFrame({
        'Name': ['Eve', 'Frank'],
        'Age': [32, 40]
    })
    
    df4 = pd.DataFrame({
        'Name': ['Grace', 'Henry'],
        'Age': [28, 35]
    })
    
    print("\n\nConcatenate vertically:")
    print(pd.concat([df3, df4], ignore_index=True))


# ============================================================================
# TIME SERIES
# ============================================================================

def time_series():
    """Demonstrate time series operations"""
    print("\n" + "=" * 60)
    print("TIME SERIES")
    print("=" * 60)
    
    # Create date range
    dates = pd.date_range('2024-01-01', periods=10, freq='D')
    print("\nDate range:")
    print(dates)
    
    # Create time series DataFrame
    df = pd.DataFrame({
        'Date': dates,
        'Value': np.random.randint(10, 100, 10)
    })
    print("\n\nTime series DataFrame:")
    print(df)
    
    # Set date as index
    df.set_index('Date', inplace=True)
    print("\n\nWith date as index:")
    print(df)
    
    # Extract date components
    df['Year'] = df.index.year
    df['Month'] = df.index.month
    df['Day'] = df.index.day
    df['DayOfWeek'] = df.index.dayofweek
    print("\n\nWith extracted date components:")
    print(df)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    df = data_structures()
    data_inspection(df)
    data_selection(df)
    data_cleaning()
    data_transformation(df)
    grouping_aggregation()
    merging_joining()
    time_series()
    
    print("\n" + "=" * 60)
    print("Pandas basics demonstration complete!")
    print("=" * 60)
