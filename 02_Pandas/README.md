# Pandas Revision Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Data Structures](#data-structures)
3. [Data Loading](#data-loading)
4. [Data Inspection](#data-inspection)
5. [Data Selection](#data-selection)
6. [Data Cleaning](#data-cleaning)
7. [Data Transformation](#data-transformation)
8. [Grouping and Aggregation](#grouping-and-aggregation)
9. [Merging and Joining](#merging-and-joining)
10. [Time Series](#time-series)

## Introduction
Pandas is a powerful data manipulation library built on top of NumPy. It provides:
- DataFrame and Series data structures
- Data alignment and handling of missing data
- Data manipulation and transformation tools
- Data aggregation and grouping
- Time series functionality

## Data Structures
- **Series**: 1-dimensional labeled array
- **DataFrame**: 2-dimensional labeled data structure

## Data Loading
- Reading CSV files: `pd.read_csv()`
- Reading Excel files: `pd.read_excel()`
- Reading JSON files: `pd.read_json()`
- Reading from databases: `pd.read_sql()`

## Data Inspection
- `head()`, `tail()`: View first/last rows
- `info()`: Get DataFrame information
- `describe()`: Statistical summary
- `shape`, `columns`, `dtypes`: Structural information

## Data Selection
- Selecting columns: `df['column']`, `df[['col1', 'col2']]`
- Using `.loc[]` for label-based indexing
- Using `.iloc[]` for position-based indexing
- Boolean indexing

## Data Cleaning
- Handling missing values: `isna()`, `fillna()`, `dropna()`
- Removing duplicates: `duplicated()`, `drop_duplicates()`
- Data type conversion: `astype()`
- String operations

## Data Transformation
- Creating new columns
- Applying functions: `apply()`, `map()`, `applymap()`
- Sorting: `sort_values()`, `sort_index()`
- Reshaping: `pivot()`, `melt()`, `stack()`, `unstack()`

## Grouping and Aggregation
- `groupby()`: Group data
- Aggregation functions: `sum()`, `mean()`, `count()`, etc.
- Multiple aggregations: `agg()`
- Transform and filter operations

## Merging and Joining
- `concat()`: Concatenate DataFrames
- `merge()`: SQL-style joins
- `join()`: Join on index

## Time Series
- Date parsing and indexing
- Resampling and frequency conversion
- Rolling windows
- Time shifts
