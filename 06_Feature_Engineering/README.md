# Feature Engineering Revision Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Handling Missing Data](#handling-missing-data)
3. [Encoding Categorical Variables](#encoding-categorical-variables)
4. [Feature Scaling](#feature-scaling)
5. [Feature Creation](#feature-creation)
6. [Feature Selection](#feature-selection)
7. [Handling Outliers](#handling-outliers)
8. [Text Feature Engineering](#text-feature-engineering)
9. [Time Series Features](#time-series-features)

## Introduction
Feature engineering is the process of creating new features or transforming existing features to improve machine learning model performance.

## Handling Missing Data
### Strategies:
- **Deletion**: Remove rows or columns
  - Listwise deletion
  - Pairwise deletion
- **Imputation**: Fill missing values
  - Mean/Median/Mode imputation
  - Forward fill / Backward fill
  - K-Nearest Neighbors imputation
  - Interpolation
  - Model-based imputation

## Encoding Categorical Variables
### Techniques:
- **Label Encoding**: Convert categories to integers
- **One-Hot Encoding**: Create binary columns for each category
- **Ordinal Encoding**: For ordered categories
- **Target Encoding**: Replace category with target mean
- **Binary Encoding**: Encode as binary digits
- **Frequency Encoding**: Replace with category frequency
- **Hash Encoding**: Use hashing for high cardinality

## Feature Scaling
### Methods:
- **Standardization (Z-score normalization)**
  - Mean = 0, Std = 1
  - Formula: (x - μ) / σ
- **Min-Max Normalization**
  - Scale to range [0, 1]
  - Formula: (x - min) / (max - min)
- **Robust Scaling**
  - Uses median and IQR
  - Less sensitive to outliers
- **Log Transformation**
  - For skewed data

## Feature Creation
### Techniques:
- **Polynomial Features**: x², x³, x*y
- **Interaction Features**: Combinations of features
- **Domain-specific Features**: Based on problem knowledge
- **Binning/Discretization**: Convert continuous to categorical
- **Aggregations**: Sum, mean, count, etc.
- **Date/Time Features**: Extract year, month, day, hour, etc.

## Feature Selection
### Methods:
- **Filter Methods**
  - Correlation coefficient
  - Chi-square test
  - Mutual information
  - Variance threshold
- **Wrapper Methods**
  - Forward selection
  - Backward elimination
  - Recursive Feature Elimination (RFE)
- **Embedded Methods**
  - Lasso (L1 regularization)
  - Ridge (L2 regularization)
  - Tree-based feature importance

## Handling Outliers
### Detection Methods:
- Z-score method
- IQR method
- Isolation Forest
- Local Outlier Factor (LOF)

### Treatment:
- Remove outliers
- Cap/floor (winsorization)
- Transform (log, sqrt)
- Treat separately

## Text Feature Engineering
### Techniques:
- **Bag of Words (BoW)**
- **TF-IDF**: Term Frequency-Inverse Document Frequency
- **Word Embeddings**: Word2Vec, GloVe
- **N-grams**: Bigrams, trigrams
- **Character-level features**
- **Sentiment scores**
- **Text length, word count**

## Time Series Features
### Time-based Features:
- Year, month, day, hour, minute
- Day of week, day of year
- Quarter, semester
- Is weekend, is holiday
- Time since event
- Cyclic encoding (sin/cos for cyclical features)

### Lag Features:
- Previous values (lag-1, lag-2, etc.)
- Rolling statistics (mean, std, min, max)
- Exponential moving averages
- Difference features (first difference, seasonal difference)
