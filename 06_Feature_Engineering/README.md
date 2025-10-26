# Feature Engineering - Data Preparation Techniques

## Overview
Feature engineering is the process of using domain knowledge to extract features from raw data that make machine learning algorithms work better.

## Key Concepts

### 1. Feature Scaling
- **Standardization (Z-score normalization)**: Mean=0, Std=1
- **Min-Max Scaling**: Scale to [0, 1] range
- **Robust Scaling**: Uses median and IQR

### 2. Encoding Categorical Variables
- **Label Encoding**: Convert categories to numbers
- **One-Hot Encoding**: Binary columns for each category
- **Ordinal Encoding**: For ordered categories
- **Target Encoding**: Replace with target mean

### 3. Feature Creation
- Polynomial features
- Interaction features
- Binning/Discretization
- Date/Time features

### 4. Feature Selection
- Filter methods (correlation, chi-square)
- Wrapper methods (RFE)
- Embedded methods (Lasso, tree-based)
- Principal Component Analysis (PCA)

### 5. Handling Missing Values
- Deletion (rows/columns)
- Imputation (mean, median, mode, forward/backward fill)
- Advanced imputation (KNN, iterative)

### 6. Handling Outliers
- Detection (IQR, Z-score, Isolation Forest)
- Treatment (capping, transformation, removal)

## Quick Reference

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Label encoding
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# One-hot encoding
ohe = OneHotEncoder()
X_encoded = ohe.fit_transform(X)

# Imputation
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)
```

## Files in This Directory

- **feature_scaling.py**: Scaling techniques
- **encoding_techniques.py**: Categorical encoding
- **feature_selection.py**: Feature selection methods
- **handling_missing_data.py**: Missing value techniques

---
← [Statistics](../05_Statistics/) | [Machine Learning →](../07_Machine_Learning/)
