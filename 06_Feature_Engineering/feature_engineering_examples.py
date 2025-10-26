"""
Feature Engineering - Scaling and Encoding Techniques
=====================================================

This module demonstrates feature engineering techniques.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer


def feature_scaling_examples():
    """Demonstrate feature scaling techniques."""
    print("=" * 60)
    print("FEATURE SCALING")
    print("=" * 60)
    
    # Sample data
    data = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    
    print("\nOriginal data:")
    print(data)
    
    # Standardization
    scaler = StandardScaler()
    standardized = scaler.fit_transform(data)
    print("\nStandardized (StandardScaler):")
    print(standardized)
    print(f"Mean: {standardized.mean(axis=0)}")
    print(f"Std: {standardized.std(axis=0)}")
    
    # Min-Max Scaling
    minmax_scaler = MinMaxScaler()
    minmax_scaled = minmax_scaler.fit_transform(data)
    print("\nMin-Max Scaled:")
    print(minmax_scaled)
    print(f"Min: {minmax_scaled.min(axis=0)}")
    print(f"Max: {minmax_scaled.max(axis=0)}")
    
    # Robust Scaling
    robust_scaler = RobustScaler()
    robust_scaled = robust_scaler.fit_transform(data)
    print("\nRobust Scaled:")
    print(robust_scaled)


def encoding_examples():
    """Demonstrate categorical encoding techniques."""
    print("\n" + "=" * 60)
    print("CATEGORICAL ENCODING")
    print("=" * 60)
    
    # Label Encoding
    categories = np.array(['cat', 'dog', 'bird', 'cat', 'dog'])
    
    le = LabelEncoder()
    encoded = le.fit_transform(categories)
    print("\nOriginal categories:")
    print(categories)
    print("\nLabel Encoded:")
    print(encoded)
    print(f"Classes: {le.classes_}")
    
    # One-Hot Encoding
    categories_2d = categories.reshape(-1, 1)
    ohe = OneHotEncoder(sparse_output=False)
    onehot = ohe.fit_transform(categories_2d)
    print("\nOne-Hot Encoded:")
    print(onehot)
    print(f"Categories: {ohe.categories_}")


def missing_data_handling():
    """Demonstrate handling missing data."""
    print("\n" + "=" * 60)
    print("HANDLING MISSING DATA")
    print("=" * 60)
    
    # Sample data with missing values
    data = np.array([[1, 2], [3, np.nan], [np.nan, 6], [7, 8]])
    
    print("\nOriginal data with missing values:")
    print(data)
    
    # Mean imputation
    imputer_mean = SimpleImputer(strategy='mean')
    imputed_mean = imputer_mean.fit_transform(data)
    print("\nMean Imputation:")
    print(imputed_mean)
    
    # Median imputation
    imputer_median = SimpleImputer(strategy='median')
    imputed_median = imputer_median.fit_transform(data)
    print("\nMedian Imputation:")
    print(imputed_median)
    
    # Most frequent imputation
    data_categorical = np.array([['A'], ['B'], [np.nan], ['A'], ['B']])
    imputer_freq = SimpleImputer(strategy='most_frequent')
    imputed_freq = imputer_freq.fit_transform(data_categorical)
    print("\nMost Frequent Imputation (categorical):")
    print(imputed_freq.ravel())


def feature_creation_examples():
    """Demonstrate feature creation."""
    print("\n" + "=" * 60)
    print("FEATURE CREATION")
    print("=" * 60)
    
    # Polynomial features
    from sklearn.preprocessing import PolynomialFeatures
    
    X = np.array([[1, 2], [3, 4], [5, 6]])
    print("\nOriginal features:")
    print(X)
    
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)
    print("\nPolynomial features (degree 2):")
    print(X_poly)
    print(f"Feature names: {poly.get_feature_names_out(['x1', 'x2'])}")
    
    # Binning
    ages = np.array([5, 15, 25, 35, 45, 55, 65, 75])
    bins = [0, 18, 35, 60, 100]
    labels = ['Child', 'Young Adult', 'Adult', 'Senior']
    
    age_groups = pd.cut(ages, bins=bins, labels=labels)
    print("\nBinning example (ages):")
    print(f"Ages: {ages}")
    print(f"Age groups: {age_groups.tolist()}")


def outlier_detection():
    """Demonstrate outlier detection."""
    print("\n" + "=" * 60)
    print("OUTLIER DETECTION")
    print("=" * 60)
    
    np.random.seed(42)
    data = np.random.randn(100)
    # Add some outliers
    data = np.append(data, [5, 6, -5, -6])
    
    # Z-score method
    z_scores = np.abs((data - data.mean()) / data.std())
    outliers_z = np.where(z_scores > 3)[0]
    print(f"\nOutliers detected using Z-score (>3): {len(outliers_z)}")
    
    # IQR method
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers_iqr = np.where((data < lower_bound) | (data > upper_bound))[0]
    print(f"Outliers detected using IQR method: {len(outliers_iqr)}")
    print(f"Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")


def main():
    """Run all examples."""
    feature_scaling_examples()
    encoding_examples()
    missing_data_handling()
    feature_creation_examples()
    outlier_detection()


if __name__ == "__main__":
    main()
