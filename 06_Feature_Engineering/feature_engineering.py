"""
Feature Engineering - Data Preprocessing and Feature Creation
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import (StandardScaler, MinMaxScaler, RobustScaler,
                                   LabelEncoder, OneHotEncoder)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import (SelectKBest, chi2, f_classif, RFE,
                                        VarianceThreshold)
from sklearn.ensemble import RandomForestClassifier

# ============================================================================
# HANDLING MISSING DATA
# ============================================================================

def handling_missing_data():
    """Demonstrate missing data handling techniques"""
    print("=" * 60)
    print("HANDLING MISSING DATA")
    print("=" * 60)
    
    # Create data with missing values
    data = pd.DataFrame({
        'A': [1, 2, np.nan, 4, 5, 6, np.nan, 8],
        'B': [10, np.nan, 30, 40, np.nan, 60, 70, 80],
        'C': [100, 200, 300, 400, 500, 600, 700, 800]
    })
    
    print("\nOriginal Data:")
    print(data)
    print(f"\nMissing values:\n{data.isna().sum()}")
    
    # Method 1: Drop missing values
    print("\n1. Drop rows with missing values:")
    print(data.dropna())
    
    # Method 2: Mean imputation
    print("\n2. Mean Imputation:")
    imputer = SimpleImputer(strategy='mean')
    data_mean = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
    print(data_mean)
    
    # Method 3: Median imputation
    print("\n3. Median Imputation:")
    imputer = SimpleImputer(strategy='median')
    data_median = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
    print(data_median)
    
    # Method 4: Forward fill
    print("\n4. Forward Fill:")
    print(data.fillna(method='ffill'))
    
    # Method 5: KNN imputation
    print("\n5. KNN Imputation:")
    imputer = KNNImputer(n_neighbors=2)
    data_knn = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
    print(data_knn)


# ============================================================================
# ENCODING CATEGORICAL VARIABLES
# ============================================================================

def encoding_categorical():
    """Demonstrate categorical encoding techniques"""
    print("\n" + "=" * 60)
    print("ENCODING CATEGORICAL VARIABLES")
    print("=" * 60)
    
    # Sample data
    data = pd.DataFrame({
        'Color': ['Red', 'Blue', 'Green', 'Red', 'Blue', 'Green'],
        'Size': ['Small', 'Medium', 'Large', 'Medium', 'Small', 'Large'],
        'Price': [10, 20, 30, 15, 12, 35]
    })
    
    print("\nOriginal Data:")
    print(data)
    
    # Method 1: Label Encoding
    print("\n1. Label Encoding:")
    label_encoder = LabelEncoder()
    data_label = data.copy()
    data_label['Color_Encoded'] = label_encoder.fit_transform(data['Color'])
    print(data_label[['Color', 'Color_Encoded']])
    
    # Method 2: One-Hot Encoding
    print("\n2. One-Hot Encoding:")
    data_onehot = pd.get_dummies(data, columns=['Color', 'Size'], prefix=['Color', 'Size'])
    print(data_onehot)
    
    # Method 3: Ordinal Encoding (for ordered categories)
    print("\n3. Ordinal Encoding:")
    size_mapping = {'Small': 1, 'Medium': 2, 'Large': 3}
    data_ordinal = data.copy()
    data_ordinal['Size_Ordinal'] = data['Size'].map(size_mapping)
    print(data_ordinal[['Size', 'Size_Ordinal']])
    
    # Method 4: Frequency Encoding
    print("\n4. Frequency Encoding:")
    freq = data['Color'].value_counts(normalize=True)
    data_freq = data.copy()
    data_freq['Color_Freq'] = data['Color'].map(freq)
    print(data_freq[['Color', 'Color_Freq']])


# ============================================================================
# FEATURE SCALING
# ============================================================================

def feature_scaling():
    """Demonstrate feature scaling techniques"""
    print("\n" + "=" * 60)
    print("FEATURE SCALING")
    print("=" * 60)
    
    # Sample data
    data = pd.DataFrame({
        'Feature1': [1, 2, 3, 4, 5, 100],
        'Feature2': [10, 20, 30, 40, 50, 1000],
        'Feature3': [100, 200, 300, 400, 500, 10000]
    })
    
    print("\nOriginal Data:")
    print(data)
    print(f"\nStatistics:\n{data.describe()}")
    
    # Method 1: Standardization (Z-score)
    print("\n1. Standardization (Z-score normalization):")
    scaler = StandardScaler()
    data_standardized = pd.DataFrame(
        scaler.fit_transform(data),
        columns=data.columns
    )
    print(data_standardized)
    print(f"\nMean:\n{data_standardized.mean()}")
    print(f"Std:\n{data_standardized.std()}")
    
    # Method 2: Min-Max Scaling
    print("\n2. Min-Max Normalization:")
    scaler = MinMaxScaler()
    data_minmax = pd.DataFrame(
        scaler.fit_transform(data),
        columns=data.columns
    )
    print(data_minmax)
    print(f"\nMin:\n{data_minmax.min()}")
    print(f"Max:\n{data_minmax.max()}")
    
    # Method 3: Robust Scaling
    print("\n3. Robust Scaling:")
    scaler = RobustScaler()
    data_robust = pd.DataFrame(
        scaler.fit_transform(data),
        columns=data.columns
    )
    print(data_robust)


# ============================================================================
# FEATURE CREATION
# ============================================================================

def feature_creation():
    """Demonstrate feature creation techniques"""
    print("\n" + "=" * 60)
    print("FEATURE CREATION")
    print("=" * 60)
    
    # Sample data
    data = pd.DataFrame({
        'x1': [1, 2, 3, 4, 5],
        'x2': [2, 4, 6, 8, 10],
        'date': pd.date_range('2024-01-01', periods=5, freq='D')
    })
    
    print("\nOriginal Data:")
    print(data)
    
    # 1. Polynomial features
    print("\n1. Polynomial Features:")
    data['x1_squared'] = data['x1'] ** 2
    data['x1_cubed'] = data['x1'] ** 3
    print(data[['x1', 'x1_squared', 'x1_cubed']])
    
    # 2. Interaction features
    print("\n2. Interaction Features:")
    data['x1_x2_interaction'] = data['x1'] * data['x2']
    print(data[['x1', 'x2', 'x1_x2_interaction']])
    
    # 3. Binning
    print("\n3. Binning (Discretization):")
    data['x1_binned'] = pd.cut(data['x1'], bins=3, labels=['Low', 'Medium', 'High'])
    print(data[['x1', 'x1_binned']])
    
    # 4. Date features
    print("\n4. Date/Time Features:")
    data['year'] = data['date'].dt.year
    data['month'] = data['date'].dt.month
    data['day'] = data['date'].dt.day
    data['dayofweek'] = data['date'].dt.dayofweek
    print(data[['date', 'year', 'month', 'day', 'dayofweek']])


# ============================================================================
# FEATURE SELECTION
# ============================================================================

def feature_selection():
    """Demonstrate feature selection techniques"""
    print("\n" + "=" * 60)
    print("FEATURE SELECTION")
    print("=" * 60)
    
    # Generate sample data
    np.random.seed(42)
    X = np.random.randn(100, 10)
    y = (X[:, 0] + X[:, 1] * 2 + np.random.randn(100) * 0.1) > 0
    y = y.astype(int)
    
    feature_names = [f'Feature_{i}' for i in range(10)]
    X_df = pd.DataFrame(X, columns=feature_names)
    
    print(f"\nOriginal shape: {X_df.shape}")
    print(f"Features: {list(X_df.columns)}")
    
    # Method 1: Variance Threshold
    print("\n1. Variance Threshold:")
    selector = VarianceThreshold(threshold=0.5)
    X_var = selector.fit_transform(X)
    selected_features = X_df.columns[selector.get_support()]
    print(f"   Selected {X_var.shape[1]} features: {list(selected_features)}")
    
    # Method 2: SelectKBest with f_classif
    print("\n2. SelectKBest (f_classif):")
    selector = SelectKBest(f_classif, k=5)
    X_kbest = selector.fit_transform(X, y)
    selected_features = X_df.columns[selector.get_support()]
    print(f"   Selected top 5 features: {list(selected_features)}")
    print(f"   Scores: {selector.scores_[:5]}")
    
    # Method 3: Recursive Feature Elimination
    print("\n3. Recursive Feature Elimination (RFE):")
    estimator = RandomForestClassifier(n_estimators=10, random_state=42)
    selector = RFE(estimator, n_features_to_select=5)
    X_rfe = selector.fit_transform(X, y)
    selected_features = X_df.columns[selector.get_support()]
    print(f"   Selected 5 features: {list(selected_features)}")


# ============================================================================
# HANDLING OUTLIERS
# ============================================================================

def handling_outliers():
    """Demonstrate outlier handling techniques"""
    print("\n" + "=" * 60)
    print("HANDLING OUTLIERS")
    print("=" * 60)
    
    # Generate data with outliers
    np.random.seed(42)
    data = pd.DataFrame({
        'value': np.concatenate([
            np.random.randn(95) * 10 + 50,
            [150, 200, -50, -100, 180]  # outliers
        ])
    })
    
    print(f"\nOriginal Data Statistics:")
    print(data.describe())
    
    # Method 1: Z-score method
    print("\n1. Z-score Method (|z| > 3):")
    z_scores = np.abs((data['value'] - data['value'].mean()) / data['value'].std())
    outliers_zscore = data[z_scores > 3]
    print(f"   Number of outliers: {len(outliers_zscore)}")
    print(f"   Outlier values: {outliers_zscore['value'].values}")
    
    # Method 2: IQR method
    print("\n2. IQR Method:")
    Q1 = data['value'].quantile(0.25)
    Q3 = data['value'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers_iqr = data[(data['value'] < lower_bound) | (data['value'] > upper_bound)]
    print(f"   IQR: {IQR:.2f}")
    print(f"   Lower bound: {lower_bound:.2f}")
    print(f"   Upper bound: {upper_bound:.2f}")
    print(f"   Number of outliers: {len(outliers_iqr)}")
    
    # Method 3: Capping (Winsorization)
    print("\n3. Capping/Winsorization:")
    data_capped = data.copy()
    data_capped['value_capped'] = data['value'].clip(lower=lower_bound, upper=upper_bound)
    print(f"   Original min: {data['value'].min():.2f}")
    print(f"   Capped min: {data_capped['value_capped'].min():.2f}")
    print(f"   Original max: {data['value'].max():.2f}")
    print(f"   Capped max: {data_capped['value_capped'].max():.2f}")


# ============================================================================
# TEXT FEATURE ENGINEERING
# ============================================================================

def text_feature_engineering():
    """Demonstrate text feature engineering"""
    print("\n" + "=" * 60)
    print("TEXT FEATURE ENGINEERING")
    print("=" * 60)
    
    # Sample text data
    texts = [
        "Machine learning is amazing",
        "Deep learning is a subset of machine learning",
        "Natural language processing is fun",
        "Data science requires statistics"
    ]
    
    df = pd.DataFrame({'text': texts})
    
    print("\nOriginal Texts:")
    for i, text in enumerate(texts, 1):
        print(f"{i}. {text}")
    
    # Basic text features
    print("\n1. Basic Text Features:")
    df['word_count'] = df['text'].apply(lambda x: len(x.split()))
    df['char_count'] = df['text'].apply(lambda x: len(x))
    df['avg_word_length'] = df['text'].apply(lambda x: np.mean([len(word) for word in x.split()]))
    print(df[['text', 'word_count', 'char_count', 'avg_word_length']])
    
    # Note: For more advanced text features like TF-IDF, we would need sklearn's TfidfVectorizer
    # which is demonstrated in the machine learning sections


# ============================================================================
# TIME SERIES FEATURES
# ============================================================================

def time_series_features():
    """Demonstrate time series feature creation"""
    print("\n" + "=" * 60)
    print("TIME SERIES FEATURES")
    print("=" * 60)
    
    # Create time series data
    dates = pd.date_range('2024-01-01', periods=30, freq='D')
    data = pd.DataFrame({
        'date': dates,
        'value': np.random.randn(30).cumsum() + 100
    })
    
    print("\nOriginal Time Series Data (first 5 rows):")
    print(data.head())
    
    # Extract date features
    print("\n1. Date/Time Features:")
    data['year'] = data['date'].dt.year
    data['month'] = data['date'].dt.month
    data['day'] = data['date'].dt.day
    data['dayofweek'] = data['date'].dt.dayofweek
    data['is_weekend'] = data['dayofweek'].isin([5, 6]).astype(int)
    print(data[['date', 'year', 'month', 'day', 'dayofweek', 'is_weekend']].head())
    
    # Lag features
    print("\n2. Lag Features:")
    data['value_lag1'] = data['value'].shift(1)
    data['value_lag2'] = data['value'].shift(2)
    print(data[['date', 'value', 'value_lag1', 'value_lag2']].head(5))
    
    # Rolling statistics
    print("\n3. Rolling Statistics:")
    data['rolling_mean_3'] = data['value'].rolling(window=3).mean()
    data['rolling_std_3'] = data['value'].rolling(window=3).std()
    print(data[['date', 'value', 'rolling_mean_3', 'rolling_std_3']].head(5))
    
    # Difference features
    print("\n4. Difference Features:")
    data['value_diff'] = data['value'].diff()
    print(data[['date', 'value', 'value_diff']].head(5))


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    handling_missing_data()
    encoding_categorical()
    feature_scaling()
    feature_creation()
    feature_selection()
    handling_outliers()
    text_feature_engineering()
    time_series_features()
    
    print("\n" + "=" * 60)
    print("Feature engineering demonstration complete!")
    print("=" * 60)
