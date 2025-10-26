"""
Supervised Learning with Scikit-Learn
Regression and Classification Examples
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error,
                            accuracy_score, precision_score, recall_score, f1_score,
                            confusion_matrix, classification_report, roc_auc_score)

# Regression models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR

# Classification models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# Datasets
from sklearn.datasets import make_regression, make_classification

# ============================================================================
# REGRESSION EXAMPLES
# ============================================================================

def regression_examples():
    """Demonstrate regression algorithms"""
    print("=" * 60)
    print("REGRESSION EXAMPLES")
    print("=" * 60)
    
    # Generate regression dataset
    X, y = make_regression(n_samples=1000, n_features=10, noise=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Dictionary of models
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=1.0),
        'ElasticNet': ElasticNet(alpha=1.0, l1_ratio=0.5),
        'Decision Tree': DecisionTreeRegressor(max_depth=5, random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42),
        'SVR': SVR(kernel='rbf', C=1.0)
    }
    
    print("\nModel Performance Comparison:")
    print("-" * 80)
    print(f"{'Model':<20} {'MAE':<12} {'MSE':<12} {'RMSE':<12} {'R²':<12}")
    print("-" * 80)
    
    results = []
    for name, model in models.items():
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = model.predict(X_test_scaled)
        
        # Metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        results.append({
            'Model': name,
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R²': r2
        })
        
        print(f"{name:<20} {mae:<12.2f} {mse:<12.2f} {rmse:<12.2f} {r2:<12.4f}")
    
    return pd.DataFrame(results)


# ============================================================================
# CLASSIFICATION EXAMPLES
# ============================================================================

def classification_examples():
    """Demonstrate classification algorithms"""
    print("\n" + "=" * 60)
    print("CLASSIFICATION EXAMPLES")
    print("=" * 60)
    
    # Generate classification dataset
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=15,
        n_redundant=5, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Dictionary of models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42),
        'SVM': SVC(kernel='rbf', C=1.0, probability=True, random_state=42),
        'Naive Bayes': GaussianNB()
    }
    
    print("\nModel Performance Comparison:")
    print("-" * 90)
    print(f"{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 90)
    
    results = []
    for name, model in models.items():
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = model.predict(X_test_scaled)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        results.append({
            'Model': name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        })
        
        print(f"{name:<20} {accuracy:<12.4f} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f}")
    
    # Show confusion matrix for best model
    print("\n\nConfusion Matrix (Random Forest):")
    rf_model = models['Random Forest']
    y_pred_rf = rf_model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred_rf)
    print(cm)
    
    print("\n\nClassification Report (Random Forest):")
    print(classification_report(y_test, y_pred_rf))
    
    return pd.DataFrame(results)


# ============================================================================
# CROSS-VALIDATION
# ============================================================================

def cross_validation_example():
    """Demonstrate cross-validation"""
    print("\n" + "=" * 60)
    print("CROSS-VALIDATION EXAMPLE")
    print("=" * 60)
    
    # Generate data
    X, y = make_classification(n_samples=500, n_features=20, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Models to compare
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', random_state=42)
    }
    
    print("\nK-Fold Cross-Validation (k=5):")
    print("-" * 60)
    print(f"{'Model':<20} {'Mean Accuracy':<15} {'Std Dev':<15}")
    print("-" * 60)
    
    for name, model in models.items():
        # Perform 5-fold cross-validation
        scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
        mean_score = scores.mean()
        std_score = scores.std()
        
        print(f"{name:<20} {mean_score:<15.4f} {std_score:<15.4f}")


# ============================================================================
# HYPERPARAMETER TUNING
# ============================================================================

def hyperparameter_tuning():
    """Demonstrate hyperparameter tuning with Grid Search"""
    print("\n" + "=" * 60)
    print("HYPERPARAMETER TUNING (GRID SEARCH)")
    print("=" * 60)
    
    # Generate data
    X, y = make_classification(n_samples=500, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Random Forest hyperparameter tuning
    print("\nTuning Random Forest Classifier...")
    
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5, 10]
    }
    
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        rf, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=0
    )
    
    grid_search.fit(X_train_scaled, y_train)
    
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    # Test on test set
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Test set accuracy: {test_accuracy:.4f}")


# ============================================================================
# FEATURE IMPORTANCE
# ============================================================================

def feature_importance_example():
    """Demonstrate feature importance"""
    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE")
    print("=" * 60)
    
    # Generate data with feature names
    X, y = make_classification(n_samples=500, n_features=10, n_informative=5, random_state=42)
    feature_names = [f'Feature_{i}' for i in range(10)]
    
    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    # Get feature importances
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("\nTop 5 Most Important Features:")
    print("-" * 40)
    for i in range(5):
        print(f"{i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Run all examples
    regression_results = regression_examples()
    classification_results = classification_examples()
    cross_validation_example()
    hyperparameter_tuning()
    feature_importance_example()
    
    print("\n" + "=" * 60)
    print("Supervised learning examples complete!")
    print("=" * 60)
