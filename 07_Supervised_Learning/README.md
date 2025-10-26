# Supervised Learning Revision Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Regression Algorithms](#regression-algorithms)
3. [Classification Algorithms](#classification-algorithms)
4. [Model Evaluation](#model-evaluation)
5. [Ensemble Methods](#ensemble-methods)
6. [Hyperparameter Tuning](#hyperparameter-tuning)

## Introduction
Supervised learning is a machine learning paradigm where the model learns from labeled training data to make predictions on unseen data.

## Regression Algorithms
Predicting continuous values

### 1. Linear Regression
- **Concept**: Fits a linear relationship between features and target
- **Equation**: y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
- **Use cases**: Price prediction, demand forecasting
- **Assumptions**: Linearity, independence, homoscedasticity, normality

### 2. Polynomial Regression
- **Concept**: Extends linear regression with polynomial terms
- **Equation**: y = β₀ + β₁x + β₂x² + ... + βₙxⁿ

### 3. Ridge Regression (L2 Regularization)
- **Concept**: Adds penalty proportional to square of coefficients
- **Penalty**: λΣβ²

### 4. Lasso Regression (L1 Regularization)
- **Concept**: Adds penalty proportional to absolute value of coefficients
- **Penalty**: λΣ|β|
- **Feature**: Can perform feature selection

### 5. ElasticNet
- **Concept**: Combines L1 and L2 regularization

### 6. Decision Tree Regression
- **Concept**: Uses tree structure for predictions

### 7. Random Forest Regression
- **Concept**: Ensemble of decision trees

### 8. Gradient Boosting Regression
- **Concept**: Sequential ensemble method

### 9. Support Vector Regression (SVR)
- **Concept**: Applies SVM principles to regression

## Classification Algorithms
Predicting discrete categories

### 1. Logistic Regression
- **Concept**: Uses sigmoid function for binary classification
- **Equation**: P(y=1|x) = 1 / (1 + e^(-z))
- **Use cases**: Binary outcomes (yes/no, spam/not spam)

### 2. K-Nearest Neighbors (KNN)
- **Concept**: Classifies based on majority vote of k nearest neighbors
- **Distance metrics**: Euclidean, Manhattan, Minkowski

### 3. Decision Tree Classifier
- **Concept**: Tree structure with decision nodes
- **Metrics**: Gini impurity, Information gain (entropy)

### 4. Random Forest Classifier
- **Concept**: Ensemble of decision trees
- **Advantages**: Reduces overfitting, handles non-linearity

### 5. Support Vector Machine (SVM)
- **Concept**: Finds optimal hyperplane to separate classes
- **Kernel trick**: Linear, RBF, polynomial

### 6. Naive Bayes
- **Concept**: Based on Bayes' theorem with independence assumption
- **Types**: Gaussian, Multinomial, Bernoulli

### 7. Gradient Boosting Classifier
- **Variants**: XGBoost, LightGBM, CatBoost

### 8. Neural Networks
- **Concept**: Multi-layer perceptron

## Model Evaluation

### Regression Metrics
- **Mean Absolute Error (MAE)**: Average absolute difference
- **Mean Squared Error (MSE)**: Average squared difference
- **Root Mean Squared Error (RMSE)**: Square root of MSE
- **R² Score**: Proportion of variance explained
- **Adjusted R²**: R² adjusted for number of features

### Classification Metrics
- **Accuracy**: (TP + TN) / Total
- **Precision**: TP / (TP + FP)
- **Recall (Sensitivity)**: TP / (TP + FN)
- **F1-Score**: Harmonic mean of precision and recall
- **ROC Curve**: True Positive Rate vs False Positive Rate
- **AUC**: Area Under ROC Curve
- **Confusion Matrix**: Visualizes predictions vs actuals

### Cross-Validation
- K-Fold Cross-Validation
- Stratified K-Fold
- Leave-One-Out Cross-Validation (LOOCV)

## Ensemble Methods
Combining multiple models

### Bagging (Bootstrap Aggregating)
- Random Forest
- Reduces variance

### Boosting
- AdaBoost
- Gradient Boosting
- XGBoost, LightGBM, CatBoost
- Reduces bias

### Stacking
- Combines predictions from multiple models

### Voting
- Hard voting: Majority vote
- Soft voting: Average probabilities

## Hyperparameter Tuning

### Grid Search
- Exhaustive search over parameter grid

### Random Search
- Random sampling from parameter distributions

### Bayesian Optimization
- Smart search using probability models

### Key Hyperparameters
- **Tree-based**: max_depth, min_samples_split, n_estimators
- **SVM**: C, kernel, gamma
- **Neural Networks**: learning_rate, hidden_layers, neurons
- **KNN**: n_neighbors, distance_metric
