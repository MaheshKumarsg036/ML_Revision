# Machine Learning - Algorithms from Scratch

## Overview
This section contains implementations of machine learning algorithms from scratch to understand their inner workings, along with examples using scikit-learn.

## Algorithms Covered

### 1. Supervised Learning

#### Regression
- **Linear Regression**: Predicting continuous values
- **Ridge Regression**: L2 regularization
- **Lasso Regression**: L1 regularization
- **Polynomial Regression**: Non-linear relationships

#### Classification
- **Logistic Regression**: Binary and multi-class classification
- **K-Nearest Neighbors (KNN)**: Instance-based learning
- **Naive Bayes**: Probabilistic classifier
- **Decision Trees**: Tree-based classification
- **Random Forest**: Ensemble of decision trees
- **Support Vector Machines (SVM)**: Maximum margin classifier

### 2. Unsupervised Learning
- **K-Means Clustering**: Partition-based clustering
- **Hierarchical Clustering**: Tree-based clustering
- **DBSCAN**: Density-based clustering
- **Principal Component Analysis (PCA)**: Dimensionality reduction

### 3. Optimization
- **Gradient Descent**: Optimization algorithm
- **Stochastic Gradient Descent (SGD)**: Mini-batch optimization
- **Adam Optimizer**: Adaptive learning rate

## Directory Structure

```
07_Machine_Learning/
├── README.md
├── 01_Linear_Regression/
│   ├── linear_regression_scratch.py
│   └── linear_regression_sklearn.py
├── 02_Logistic_Regression/
│   ├── logistic_regression_scratch.py
│   └── logistic_regression_sklearn.py
├── 03_KNN/
│   └── knn_scratch.py
├── 04_Decision_Trees/
│   └── decision_tree_scratch.py
├── 05_Naive_Bayes/
│   └── naive_bayes_scratch.py
├── 06_SVM/
│   └── svm_examples.py
├── 07_Clustering/
│   ├── kmeans_scratch.py
│   └── clustering_examples.py
└── 08_Dimensionality_Reduction/
    └── pca_scratch.py
```

## Implementation Philosophy

All from-scratch implementations follow these principles:
1. Clear, readable code with comments
2. Use only NumPy for computations (no ML libraries)
3. Include detailed mathematical explanations
4. Provide comparison with scikit-learn implementations
5. Include visualization where applicable

## Quick Reference

### Linear Regression
```python
# From scratch
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

# Scikit-learn
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
```

### Classification
```python
# Logistic Regression
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(X_train, y_train)

# KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
```

### Clustering
```python
# K-Means
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
labels = kmeans.labels_
```

## Resources

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Pattern Recognition and Machine Learning - Bishop](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/)
- [The Elements of Statistical Learning](https://hastie.su.domains/ElemStatLearn/)

---
← [Feature Engineering](../06_Feature_Engineering/)
