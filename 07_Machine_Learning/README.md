# Machine Learning - Building Intelligent Systems 🤖

Machine Learning is the science of getting computers to learn and act like humans do, improving their learning over time through data and experience.

## 📚 Topics Covered

### 1. Supervised Learning

#### Regression Algorithms
- **Linear Regression**
  - Simple linear regression
  - Multiple linear regression
  - Polynomial regression
  - Ridge regression (L2)
  - Lasso regression (L1)
  - Elastic Net

- **Tree-based Regression**
  - Decision tree regressor
  - Random forest regressor
  - Gradient boosting regressor
  - XGBoost, LightGBM, CatBoost

#### Classification Algorithms
- **Logistic Regression**
  - Binary classification
  - Multiclass classification
  - Regularization

- **Support Vector Machines (SVM)**
  - Linear SVM
  - Kernel SVM (RBF, polynomial)
  - Multi-class SVM

- **K-Nearest Neighbors (KNN)**
  - Distance metrics
  - Choosing K
  - Weighted KNN

- **Naive Bayes**
  - Gaussian Naive Bayes
  - Multinomial Naive Bayes
  - Bernoulli Naive Bayes

- **Tree-based Classification**
  - Decision trees
  - Random forests
  - Gradient boosting
  - AdaBoost

### 2. Unsupervised Learning

#### Clustering
- **K-Means Clustering**
  - Elbow method
  - Silhouette analysis
  - K-Means++

- **Hierarchical Clustering**
  - Agglomerative clustering
  - Dendrograms
  - Linkage methods

- **Density-based Clustering**
  - DBSCAN
  - OPTICS

- **Other Methods**
  - Gaussian Mixture Models
  - Mean Shift
  - Spectral Clustering

#### Dimensionality Reduction
- Principal Component Analysis (PCA)
- Linear Discriminant Analysis (LDA)
- t-SNE
- UMAP

### 3. Model Evaluation

#### Regression Metrics
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R-squared (R²)
- Adjusted R-squared

#### Classification Metrics
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix
- Classification Report

#### Cross-Validation
- K-Fold Cross-Validation
- Stratified K-Fold
- Leave-One-Out (LOO)
- Time Series Split

### 4. Model Selection and Tuning

- **Hyperparameter Tuning**
  - Grid Search
  - Random Search
  - Bayesian Optimization

- **Model Selection**
  - Train-test split
  - Validation strategies
  - Ensemble methods

### 5. Important Concepts

- **Bias-Variance Tradeoff**
- **Overfitting and Underfitting**
- **Regularization**
- **Feature Importance**
- **Model Interpretability**
- **Pipeline Creation**

## 🎯 Learning Objectives

After completing this section, you will be able to:
- Implement various ML algorithms from scratch and using scikit-learn
- Choose appropriate algorithms for different problems
- Evaluate model performance correctly
- Tune hyperparameters effectively
- Avoid common pitfalls in ML
- Build end-to-end ML pipelines

## 📖 Resources

- **Official Documentation:** [scikit-learn.org](https://scikit-learn.org/)
- **Notebooks in this folder:**
  - `supervised_learning.ipynb` - Comprehensive guide to supervised learning
  - `unsupervised_learning.ipynb` - Comprehensive guide to unsupervised learning

## 💡 Quick Examples

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))
```

## 🔗 Project Ideas

1. **Iris Flower Classification** - Classic classification problem
2. **House Price Prediction** - Regression problem
3. **Customer Segmentation** - Clustering problem
4. **Spam Email Detection** - Text classification
5. **Credit Card Fraud Detection** - Imbalanced classification

---

**Congratulations on completing the ML Revision journey! 🎉**
