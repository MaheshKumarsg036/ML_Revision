# Logistic Regression Mastery

Logistic regression is the go-to baseline for binary classification. This guide dissects its assumptions, optimization, evaluation, and regularization, with end-to-end code that avoids high-level ML libraries so you can showcase core understanding in interviews.

```python
import numpy as np
from dataclasses import dataclass

rng = np.random.default_rng(seed=7)
```

---

## 1. Concept Recap

Logistic regression models the log-odds of the positive class as a linear function of inputs:

\[
\log \frac{P(y=1 \mid x)}{1 - P(y=1 \mid x)} = w^\top x + b
\]

Predicted probability: `σ(z) = 1 / (1 + exp(-z))`. Decision boundary occurs where `σ(z) ≥ threshold` (default 0.5).

---

## 2. Assumptions

- **Binary or binarized target**: Multiclass uses one-vs-rest or multinomial extensions.
- **Linear decision boundary**: Logit is linear in features (feature engineering/interaction terms may be needed).
- **Independence**: Observations are independent; residuals show no autocorrelation.
- **Low multicollinearity**: Correlated predictors destabilize coefficients; regularization alleviates.
- **Large sample size**: Maximum likelihood estimates benefit from sufficient positive/negative examples.
- **Feature scaling recommended**: Improves convergence and interpretability of regularization.

Violations (e.g., overlapping class distributions, imbalanced classes) necessitate adjustments like resampling, feature engineering, or alternate models.

---

## 3. Loss Function and Metrics

- **Log Loss (Binary Cross-Entropy)**: `-1/n Σ [y log(p) + (1 - y) log(1 - p)]` (optimized by logistic regression).
- **Accuracy**: `(TP + TN) / total`—misleading for imbalanced classes.
- **Precision**: `TP / (TP + FP)`—how many predicted positives are correct.
- **Recall (Sensitivity)**: `TP / (TP + FN)`—how many actual positives were found.
- **F1 Score**: Harmonic mean of precision and recall.
- **ROC-AUC**: Probability the classifier ranks a random positive higher than a negative.
- **PR-AUC**: Better for heavy class imbalance.

Mention threshold selection and how precision-recall trade-offs matter for business decisions.

---

## 4. Gradient Descent Implementation

```python
@dataclass
class LogisticRegressionGD:
    lr: float = 0.1
    n_iter: int = 10_000
    l1: float = 0.0
    l2: float = 0.0
    weights: np.ndarray | None = None

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionGD":
        Xb = np.hstack([np.ones((X.shape[0], 1)), X])
        self.weights = np.zeros(Xb.shape[1])

        for _ in range(self.n_iter):
            logits = Xb @ self.weights
            probs = self._sigmoid(logits)
            errors = probs - y

            gradient = (Xb.T @ errors) / Xb.shape[0]
            gradient[1:] += self.l2 * self.weights[1:]
            gradient[1:] += self.l1 * np.sign(self.weights[1:])

            self.weights -= self.lr * gradient

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise ValueError("Model not fitted")
        logits = np.hstack([np.ones((X.shape[0], 1)), X]) @ self.weights
        return self._sigmoid(logits)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)
```

- Uses batch gradient descent; stochastic or mini-batch variants are preferred for large datasets.
- Applies L1/L2 penalties (same idea as lasso/ridge) to avoid overfitting and manage collinearity.
- Feature scaling (zero mean, unit variance) accelerates convergence.

---

## 5. Synthetic Dataset Demo

```python
n_samples = 300
X = rng.normal(size=(n_samples, 2))
true_w = np.array([1.5, -2.0])
logits = X @ true_w + 0.5
probs = 1 / (1 + np.exp(-logits))
y = (rng.uniform(size=n_samples) < probs).astype(int)

# Standardize features
X_mean, X_std = X.mean(axis=0), X.std(axis=0)
X_scaled = (X - X_mean) / X_std
```

```python
model = LogisticRegressionGD(lr=0.1, n_iter=8_000, l2=0.1)
model.fit(X_scaled, y)
probs = model.predict_proba(X_scaled)
preds = model.predict(X_scaled)

log_loss = -(y * np.log(probs + 1e-9) + (1 - y) * np.log(1 - probs + 1e-9)).mean()
accuracy = (preds == y).mean()
```

Evaluate additional metrics via manual counts:

```python
TP = np.sum((preds == 1) & (y == 1))
FP = np.sum((preds == 1) & (y == 0))
FN = np.sum((preds == 0) & (y == 1))
precision = TP / (TP + FP + 1e-9)
recall = TP / (TP + FN + 1e-9)
f1 = 2 * precision * recall / (precision + recall + 1e-9)
```

For ROC-AUC, compute TPR/FPR at multiple thresholds or implement the trapezoidal rule on sorted probabilities—doing so manually is a great interview talking point.

---

## 6. Regularization Variants

- **Ridge (L2)**: Adds `λ ||w||₂²`; keeps coefficients small, beneficial under multicollinearity.
- **Lasso (L1)**: Adds `λ ||w||₁`; encourages sparsity and feature selection.
- **Elastic Net**: Combines both penalties.

These directly correspond to the `l1` and `l2` parameters in the `LogisticRegressionGD` class. Cross-validate hyperparameters, especially when classes are imbalanced or features numerous.

---

## 7. Decision Boundary Visualization (Optional Snippet)

```python
xx, yy = np.meshgrid(
    np.linspace(X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1, 200),
    np.linspace(X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1, 200),
)
mesh = np.c_[xx.ravel(), yy.ravel()]
Z = model.predict_proba(mesh).reshape(xx.shape)
# Use matplotlib contour plots to visualize (omitted for brevity)
```

Being able to explain how feature scaling shapes decision boundaries is a good interview add-on.

---

## 8. When to Use Logistic Regression

| Situation | Why Logistic Regression Works |
| --------- | ----------------------------- |
| Need interpretable coefficients | Odds ratios (`e^{w_i}`) show multiplicative impact of features. |
| Baseline for binary classification | Fast to train, few hyperparameters, strong baseline metric. |
| High-dimensional sparse data (bag-of-words) | With L1/L2 regularization, handles thousands of features. |
| Online learning context | Stochastic gradient descent scales to streaming data. |

---

## 9. When to Avoid

- Relationship between features and log-odds is highly non-linear (consider tree-based models or kernel methods).
- Severe class imbalance requiring complex thresholding or cost-sensitive learning (logistic still usable but needs adjustments).
- Small datasets with complete separation (one class perfectly separated by a linear boundary) cause coefficient blow-up—use regularization or Bayesian priors.
- Dependent observations (time-series, panel data) without proper handling—consider generalized estimating equations or mixed models.

---

## 10. Interview Checklist

- Derive gradient of log loss: `∂L/∂w = Xᵀ (σ(Xw) - y) / n`.
- Explain odds, log-odds, and how to interpret coefficients in terms of odds ratios.
- Discuss threshold tuning (Youden’s J statistic, maximizing F1, cost-based thresholds).
- Handle class imbalance with weighting (sample weights in loss), resampling, or different metrics (PR-AUC).
- Compare to other classifiers: logistic vs SVM (hinge loss), vs Naive Bayes (probabilistic assumptions), vs trees (non-linear).

Being comfortable with these derivations and trade-offs shows mastery and readiness for applied ML discussions.
