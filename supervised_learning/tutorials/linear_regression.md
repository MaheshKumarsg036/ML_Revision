# Linear Regression Mastery

In this guide we revisit linear regression from first principles, cover popular variants (simple, polynomial, ridge, lasso, elastic net), and outline when each is appropriate. All code examples avoid high-level ML libraries so you can demonstrate the underlying math in interviews.

```python
import numpy as np
from dataclasses import dataclass

rng = np.random.default_rng(seed=42)
```

---

## 1. Concept Recap

Linear regression models the relationship between predictors `X` and a continuous target `y` as a weighted sum plus an intercept: `y ≈ Xw + b`. Variants differ in feature engineering (polynomial terms) and regularization (penalty terms on `w`).

---

## 2. Core Assumptions

- **Linearity**: Expected target is a linear combination of features (after transformations such as polynomial expansion).
- **Independence**: Observations and residuals are independent.
- **Homoscedasticity**: Constant variance of residuals across levels of predictors.
- **Normality**: Residuals follow a normal distribution (primarily matters for inference, not prediction).
- **No multicollinearity**: Predictors are not highly correlated (regularization can mitigate but not fully remove issues).
- **Exogeneity**: Predictors are measured without error and uncorrelated with the residual term.

Violating assumptions leads to biased estimates, inflated variance, or misleading inference. Regularization can help with multicollinearity but not with bad feature-target relationships.

---

## 3. Evaluation Metrics

| Metric | Formula | Interpretation |
| ------ | ------- | -------------- |
| Mean Squared Error (MSE) | `1/n Σ (y - y_hat)^2` | Penalizes large errors; sensitive to outliers. |
| Root Mean Squared Error (RMSE) | `sqrt(MSE)` | Same units as target; popular for model comparison. |
| Mean Absolute Error (MAE) | `1/n Σ |y - y_hat|` | Robust to outliers; median-based analogue. |
| R-squared (R²) | `1 - SS_res / SS_tot` | Fraction of variance explained; beware of inflated values with many features. |
| Adjusted R² | `1 - (1 - R²)*(n-1)/(n-p-1)` | Penalizes number of predictors `p`; use for model comparison. |

Regularized models often focus on cross-validated RMSE or MAE instead of raw R² due to bias introduced by penalties.

---

## 4. Plain Linear Regression (Normal Equation)

### Algorithm

1. Prepare design matrix `X` (include bias term).
2. Solve normal equation `w = (XᵀX)^(-1) Xᵀ y`.
3. Compute predictions `y_hat = X w`.

### Code

```python
def add_bias(X: np.ndarray) -> np.ndarray:
    return np.hstack([np.ones((X.shape[0], 1)), X])

@dataclass
class NormalEquationRegressor:
    weights: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NormalEquationRegressor":
        Xb = add_bias(X)
        XtX = Xb.T @ Xb
        self.weights = np.linalg.pinv(XtX) @ Xb.T @ y
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise ValueError("Model not fitted")
        return add_bias(X) @ self.weights
```

Use `np.linalg.pinv` instead of inverse for numerical stability. Avoid this closed-form when `X` has many columns (costly) or is ill-conditioned.

---

## 5. Gradient Descent Implementation (Reusable Base)

```python
@dataclass
class GradientDescentRegressor:
    lr: float = 0.01
    n_iter: int = 1_000
    l1: float = 0.0  # lasso coefficient
    l2: float = 0.0  # ridge coefficient
    weights: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientDescentRegressor":
        Xb = add_bias(X)
        n_samples, n_features = Xb.shape
        self.weights = np.zeros(n_features)

        for _ in range(self.n_iter):
            preds = Xb @ self.weights
            residuals = preds - y

            gradient = (Xb.T @ residuals) / n_samples
            gradient[1:] += self.l2 * self.weights[1:]  # ridge term (no bias)
            gradient[1:] += self.l1 * np.sign(self.weights[1:])  # lasso term

            self.weights -= self.lr * gradient

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise ValueError("Model not fitted")
        return add_bias(X) @ self.weights
```

- Set `l1=0, l2=0` for ordinary least squares with gradient descent.
- Lasso penalty uses subgradient (`sign`)—this simple loop works for educational demos but coordinate descent is preferred in production.
- Feature scaling is critical; always standardize inputs before applying regularization.

---

## 6. Simple Linear Regression Example

```python
X = rng.normal(size=(100, 1))
y = 3.5 * X[:, 0] + 2.0 + rng.normal(scale=0.5, size=100)

model = GradientDescentRegressor(lr=0.05, n_iter=5_000)
model.fit(X, y)
predictions = model.predict(X)

mse = np.mean((y - predictions) ** 2)
r2 = 1 - ((y - predictions) ** 2).sum() / ((y - y.mean()) ** 2).sum()
print(model.weights, mse, r2)
```

**When to use**: Relationship is roughly linear, interpretability matters, dataset is of modest size.

**When to avoid**: Strong non-linearity, heteroscedasticity, heavy outliers, or suspect feature-target relationship.

---

## 7. Polynomial Regression

Polynomial regression augments features with powers (`x`, `x²`, `x³`, …), enabling a non-linear fit while still using linear coefficients.

```python
def polynomial_features(X: np.ndarray, degree: int) -> np.ndarray:
    X = np.asarray(X)
    powers = [X ** d for d in range(1, degree + 1)]
    return np.hstack(powers)

X_base = rng.uniform(-3, 3, size=(200, 1))
y_true = 0.5 * X_base[:, 0] ** 3 - 2 * X_base[:, 0] ** 2 + X_base[:, 0] + 5
noise = rng.normal(scale=5, size=200)
y = y_true + noise

X_poly = polynomial_features(X_base, degree=3)
poly_model = GradientDescentRegressor(lr=0.01, n_iter=10_000)
poly_model.fit(X_poly, y)
```

**Assumptions** match linear regression but on the engineered feature space. Watch out for multicollinearity between polynomial terms—regularization and standardization become essential as degree increases.

---

## 8. Ridge Regression (L2 Regularization)

Adds penalty `λ ||w||₂²` to discourage large coefficients, reducing variance and handling multicollinearity.

```python
ridge_model = GradientDescentRegressor(lr=0.05, n_iter=8_000, l2=0.5)
X_scaled = (X_poly - X_poly.mean(axis=0)) / X_poly.std(axis=0)
ridge_model.fit(X_scaled, y)
```

- Closed-form solution exists: `w = (XᵀX + λI)^(-1) Xᵀy`.
- Keeps all features but shrinks coefficients.
- **Use** when you expect many correlated predictors and want stability.
- **Avoid** if interpretability of raw coefficients is paramount (regularization biases them) or if features are on different scales and you cannot standardize.

---

## 9. Lasso Regression (L1 Regularization)

Adds penalty `λ ||w||₁`, promoting sparse weights (feature selection).

```python
lasso_model = GradientDescentRegressor(lr=0.01, n_iter=20_000, l1=0.1)
lasso_model.fit(X_scaled, y)
non_zero = np.sum(np.abs(lasso_model.weights[1:]) > 1e-3)
```

- Forces some coefficients exactly to zero.
- Sensitive to correlated predictors (may arbitrarily pick one); elastic net mitigates this.
- **Use** when feature selection is desired.
- **Avoid** when you need stability across correlated predictors or have more samples than features with little noise.

---

## 10. Elastic Net (Combined L1 + L2)

Balances ridge’s stability and lasso’s sparsity using penalties `λ1 ||w||₁ + λ2 ||w||₂²`.

```python
elastic_model = GradientDescentRegressor(lr=0.02, n_iter=15_000, l1=0.1, l2=0.1)
elastic_model.fit(X_scaled, y)
```

- Encourages grouped selection of correlated features.
- Hyperparameters typically tuned via cross-validation (e.g., grid over `(l1, l2)`).
- **Use** with high-dimensional data where correlations exist but you still want sparsity.
- **Avoid** if interpretability of penalty contributions is unclear to stakeholders (requires explaining two hyperparameters).

---

## 11. Practical Guidance

| Scenario | Recommended Variant |
| -------- | ------------------- |
| Few predictors, strong linear relationship, interpretability needed | Ordinary Least Squares |
| Non-linear patterns but still one main predictor | Polynomial (with low degree) + regularization |
| Many correlated predictors, want stable coefficients | Ridge |
| Need automatic feature selection | Lasso |
| High-dimensional data with correlated groups | Elastic Net |

Always standardize predictors before applying lasso, ridge, or elastic net; otherwise penalties unfairly favor certain features.

---

## 12. When to Avoid Linear Regression

- Target-response relationship is highly non-linear and feature engineering does not rectify it.
- Residuals show strong patterns (autocorrelation in time-series); consider ARIMA/GLS models.
- Outliers dominate the signal; consider robust regression (Huber, RANSAC).
- Dependent variable is categorical, count, or bounded; use logistic, Poisson, or other GLMs instead.

---

## 13. Interview Checklist

- Explain assumptions and how to diagnose violations (residual plots, VIF, Durbin–Watson).
- Derive gradient for least squares: `∂MSE/∂w = 2 Xᵀ(Xw - y) / n`.
- Discuss regularization bias-variance trade-off and how λ affects coefficients.
- Clarify metric choice (RMSE vs MAE) and what each emphasizes.
- Practice narrating feature scaling importance when using penalties.

Reimplementing these variants from scratch demonstrates comfort with linear models, optimization, and regularization—core ingredients of many machine learning interviews.
