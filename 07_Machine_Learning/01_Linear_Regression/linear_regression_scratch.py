"""
Linear Regression from Scratch
===============================

Implementation of Linear Regression using only NumPy.

Linear Regression formula: y = mx + b
Or in vector form: y = X @ w + b

Where:
- X: Feature matrix
- w: Weights (coefficients)
- b: Bias (intercept)
- y: Target values

Cost function (Mean Squared Error):
J(w, b) = (1/2m) * Σ(y_pred - y)²

Gradient Descent Update Rules:
w := w - α * (1/m) * X^T @ (y_pred - y)
b := b - α * (1/m) * Σ(y_pred - y)
"""

import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:
    """Linear Regression implemented from scratch using gradient descent."""
    
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        """
        Initialize Linear Regression model.
        
        Parameters:
        -----------
        learning_rate : float
            Learning rate for gradient descent
        n_iterations : int
            Number of iterations for training
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.cost_history = []
    
    def fit(self, X, y):
        """
        Train the linear regression model.
        
        Parameters:
        -----------
        X : numpy array, shape (n_samples, n_features)
            Training features
        y : numpy array, shape (n_samples,)
            Target values
        """
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for i in range(self.n_iterations):
            # Predictions
            y_pred = self.predict(X)
            
            # Calculate gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Calculate cost
            cost = self._compute_cost(y, y_pred)
            self.cost_history.append(cost)
            
            if i % 100 == 0:
                print(f"Iteration {i}: Cost = {cost:.4f}")
    
    def predict(self, X):
        """
        Make predictions using the trained model.
        
        Parameters:
        -----------
        X : numpy array, shape (n_samples, n_features)
            Features to predict
            
        Returns:
        --------
        predictions : numpy array, shape (n_samples,)
        """
        return np.dot(X, self.weights) + self.bias
    
    def _compute_cost(self, y_true, y_pred):
        """Compute Mean Squared Error cost."""
        n_samples = len(y_true)
        cost = (1 / (2 * n_samples)) * np.sum((y_pred - y_true) ** 2)
        return cost
    
    def score(self, X, y):
        """
        Calculate R² score.
        
        Parameters:
        -----------
        X : numpy array, shape (n_samples, n_features)
            Features
        y : numpy array, shape (n_samples,)
            True values
            
        Returns:
        --------
        r2_score : float
            R² score
        """
        y_pred = self.predict(X)
        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_residual = np.sum((y - y_pred) ** 2)
        r2 = 1 - (ss_residual / ss_total)
        return r2


def demo_linear_regression():
    """Demonstrate Linear Regression from scratch."""
    print("=" * 60)
    print("LINEAR REGRESSION FROM SCRATCH")
    print("=" * 60)
    
    # Generate synthetic data
    np.random.seed(42)
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X.squeeze() + np.random.randn(100)
    
    # Split data
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    print("\nTraining Linear Regression model...")
    
    # Train model
    model = LinearRegression(learning_rate=0.1, n_iterations=1000)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Evaluate
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"\nTraining R² score: {train_score:.4f}")
    print(f"Testing R² score: {test_score:.4f}")
    print(f"\nLearned parameters:")
    print(f"Weight: {model.weights[0]:.4f}")
    print(f"Bias: {model.bias:.4f}")
    
    # Plot results
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Training data and predictions
    plt.subplot(1, 3, 1)
    plt.scatter(X_train, y_train, color='blue', alpha=0.5, label='Training data')
    plt.plot(X_train, y_pred_train, color='red', linewidth=2, label='Predictions')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Training Data and Predictions')
    plt.legend()
    
    # Plot 2: Test data and predictions
    plt.subplot(1, 3, 2)
    plt.scatter(X_test, y_test, color='green', alpha=0.5, label='Test data')
    plt.plot(X_test, y_pred_test, color='red', linewidth=2, label='Predictions')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Test Data and Predictions')
    plt.legend()
    
    # Plot 3: Cost history
    plt.subplot(1, 3, 3)
    plt.plot(model.cost_history)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Cost Function History')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('/tmp/linear_regression_scratch.png')
    plt.close()
    print("\nPlot saved to /tmp/linear_regression_scratch.png")


def compare_with_sklearn():
    """Compare with scikit-learn implementation."""
    print("\n" + "=" * 60)
    print("COMPARISON WITH SCIKIT-LEARN")
    print("=" * 60)
    
    from sklearn.linear_model import LinearRegression as SklearnLR
    from sklearn.metrics import r2_score
    
    # Generate data
    np.random.seed(42)
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X.squeeze() + np.random.randn(100)
    
    # Our implementation
    our_model = LinearRegression(learning_rate=0.1, n_iterations=1000)
    our_model.fit(X, y)
    our_pred = our_model.predict(X)
    our_r2 = our_model.score(X, y)
    
    # Scikit-learn
    sklearn_model = SklearnLR()
    sklearn_model.fit(X, y)
    sklearn_pred = sklearn_model.predict(X)
    sklearn_r2 = r2_score(y, sklearn_pred)
    
    print("\nOur Implementation:")
    print(f"  Weight: {our_model.weights[0]:.4f}")
    print(f"  Bias: {our_model.bias:.4f}")
    print(f"  R² score: {our_r2:.4f}")
    
    print("\nScikit-learn:")
    print(f"  Weight: {sklearn_model.coef_[0]:.4f}")
    print(f"  Bias: {sklearn_model.intercept_:.4f}")
    print(f"  R² score: {sklearn_r2:.4f}")
    
    print("\nDifference:")
    print(f"  Weight difference: {abs(our_model.weights[0] - sklearn_model.coef_[0]):.6f}")
    print(f"  Bias difference: {abs(our_model.bias - sklearn_model.intercept_):.6f}")


def main():
    """Run all demonstrations."""
    demo_linear_regression()
    compare_with_sklearn()


if __name__ == "__main__":
    main()
