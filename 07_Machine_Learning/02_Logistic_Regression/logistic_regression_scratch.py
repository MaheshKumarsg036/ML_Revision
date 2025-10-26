"""
Logistic Regression from Scratch
=================================

Implementation of Logistic Regression using only NumPy.

Sigmoid function: σ(z) = 1 / (1 + e^(-z))
Prediction: y_pred = σ(X @ w + b)

Cost function (Binary Cross-Entropy):
J(w, b) = -(1/m) * Σ[y*log(y_pred) + (1-y)*log(1-y_pred)]

Gradient Descent Update Rules:
w := w - α * (1/m) * X^T @ (y_pred - y)
b := b - α * (1/m) * Σ(y_pred - y)
"""

import numpy as np
import matplotlib.pyplot as plt


class LogisticRegression:
    """Logistic Regression implemented from scratch."""
    
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        """Initialize Logistic Regression model."""
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.cost_history = []
    
    def _sigmoid(self, z):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        """Train the logistic regression model."""
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for i in range(self.n_iterations):
            # Linear combination
            linear_model = np.dot(X, self.weights) + self.bias
            # Apply sigmoid
            y_pred = self._sigmoid(linear_model)
            
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
    
    def predict_proba(self, X):
        """Predict probabilities."""
        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)
    
    def predict(self, X, threshold=0.5):
        """Make binary predictions."""
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)
    
    def _compute_cost(self, y_true, y_pred):
        """Compute binary cross-entropy cost."""
        epsilon = 1e-15  # To avoid log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        cost = -(1 / len(y_true)) * np.sum(
            y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
        )
        return cost
    
    def score(self, X, y):
        """Calculate accuracy score."""
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy


def demo_logistic_regression():
    """Demonstrate Logistic Regression from scratch."""
    print("=" * 60)
    print("LOGISTIC REGRESSION FROM SCRATCH")
    print("=" * 60)
    
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    # Generate synthetic binary classification data
    X, y = make_classification(
        n_samples=200,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_clusters_per_class=1,
        random_state=42
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print("\nTraining Logistic Regression model...")
    
    # Train model
    model = LogisticRegression(learning_rate=0.1, n_iterations=1000)
    model.fit(X_train, y_train)
    
    # Evaluate
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    
    print(f"\nTraining Accuracy: {train_accuracy:.4f}")
    print(f"Testing Accuracy: {test_accuracy:.4f}")
    
    # Visualize decision boundary
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Training data with decision boundary
    plt.subplot(1, 3, 1)
    
    # Create mesh
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='RdYlBu', edgecolors='black')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Training Data with Decision Boundary')
    
    # Plot 2: Test data
    plt.subplot(1, 3, 2)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='RdYlBu', edgecolors='black')
    y_pred_test = model.predict(X_test)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred_test, cmap='RdYlBu', 
                marker='x', s=100, linewidths=2, alpha=0.5)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Test Data Predictions')
    
    # Plot 3: Cost history
    plt.subplot(1, 3, 3)
    plt.plot(model.cost_history)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Cost Function History')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('/tmp/logistic_regression_scratch.png')
    plt.close()
    print("\nPlot saved to /tmp/logistic_regression_scratch.png")


def main():
    """Run demonstration."""
    demo_logistic_regression()


if __name__ == "__main__":
    main()
