"""
K-Nearest Neighbors (KNN) from Scratch
=======================================

KNN is a simple, instance-based learning algorithm.
For classification, it predicts the class by majority voting of k nearest neighbors.
For regression, it predicts by averaging the values of k nearest neighbors.

Distance metrics:
- Euclidean: √(Σ(x₁ - x₂)²)
- Manhattan: Σ|x₁ - x₂|
"""

import numpy as np
from collections import Counter
import matplotlib.pyplot as plt


class KNearestNeighbors:
    """K-Nearest Neighbors implemented from scratch."""
    
    def __init__(self, k=3, distance_metric='euclidean'):
        """
        Initialize KNN classifier.
        
        Parameters:
        -----------
        k : int
            Number of neighbors to consider
        distance_metric : str
            Distance metric ('euclidean' or 'manhattan')
        """
        self.k = k
        self.distance_metric = distance_metric
        self.X_train = None
        self.y_train = None
    
    def fit(self, X, y):
        """Store training data."""
        self.X_train = X
        self.y_train = y
    
    def _compute_distance(self, x1, x2):
        """Compute distance between two points."""
        if self.distance_metric == 'euclidean':
            return np.sqrt(np.sum((x1 - x2) ** 2))
        elif self.distance_metric == 'manhattan':
            return np.sum(np.abs(x1 - x2))
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
    
    def predict(self, X):
        """Predict class labels for samples in X."""
        predictions = [self._predict_single(x) for x in X]
        return np.array(predictions)
    
    def _predict_single(self, x):
        """Predict class label for a single sample."""
        # Calculate distances to all training samples
        distances = [self._compute_distance(x, x_train) for x_train in self.X_train]
        
        # Get indices of k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        
        # Get labels of k nearest neighbors
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        # Return most common label
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
    
    def score(self, X, y):
        """Calculate accuracy score."""
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy


def demo_knn():
    """Demonstrate KNN from scratch."""
    print("=" * 60)
    print("K-NEAREST NEIGHBORS FROM SCRATCH")
    print("=" * 60)
    
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    # Generate synthetic data
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
    
    print("\nTraining KNN classifier...")
    
    # Train model
    model = KNearestNeighbors(k=5, distance_metric='euclidean')
    model.fit(X_train, y_train)
    
    # Evaluate
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    
    print(f"\nTraining Accuracy: {train_accuracy:.4f}")
    print(f"Testing Accuracy: {test_accuracy:.4f}")
    
    # Visualize
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Training data
    plt.subplot(1, 3, 1)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='RdYlBu', edgecolors='black')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Training Data')
    
    # Plot 2: Test data with predictions
    plt.subplot(1, 3, 2)
    y_pred = model.predict(X_test)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='RdYlBu', 
                edgecolors='black', s=100, label='True')
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='RdYlBu', 
                marker='x', s=100, linewidths=3, alpha=0.5, label='Predicted')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Test Predictions')
    plt.legend()
    
    # Plot 3: Decision boundary
    plt.subplot(1, 3, 3)
    
    # Create mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='RdYlBu', edgecolors='black')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(f'Decision Boundary (k={model.k})')
    
    plt.tight_layout()
    plt.savefig('/tmp/knn_scratch.png')
    plt.close()
    print("\nPlot saved to /tmp/knn_scratch.png")


def compare_k_values():
    """Compare different k values."""
    print("\n" + "=" * 60)
    print("COMPARING DIFFERENT K VALUES")
    print("=" * 60)
    
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    # Generate data
    X, y = make_classification(
        n_samples=200, n_features=2, n_informative=2,
        n_redundant=0, n_clusters_per_class=1, random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    k_values = [1, 3, 5, 7, 9, 11, 15, 20]
    accuracies = []
    
    for k in k_values:
        model = KNearestNeighbors(k=k)
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        accuracies.append(accuracy)
        print(f"k={k:2d}: Accuracy = {accuracy:.4f}")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, accuracies, marker='o', linewidth=2, markersize=8)
    plt.xlabel('k (Number of Neighbors)')
    plt.ylabel('Accuracy')
    plt.title('KNN Accuracy vs k Value')
    plt.grid(True)
    plt.savefig('/tmp/knn_k_comparison.png')
    plt.close()
    print("\nComparison plot saved to /tmp/knn_k_comparison.png")


def main():
    """Run all demonstrations."""
    demo_knn()
    compare_k_values()


if __name__ == "__main__":
    main()
