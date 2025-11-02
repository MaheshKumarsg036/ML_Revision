"""
Decision Tree Classifier from Scratch
======================================

A decision tree recursively splits data based on feature values
to create a tree structure for classification.

Key concepts:
- Entropy: Measure of impurity H(S) = -Σ(p_i * log2(p_i))
- Information Gain: Reduction in entropy after split
- Gini Impurity: Alternative to entropy G(S) = 1 - Σ(p_i²)
"""

import numpy as np
from collections import Counter


class Node:
    """Node in a decision tree."""
    
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        """
        Initialize tree node.
        
        Parameters:
        -----------
        feature : int
            Feature index to split on
        threshold : float
            Threshold value for split
        left : Node
            Left child node
        right : Node
            Right child node
        value : int
            Class label (for leaf nodes)
        """
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
    
    def is_leaf(self):
        """Check if node is a leaf."""
        return self.value is not None


class DecisionTreeClassifier:
    """Decision Tree Classifier implemented from scratch."""
    
    def __init__(self, max_depth=10, min_samples_split=2, criterion='gini'):
        """
        Initialize Decision Tree.
        
        Parameters:
        -----------
        max_depth : int
            Maximum depth of the tree
        min_samples_split : int
            Minimum samples required to split a node
        criterion : str
            Split criterion ('gini' or 'entropy')
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.root = None
    
    def fit(self, X, y):
        """Build decision tree."""
        self.root = self._build_tree(X, y)
        return self
    
    def _build_tree(self, X, y, depth=0):
        """Recursively build decision tree."""
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # Stopping criteria
        if (depth >= self.max_depth or 
            n_classes == 1 or 
            n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        
        # Find best split
        best_feature, best_threshold = self._best_split(X, y)
        
        if best_feature is None:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        
        # Split data
        left_indices = X[:, best_feature] <= best_threshold
        right_indices = ~left_indices
        
        # Recursively build children
        left_child = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_child = self._build_tree(X[right_indices], y[right_indices], depth + 1)
        
        return Node(best_feature, best_threshold, left_child, right_child)
    
    def _best_split(self, X, y):
        """Find best feature and threshold to split on."""
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        n_samples, n_features = X.shape
        
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            
            for threshold in thresholds:
                gain = self._information_gain(X, y, feature, threshold)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def _information_gain(self, X, y, feature, threshold):
        """Calculate information gain from a split."""
        # Parent impurity
        parent_impurity = self._calculate_impurity(y)
        
        # Split data
        left_indices = X[:, feature] <= threshold
        right_indices = ~left_indices
        
        if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
            return 0
        
        # Children impurity
        n = len(y)
        n_left = np.sum(left_indices)
        n_right = np.sum(right_indices)
        
        left_impurity = self._calculate_impurity(y[left_indices])
        right_impurity = self._calculate_impurity(y[right_indices])
        
        # Weighted average of children impurity
        child_impurity = (n_left / n) * left_impurity + (n_right / n) * right_impurity
        
        # Information gain
        gain = parent_impurity - child_impurity
        return gain
    
    def _calculate_impurity(self, y):
        """Calculate impurity (Gini or Entropy)."""
        if self.criterion == 'gini':
            return self._gini_impurity(y)
        elif self.criterion == 'entropy':
            return self._entropy(y)
        else:
            raise ValueError(f"Unknown criterion: {self.criterion}")
    
    def _gini_impurity(self, y):
        """Calculate Gini impurity."""
        counts = np.bincount(y)
        probabilities = counts / len(y)
        gini = 1 - np.sum(probabilities ** 2)
        return gini
    
    def _entropy(self, y):
        """Calculate entropy."""
        counts = np.bincount(y)
        probabilities = counts / len(y)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        return entropy
    
    def _most_common_label(self, y):
        """Find most common label."""
        counter = Counter(y)
        return counter.most_common(1)[0][0]
    
    def predict(self, X):
        """Predict class labels."""
        predictions = [self._traverse_tree(x, self.root) for x in X]
        return np.array(predictions)
    
    def _traverse_tree(self, x, node):
        """Traverse tree to make prediction."""
        if node.is_leaf():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)
    
    def score(self, X, y):
        """Calculate accuracy."""
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy


def demo_decision_tree():
    """Demonstrate Decision Tree from scratch."""
    print("=" * 60)
    print("DECISION TREE CLASSIFIER FROM SCRATCH")
    print("=" * 60)
    
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    
    # Load iris dataset
    iris = load_iris()
    X = iris.data[:, :2]  # Use only first 2 features for visualization
    y = iris.target
    
    # Binary classification (simplify for visualization)
    binary_mask = y != 2
    X = X[binary_mask]
    y = y[binary_mask]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print("\nTraining Decision Tree...")
    
    # Train models with different criteria
    dt_gini = DecisionTreeClassifier(max_depth=5, criterion='gini')
    dt_gini.fit(X_train, y_train)
    
    dt_entropy = DecisionTreeClassifier(max_depth=5, criterion='entropy')
    dt_entropy.fit(X_train, y_train)
    
    # Evaluate
    print(f"\nGini criterion:")
    print(f"  Training Accuracy: {dt_gini.score(X_train, y_train):.4f}")
    print(f"  Testing Accuracy: {dt_gini.score(X_test, y_test):.4f}")
    
    print(f"\nEntropy criterion:")
    print(f"  Training Accuracy: {dt_entropy.score(X_train, y_train):.4f}")
    print(f"  Testing Accuracy: {dt_entropy.score(X_test, y_test):.4f}")
    
    # Visualize decision boundaries
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    for idx, (dt, title) in enumerate([(dt_gini, 'Gini'), (dt_entropy, 'Entropy')]):
        ax = axes[idx]
        
        # Create mesh
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))
        
        Z = dt.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='RdYlBu', 
                  edgecolors='black', s=50)
        ax.set_xlabel('Sepal Length')
        ax.set_ylabel('Sepal Width')
        ax.set_title(f'Decision Tree ({title})')
    
    plt.tight_layout()
    plt.savefig('/tmp/decision_tree_scratch.png')
    plt.close()
    print("\nPlot saved to /tmp/decision_tree_scratch.png")


def main():
    """Run demonstration."""
    demo_decision_tree()


if __name__ == "__main__":
    main()
