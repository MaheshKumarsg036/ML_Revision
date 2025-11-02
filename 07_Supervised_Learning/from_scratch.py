"""
Supervised Learning - Algorithms from Scratch
Implementation of core ML algorithms without sklearn
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# LINEAR REGRESSION FROM SCRATCH
# ============================================================================

class LinearRegressionScratch:
    """Linear Regression implementation from scratch"""
    
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.losses = []
    
    def fit(self, X, y):
        """Train the model using gradient descent"""
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for i in range(self.n_iterations):
            # Forward pass
            y_predicted = np.dot(X, self.weights) + self.bias
            
            # Compute loss (MSE)
            loss = np.mean((y - y_predicted) ** 2)
            self.losses.append(loss)
            
            # Compute gradients
            dw = (1/n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1/n_samples) * np.sum(y_predicted - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict(self, X):
        """Make predictions"""
        return np.dot(X, self.weights) + self.bias


# ============================================================================
# LOGISTIC REGRESSION FROM SCRATCH
# ============================================================================

class LogisticRegressionScratch:
    """Logistic Regression implementation from scratch"""
    
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.losses = []
    
    def sigmoid(self, z):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        """Train the model using gradient descent"""
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for i in range(self.n_iterations):
            # Forward pass
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)
            
            # Compute loss (Binary Cross-Entropy)
            loss = -np.mean(y * np.log(y_predicted + 1e-15) + 
                          (1 - y) * np.log(1 - y_predicted + 1e-15))
            self.losses.append(loss)
            
            # Compute gradients
            dw = (1/n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1/n_samples) * np.sum(y_predicted - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict(self, X, threshold=0.5):
        """Make predictions"""
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        return (y_predicted >= threshold).astype(int)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        linear_model = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_model)


# ============================================================================
# K-NEAREST NEIGHBORS FROM SCRATCH
# ============================================================================

class KNNScratch:
    """K-Nearest Neighbors implementation from scratch"""
    
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None
    
    def fit(self, X, y):
        """Store training data"""
        self.X_train = X
        self.y_train = y
    
    def euclidean_distance(self, x1, x2):
        """Calculate Euclidean distance"""
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def predict(self, X):
        """Make predictions for each sample in X"""
        predictions = [self._predict_single(x) for x in X]
        return np.array(predictions)
    
    def _predict_single(self, x):
        """Make prediction for a single sample"""
        # Compute distances from x to all examples in the training set
        distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]
        
        # Get k nearest samples
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = self.y_train[k_indices]
        
        # Return most common class label
        most_common = np.bincount(k_nearest_labels).argmax()
        return most_common


# ============================================================================
# DECISION TREE FROM SCRATCH (SIMPLIFIED)
# ============================================================================

class DecisionTreeScratch:
    """Simplified Decision Tree Classifier from scratch"""
    
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None
    
    def gini_impurity(self, y):
        """Calculate Gini impurity"""
        classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities ** 2)
    
    def split(self, X, y, feature_index, threshold):
        """Split dataset based on a feature and threshold"""
        left_mask = X[:, feature_index] <= threshold
        right_mask = ~left_mask
        return X[left_mask], X[right_mask], y[left_mask], y[right_mask]
    
    def find_best_split(self, X, y):
        """Find the best feature and threshold to split on"""
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        n_features = X.shape[1]
        parent_gini = self.gini_impurity(y)
        
        for feature_index in range(n_features):
            thresholds = np.unique(X[:, feature_index])
            
            for threshold in thresholds:
                X_left, X_right, y_left, y_right = self.split(X, y, feature_index, threshold)
                
                if len(y_left) == 0 or len(y_right) == 0:
                    continue
                
                # Calculate weighted Gini impurity
                n = len(y)
                n_left, n_right = len(y_left), len(y_right)
                gini_left = self.gini_impurity(y_left)
                gini_right = self.gini_impurity(y_right)
                weighted_gini = (n_left/n) * gini_left + (n_right/n) * gini_right
                
                # Information gain
                gain = parent_gini - weighted_gini
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_index
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def build_tree(self, X, y, depth=0):
        """Recursively build the decision tree"""
        n_samples = len(y)
        n_classes = len(np.unique(y))
        
        # Stopping criteria
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or 
            n_classes == 1):
            # Return leaf node with majority class
            leaf_value = np.bincount(y).argmax()
            return {'leaf': True, 'value': leaf_value}
        
        # Find best split
        best_feature, best_threshold = self.find_best_split(X, y)
        
        if best_feature is None:
            leaf_value = np.bincount(y).argmax()
            return {'leaf': True, 'value': leaf_value}
        
        # Split the data
        X_left, X_right, y_left, y_right = self.split(X, y, best_feature, best_threshold)
        
        # Build subtrees
        left_subtree = self.build_tree(X_left, y_left, depth + 1)
        right_subtree = self.build_tree(X_right, y_right, depth + 1)
        
        return {
            'leaf': False,
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left_subtree,
            'right': right_subtree
        }
    
    def fit(self, X, y):
        """Build the decision tree"""
        self.tree = self.build_tree(X, y)
    
    def _predict_single(self, x, tree):
        """Predict single sample"""
        if tree['leaf']:
            return tree['value']
        
        if x[tree['feature']] <= tree['threshold']:
            return self._predict_single(x, tree['left'])
        else:
            return self._predict_single(x, tree['right'])
    
    def predict(self, X):
        """Make predictions"""
        return np.array([self._predict_single(x, self.tree) for x in X])


# ============================================================================
# NAIVE BAYES FROM SCRATCH
# ============================================================================

class NaiveBayesScratch:
    """Gaussian Naive Bayes implementation from scratch"""
    
    def __init__(self):
        self.classes = None
        self.mean = {}
        self.var = {}
        self.priors = {}
    
    def fit(self, X, y):
        """Train the model"""
        self.classes = np.unique(y)
        n_samples = len(y)
        
        for c in self.classes:
            X_c = X[y == c]
            self.mean[c] = np.mean(X_c, axis=0)
            self.var[c] = np.var(X_c, axis=0)
            self.priors[c] = len(X_c) / n_samples
    
    def gaussian_pdf(self, x, mean, var):
        """Calculate Gaussian probability density function"""
        eps = 1e-6  # To avoid division by zero
        coefficient = 1 / np.sqrt(2 * np.pi * var + eps)
        exponent = np.exp(-((x - mean) ** 2) / (2 * var + eps))
        return coefficient * exponent
    
    def predict(self, X):
        """Make predictions"""
        predictions = []
        
        for x in X:
            posteriors = []
            
            for c in self.classes:
                prior = np.log(self.priors[c])
                likelihood = np.sum(np.log(self.gaussian_pdf(x, self.mean[c], self.var[c])))
                posterior = prior + likelihood
                posteriors.append(posterior)
            
            predictions.append(self.classes[np.argmax(posteriors)])
        
        return np.array(predictions)


# ============================================================================
# DEMONSTRATION
# ============================================================================

def demonstrate_algorithms():
    """Demonstrate all algorithms from scratch"""
    print("=" * 60)
    print("MACHINE LEARNING ALGORITHMS FROM SCRATCH")
    print("=" * 60)
    
    # Generate synthetic data for regression
    np.random.seed(42)
    X_reg = np.random.randn(100, 1)
    y_reg = 2 * X_reg.squeeze() + 1 + np.random.randn(100) * 0.1
    
    # Linear Regression
    print("\n1. LINEAR REGRESSION")
    print("-" * 60)
    lr = LinearRegressionScratch(learning_rate=0.1, n_iterations=100)
    lr.fit(X_reg, y_reg)
    y_pred_reg = lr.predict(X_reg)
    mse = np.mean((y_reg - y_pred_reg) ** 2)
    print(f"Final MSE: {mse:.4f}")
    print(f"Weights: {lr.weights}")
    print(f"Bias: {lr.bias:.4f}")
    
    # Generate synthetic data for classification
    from sklearn.datasets import make_classification
    X_clf, y_clf = make_classification(
        n_samples=200, n_features=2, n_redundant=0, 
        n_informative=2, n_clusters_per_class=1, random_state=42
    )
    
    # Logistic Regression
    print("\n2. LOGISTIC REGRESSION")
    print("-" * 60)
    log_reg = LogisticRegressionScratch(learning_rate=0.1, n_iterations=1000)
    log_reg.fit(X_clf, y_clf)
    y_pred_log = log_reg.predict(X_clf)
    accuracy = np.mean(y_pred_log == y_clf)
    print(f"Accuracy: {accuracy:.4f}")
    
    # K-Nearest Neighbors
    print("\n3. K-NEAREST NEIGHBORS (k=3)")
    print("-" * 60)
    knn = KNNScratch(k=3)
    knn.fit(X_clf, y_clf)
    y_pred_knn = knn.predict(X_clf)
    accuracy = np.mean(y_pred_knn == y_clf)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Decision Tree
    print("\n4. DECISION TREE")
    print("-" * 60)
    dt = DecisionTreeScratch(max_depth=3, min_samples_split=2)
    dt.fit(X_clf, y_clf)
    y_pred_dt = dt.predict(X_clf)
    accuracy = np.mean(y_pred_dt == y_clf)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Naive Bayes
    print("\n5. NAIVE BAYES")
    print("-" * 60)
    nb = NaiveBayesScratch()
    nb.fit(X_clf, y_clf)
    y_pred_nb = nb.predict(X_clf)
    accuracy = np.mean(y_pred_nb == y_clf)
    print(f"Accuracy: {accuracy:.4f}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    demonstrate_algorithms()
    
    print("\n" + "=" * 60)
    print("From-scratch implementations demonstration complete!")
    print("=" * 60)
