"""
Naive Bayes Classifier from Scratch
====================================

Naive Bayes is a probabilistic classifier based on Bayes' theorem
with the "naive" assumption of feature independence.

Bayes' Theorem:
P(y|X) = P(X|y) * P(y) / P(X)

For classification, we find:
y_pred = argmax_y P(y) * Π P(x_i|y)

This implementation uses Gaussian Naive Bayes, which assumes
features follow a normal distribution within each class.
"""

import numpy as np


class GaussianNaiveBayes:
    """Gaussian Naive Bayes classifier implemented from scratch."""
    
    def __init__(self):
        """Initialize Gaussian Naive Bayes."""
        self.classes = None
        self.mean = None
        self.var = None
        self.priors = None
    
    def fit(self, X, y):
        """
        Fit Gaussian Naive Bayes.
        
        Parameters:
        -----------
        X : numpy array, shape (n_samples, n_features)
            Training features
        y : numpy array, shape (n_samples,)
            Target labels
        """
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        
        # Initialize parameters
        self.mean = np.zeros((n_classes, n_features))
        self.var = np.zeros((n_classes, n_features))
        self.priors = np.zeros(n_classes)
        
        # Calculate mean, variance, and prior for each class
        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.mean[idx, :] = X_c.mean(axis=0)
            self.var[idx, :] = X_c.var(axis=0)
            self.priors[idx] = X_c.shape[0] / n_samples
        
        return self
    
    def predict(self, X):
        """
        Predict class labels.
        
        Parameters:
        -----------
        X : numpy array, shape (n_samples, n_features)
            Features to predict
            
        Returns:
        --------
        predictions : numpy array, shape (n_samples,)
        """
        predictions = [self._predict_single(x) for x in X]
        return np.array(predictions)
    
    def _predict_single(self, x):
        """Predict class for a single sample."""
        posteriors = []
        
        for idx, c in enumerate(self.classes):
            prior = np.log(self.priors[idx])
            likelihood = np.sum(np.log(self._gaussian_pdf(x, self.mean[idx], self.var[idx])))
            posterior = prior + likelihood
            posteriors.append(posterior)
        
        return self.classes[np.argmax(posteriors)]
    
    def _gaussian_pdf(self, x, mean, var):
        """Calculate Gaussian probability density function."""
        epsilon = 1e-10  # To avoid division by zero
        numerator = np.exp(-((x - mean) ** 2) / (2 * var + epsilon))
        denominator = np.sqrt(2 * np.pi * var + epsilon)
        return numerator / denominator
    
    def score(self, X, y):
        """Calculate accuracy score."""
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Parameters:
        -----------
        X : numpy array, shape (n_samples, n_features)
            Features
            
        Returns:
        --------
        probabilities : numpy array, shape (n_samples, n_classes)
        """
        probas = []
        
        for x in X:
            posteriors = []
            for idx, c in enumerate(self.classes):
                prior = np.log(self.priors[idx])
                likelihood = np.sum(np.log(self._gaussian_pdf(x, self.mean[idx], self.var[idx])))
                posterior = prior + likelihood
                posteriors.append(posterior)
            
            # Convert log probabilities to probabilities
            posteriors = np.array(posteriors)
            posteriors = np.exp(posteriors - np.max(posteriors))  # Numerical stability
            posteriors = posteriors / np.sum(posteriors)
            probas.append(posteriors)
        
        return np.array(probas)


def demo_naive_bayes():
    """Demonstrate Naive Bayes from scratch."""
    print("=" * 60)
    print("GAUSSIAN NAIVE BAYES FROM SCRATCH")
    print("=" * 60)
    
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    
    # Load iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print("\nTraining Gaussian Naive Bayes...")
    
    # Train model
    nb = GaussianNaiveBayes()
    nb.fit(X_train, y_train)
    
    # Evaluate
    train_accuracy = nb.score(X_train, y_train)
    test_accuracy = nb.score(X_test, y_test)
    
    print(f"\nTraining Accuracy: {train_accuracy:.4f}")
    print(f"Testing Accuracy: {test_accuracy:.4f}")
    
    # Print learned parameters
    print("\nLearned Parameters:")
    print(f"Classes: {nb.classes}")
    print(f"Priors: {nb.priors}")
    print(f"\nMean (first feature per class): {nb.mean[:, 0]}")
    print(f"Variance (first feature per class): {nb.var[:, 0]}")
    
    # Visualize using first 2 features
    X_vis = iris.data[:, :2]
    X_train_vis, X_test_vis, y_train_vis, y_test_vis = train_test_split(
        X_vis, y, test_size=0.2, random_state=42
    )
    
    nb_vis = GaussianNaiveBayes()
    nb_vis.fit(X_train_vis, y_train_vis)
    
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Training data
    plt.subplot(1, 3, 1)
    scatter = plt.scatter(X_train_vis[:, 0], X_train_vis[:, 1], 
                         c=y_train_vis, cmap='viridis', edgecolors='black')
    plt.xlabel('Sepal Length')
    plt.ylabel('Sepal Width')
    plt.title('Training Data')
    plt.colorbar(scatter)
    
    # Plot 2: Test predictions
    plt.subplot(1, 3, 2)
    y_pred_vis = nb_vis.predict(X_test_vis)
    plt.scatter(X_test_vis[:, 0], X_test_vis[:, 1], 
               c=y_test_vis, cmap='viridis', edgecolors='black', s=100, label='True')
    plt.scatter(X_test_vis[:, 0], X_test_vis[:, 1], 
               c=y_pred_vis, cmap='viridis', marker='x', s=100, linewidths=3, alpha=0.5, label='Predicted')
    plt.xlabel('Sepal Length')
    plt.ylabel('Sepal Width')
    plt.title('Test Predictions')
    plt.legend()
    
    # Plot 3: Decision boundary
    plt.subplot(1, 3, 3)
    
    x_min, x_max = X_vis[:, 0].min() - 1, X_vis[:, 0].max() + 1
    y_min, y_max = X_vis[:, 1].min() - 1, X_vis[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    Z = nb_vis.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    plt.scatter(X_train_vis[:, 0], X_train_vis[:, 1], 
               c=y_train_vis, cmap='viridis', edgecolors='black')
    plt.xlabel('Sepal Length')
    plt.ylabel('Sepal Width')
    plt.title('Decision Boundary')
    
    plt.tight_layout()
    plt.savefig('/tmp/naive_bayes_scratch.png')
    plt.close()
    print("\nPlot saved to /tmp/naive_bayes_scratch.png")


def compare_with_sklearn():
    """Compare with scikit-learn implementation."""
    print("\n" + "=" * 60)
    print("COMPARISON WITH SCIKIT-LEARN")
    print("=" * 60)
    
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import GaussianNB
    
    # Load data
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Our implementation
    our_nb = GaussianNaiveBayes()
    our_nb.fit(X_train, y_train)
    our_accuracy = our_nb.score(X_test, y_test)
    
    # Scikit-learn
    sklearn_nb = GaussianNB()
    sklearn_nb.fit(X_train, y_train)
    sklearn_accuracy = sklearn_nb.score(X_test, y_test)
    
    print("\nOur Implementation:")
    print(f"  Test Accuracy: {our_accuracy:.4f}")
    print(f"  Priors: {our_nb.priors}")
    
    print("\nScikit-learn:")
    print(f"  Test Accuracy: {sklearn_accuracy:.4f}")
    print(f"  Priors: {sklearn_nb.class_prior_}")


def main():
    """Run all demonstrations."""
    demo_naive_bayes()
    compare_with_sklearn()


if __name__ == "__main__":
    main()
