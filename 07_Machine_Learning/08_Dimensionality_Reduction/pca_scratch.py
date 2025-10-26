"""
Principal Component Analysis (PCA) from Scratch
================================================

PCA is a dimensionality reduction technique that finds
the principal components (directions of maximum variance).

Steps:
1. Standardize the data
2. Compute covariance matrix
3. Compute eigenvalues and eigenvectors
4. Sort eigenvectors by eigenvalues
5. Select top k eigenvectors
6. Transform data to new space
"""

import numpy as np
import matplotlib.pyplot as plt


class PCA:
    """Principal Component Analysis implemented from scratch."""
    
    def __init__(self, n_components=2):
        """
        Initialize PCA.
        
        Parameters:
        -----------
        n_components : int
            Number of principal components to keep
        """
        self.n_components = n_components
        self.components_ = None
        self.mean_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
    
    def fit(self, X):
        """Fit PCA on data."""
        # Center the data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        # Compute covariance matrix
        cov_matrix = np.cov(X_centered.T)
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
        # Sort eigenvectors by eigenvalues in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Select top k eigenvectors
        self.components_ = eigenvectors[:, :self.n_components].T
        
        # Store explained variance
        self.explained_variance_ = eigenvalues[:self.n_components]
        total_variance = np.sum(eigenvalues)
        self.explained_variance_ratio_ = self.explained_variance_ / total_variance
        
        return self
    
    def transform(self, X):
        """Transform data to principal component space."""
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_.T)
    
    def fit_transform(self, X):
        """Fit and transform data."""
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X_transformed):
        """Transform data back to original space."""
        return np.dot(X_transformed, self.components_) + self.mean_


def demo_pca():
    """Demonstrate PCA from scratch."""
    print("=" * 60)
    print("PRINCIPAL COMPONENT ANALYSIS FROM SCRATCH")
    print("=" * 60)
    
    from sklearn.datasets import load_iris
    
    # Load iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    print(f"\nOriginal data shape: {X.shape}")
    
    # Apply PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    print(f"Transformed data shape: {X_pca.shape}")
    print(f"\nExplained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total variance explained: {np.sum(pca.explained_variance_ratio_):.4f}")
    
    # Visualize
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Original data (first 2 features)
    plt.subplot(1, 3, 1)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='black')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Original Data (First 2 Features)')
    plt.colorbar(scatter)
    
    # Plot 2: PCA transformed data
    plt.subplot(1, 3, 2)
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolors='black')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title('PCA Transformed Data')
    plt.colorbar(scatter)
    
    # Plot 3: Explained variance
    plt.subplot(1, 3, 3)
    
    # Calculate explained variance for all components
    pca_full = PCA(n_components=4)
    pca_full.fit(X)
    
    cumsum = np.cumsum(pca_full.explained_variance_ratio_)
    plt.plot(range(1, 5), cumsum, marker='o', linewidth=2, markersize=8)
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Explained Variance vs Components')
    plt.grid(True)
    plt.ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig('/tmp/pca_scratch.png')
    plt.close()
    print("\nPlot saved to /tmp/pca_scratch.png")


def compare_with_sklearn():
    """Compare with scikit-learn implementation."""
    print("\n" + "=" * 60)
    print("COMPARISON WITH SCIKIT-LEARN")
    print("=" * 60)
    
    from sklearn.datasets import load_iris
    from sklearn.decomposition import PCA as SklearnPCA
    
    # Load data
    iris = load_iris()
    X = iris.data
    
    # Our implementation
    our_pca = PCA(n_components=2)
    our_transformed = our_pca.fit_transform(X)
    
    # Scikit-learn
    sklearn_pca = SklearnPCA(n_components=2)
    sklearn_transformed = sklearn_pca.fit_transform(X)
    
    print("\nOur Implementation:")
    print(f"  Explained variance ratio: {our_pca.explained_variance_ratio_}")
    print(f"  Total variance explained: {np.sum(our_pca.explained_variance_ratio_):.4f}")
    
    print("\nScikit-learn:")
    print(f"  Explained variance ratio: {sklearn_pca.explained_variance_ratio_}")
    print(f"  Total variance explained: {np.sum(sklearn_pca.explained_variance_ratio_):.4f}")
    
    # Note: Signs of components might differ, but that's OK
    print("\nNote: Component signs may differ between implementations,")
    print("but this doesn't affect the results as they represent the same directions.")


def main():
    """Run all demonstrations."""
    demo_pca()
    compare_with_sklearn()


if __name__ == "__main__":
    main()
