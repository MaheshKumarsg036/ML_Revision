"""
K-Means Clustering from Scratch
================================

K-Means is an unsupervised learning algorithm for clustering.

Algorithm:
1. Initialize k cluster centroids randomly
2. Assign each point to the nearest centroid
3. Update centroids as the mean of assigned points
4. Repeat steps 2-3 until convergence
"""

import numpy as np
import matplotlib.pyplot as plt


class KMeans:
    """K-Means clustering implemented from scratch."""
    
    def __init__(self, n_clusters=3, max_iters=100, random_state=None):
        """
        Initialize K-Means clustering.
        
        Parameters:
        -----------
        n_clusters : int
            Number of clusters
        max_iters : int
            Maximum number of iterations
        random_state : int
            Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.random_state = random_state
        self.centroids = None
        self.labels_ = None
        self.inertia_ = None
    
    def fit(self, X):
        """Fit K-Means to data."""
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # Initialize centroids randomly from data points
        random_indices = np.random.choice(len(X), self.n_clusters, replace=False)
        self.centroids = X[random_indices]
        
        for iteration in range(self.max_iters):
            # Assign points to nearest centroid
            old_centroids = self.centroids.copy()
            self.labels_ = self._assign_clusters(X)
            
            # Update centroids
            self.centroids = self._update_centroids(X, self.labels_)
            
            # Check for convergence
            if np.allclose(old_centroids, self.centroids):
                print(f"Converged at iteration {iteration}")
                break
        
        # Calculate inertia (sum of squared distances to nearest centroid)
        self.inertia_ = self._calculate_inertia(X, self.labels_)
        
        return self
    
    def _assign_clusters(self, X):
        """Assign each point to the nearest centroid."""
        distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)
    
    def _update_centroids(self, X, labels):
        """Update centroids as mean of assigned points."""
        new_centroids = np.zeros((self.n_clusters, X.shape[1]))
        for i in range(self.n_clusters):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                new_centroids[i] = cluster_points.mean(axis=0)
            else:
                # If cluster is empty, reinitialize randomly
                new_centroids[i] = X[np.random.choice(len(X))]
        return new_centroids
    
    def _calculate_inertia(self, X, labels):
        """Calculate sum of squared distances to centroids."""
        inertia = 0
        for i in range(self.n_clusters):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                inertia += np.sum((cluster_points - self.centroids[i])**2)
        return inertia
    
    def predict(self, X):
        """Predict cluster labels for new data."""
        return self._assign_clusters(X)
    
    def fit_predict(self, X):
        """Fit and return cluster labels."""
        self.fit(X)
        return self.labels_


def demo_kmeans():
    """Demonstrate K-Means from scratch."""
    print("=" * 60)
    print("K-MEANS CLUSTERING FROM SCRATCH")
    print("=" * 60)
    
    from sklearn.datasets import make_blobs
    
    # Generate synthetic data
    X, y_true = make_blobs(
        n_samples=300,
        centers=3,
        cluster_std=0.6,
        random_state=42
    )
    
    print("\nFitting K-Means with k=3...")
    
    # Fit K-Means
    kmeans = KMeans(n_clusters=3, max_iters=100, random_state=42)
    labels = kmeans.fit_predict(X)
    
    print(f"\nInertia: {kmeans.inertia_:.2f}")
    print(f"Cluster sizes: {np.bincount(labels)}")
    
    # Visualize
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Original data
    plt.subplot(1, 3, 1)
    plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', edgecolors='black')
    plt.title('Original Data (True Labels)')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    # Plot 2: K-Means clusters
    plt.subplot(1, 3, 2)
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', edgecolors='black')
    plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1],
                c='red', marker='X', s=300, edgecolors='black', linewidths=2,
                label='Centroids')
    plt.title('K-Means Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    
    # Plot 3: Elbow method
    plt.subplot(1, 3, 3)
    inertias = []
    k_range = range(1, 11)
    
    for k in k_range:
        km = KMeans(n_clusters=k, max_iters=100, random_state=42)
        km.fit(X)
        inertias.append(km.inertia_)
    
    plt.plot(k_range, inertias, marker='o', linewidth=2, markersize=8)
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('/tmp/kmeans_scratch.png')
    plt.close()
    print("\nPlot saved to /tmp/kmeans_scratch.png")


def compare_with_sklearn():
    """Compare with scikit-learn implementation."""
    print("\n" + "=" * 60)
    print("COMPARISON WITH SCIKIT-LEARN")
    print("=" * 60)
    
    from sklearn.datasets import make_blobs
    from sklearn.cluster import KMeans as SklearnKMeans
    
    # Generate data
    X, _ = make_blobs(n_samples=300, centers=3, random_state=42)
    
    # Our implementation
    our_kmeans = KMeans(n_clusters=3, max_iters=100, random_state=42)
    our_labels = our_kmeans.fit_predict(X)
    
    # Scikit-learn
    sklearn_kmeans = SklearnKMeans(n_clusters=3, random_state=42, n_init=10)
    sklearn_labels = sklearn_kmeans.fit_predict(X)
    
    print("\nOur Implementation:")
    print(f"  Inertia: {our_kmeans.inertia_:.2f}")
    print(f"  Cluster sizes: {np.bincount(our_labels)}")
    
    print("\nScikit-learn:")
    print(f"  Inertia: {sklearn_kmeans.inertia_:.2f}")
    print(f"  Cluster sizes: {np.bincount(sklearn_labels)}")


def main():
    """Run all demonstrations."""
    demo_kmeans()
    compare_with_sklearn()


if __name__ == "__main__":
    main()
