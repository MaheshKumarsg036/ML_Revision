"""
Unsupervised Learning - Algorithms from Scratch
Implementation of core algorithms without sklearn
"""

import numpy as np

# ============================================================================
# K-MEANS FROM SCRATCH
# ============================================================================

class KMeansScratch:
    """K-Means Clustering implementation from scratch"""
    
    def __init__(self, n_clusters=3, max_iterations=100, random_state=None):
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.random_state = random_state
        self.centroids = None
        self.labels = None
    
    def initialize_centroids(self, X):
        """Initialize centroids randomly from data points"""
        if self.random_state is not None:
            np.random.seed(self.random_state)
        indices = np.random.choice(len(X), self.n_clusters, replace=False)
        return X[indices]
    
    def euclidean_distance(self, x1, x2):
        """Calculate Euclidean distance"""
        return np.sqrt(np.sum((x1 - x2) ** 2, axis=1))
    
    def assign_clusters(self, X):
        """Assign each point to nearest centroid"""
        distances = np.zeros((len(X), self.n_clusters))
        for i, centroid in enumerate(self.centroids):
            distances[:, i] = self.euclidean_distance(X, centroid)
        return np.argmin(distances, axis=1)
    
    def update_centroids(self, X, labels):
        """Update centroids as mean of assigned points"""
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        for i in range(self.n_clusters):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                centroids[i] = cluster_points.mean(axis=0)
            else:
                # If cluster is empty, reinitialize randomly
                centroids[i] = X[np.random.choice(len(X))]
        return centroids
    
    def fit(self, X):
        """Fit K-Means model"""
        # Initialize centroids
        self.centroids = self.initialize_centroids(X)
        
        for iteration in range(self.max_iterations):
            # Assign clusters
            old_centroids = self.centroids.copy()
            self.labels = self.assign_clusters(X)
            
            # Update centroids
            self.centroids = self.update_centroids(X, self.labels)
            
            # Check convergence
            if np.allclose(old_centroids, self.centroids):
                print(f"Converged at iteration {iteration + 1}")
                break
        
        return self
    
    def predict(self, X):
        """Predict cluster labels for new data"""
        return self.assign_clusters(X)
    
    def fit_predict(self, X):
        """Fit model and return labels"""
        self.fit(X)
        return self.labels


# ============================================================================
# PCA FROM SCRATCH
# ============================================================================

class PCAScratch:
    """Principal Component Analysis implementation from scratch"""
    
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.explained_variance = None
        self.explained_variance_ratio = None
    
    def fit(self, X):
        """Fit PCA model"""
        # Center the data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        
        # Compute covariance matrix
        cov_matrix = np.cov(X_centered.T)
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
        # Sort eigenvectors by eigenvalues in descending order
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Store first n_components eigenvectors as principal components
        self.components = eigenvectors[:, :self.n_components]
        
        # Store explained variance
        self.explained_variance = eigenvalues[:self.n_components]
        self.explained_variance_ratio = self.explained_variance / np.sum(eigenvalues)
        
        return self
    
    def transform(self, X):
        """Transform data to principal components"""
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)
    
    def fit_transform(self, X):
        """Fit model and transform data"""
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X_transformed):
        """Transform data back to original space"""
        return np.dot(X_transformed, self.components.T) + self.mean


# ============================================================================
# DBSCAN FROM SCRATCH (SIMPLIFIED)
# ============================================================================

class DBSCANScratch:
    """DBSCAN Clustering implementation from scratch (simplified)"""
    
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels = None
    
    def euclidean_distance(self, x1, x2):
        """Calculate Euclidean distance"""
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def get_neighbors(self, X, point_idx):
        """Get all neighbors within eps distance"""
        neighbors = []
        for i, point in enumerate(X):
            if self.euclidean_distance(X[point_idx], point) <= self.eps:
                neighbors.append(i)
        return neighbors
    
    def fit_predict(self, X):
        """Fit DBSCAN and return cluster labels"""
        n_samples = len(X)
        self.labels = np.full(n_samples, -1)  # Initialize all as noise (-1)
        cluster_id = 0
        
        for point_idx in range(n_samples):
            # Skip if already processed
            if self.labels[point_idx] != -1:
                continue
            
            # Get neighbors
            neighbors = self.get_neighbors(X, point_idx)
            
            # Check if core point
            if len(neighbors) < self.min_samples:
                continue  # Mark as noise
            
            # Start new cluster
            self.labels[point_idx] = cluster_id
            
            # Expand cluster
            i = 0
            while i < len(neighbors):
                neighbor_idx = neighbors[i]
                
                # Change noise to border point
                if self.labels[neighbor_idx] == -1:
                    self.labels[neighbor_idx] = cluster_id
                
                # Skip if already processed
                if self.labels[neighbor_idx] != -1:
                    i += 1
                    continue
                
                # Add to cluster
                self.labels[neighbor_idx] = cluster_id
                
                # Check if neighbor is also core point
                neighbor_neighbors = self.get_neighbors(X, neighbor_idx)
                if len(neighbor_neighbors) >= self.min_samples:
                    neighbors.extend(neighbor_neighbors)
                
                i += 1
            
            cluster_id += 1
        
        return self.labels


# ============================================================================
# HIERARCHICAL CLUSTERING FROM SCRATCH (SIMPLIFIED)
# ============================================================================

class HierarchicalClusteringScratch:
    """Agglomerative Hierarchical Clustering from scratch (simplified)"""
    
    def __init__(self, n_clusters=2, linkage='single'):
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.labels = None
    
    def euclidean_distance(self, x1, x2):
        """Calculate Euclidean distance"""
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def cluster_distance(self, X, cluster1, cluster2):
        """Calculate distance between two clusters"""
        distances = []
        
        for i in cluster1:
            for j in cluster2:
                distances.append(self.euclidean_distance(X[i], X[j]))
        
        if self.linkage == 'single':
            return min(distances)
        elif self.linkage == 'complete':
            return max(distances)
        elif self.linkage == 'average':
            return np.mean(distances)
        else:
            return min(distances)  # Default to single
    
    def fit_predict(self, X):
        """Fit hierarchical clustering and return labels"""
        n_samples = len(X)
        
        # Initialize: each point is its own cluster
        clusters = [[i] for i in range(n_samples)]
        
        # Merge clusters until we have n_clusters
        while len(clusters) > self.n_clusters:
            # Find pair of clusters with minimum distance
            min_distance = float('inf')
            merge_i, merge_j = 0, 1
            
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    distance = self.cluster_distance(X, clusters[i], clusters[j])
                    if distance < min_distance:
                        min_distance = distance
                        merge_i, merge_j = i, j
            
            # Merge the two closest clusters
            clusters[merge_i].extend(clusters[merge_j])
            clusters.pop(merge_j)
        
        # Assign labels
        self.labels = np.zeros(n_samples, dtype=int)
        for cluster_id, cluster in enumerate(clusters):
            for point_idx in cluster:
                self.labels[point_idx] = cluster_id
        
        return self.labels


# ============================================================================
# DEMONSTRATION
# ============================================================================

def demonstrate_algorithms():
    """Demonstrate all algorithms from scratch"""
    print("=" * 60)
    print("UNSUPERVISED LEARNING ALGORITHMS FROM SCRATCH")
    print("=" * 60)
    
    # Generate synthetic data
    np.random.seed(42)
    
    # Data for clustering
    cluster1 = np.random.randn(50, 2) + np.array([0, 0])
    cluster2 = np.random.randn(50, 2) + np.array([5, 5])
    cluster3 = np.random.randn(50, 2) + np.array([10, 0])
    X_clustering = np.vstack([cluster1, cluster2, cluster3])
    
    # K-Means Clustering
    print("\n1. K-MEANS CLUSTERING")
    print("-" * 60)
    kmeans = KMeansScratch(n_clusters=3, random_state=42)
    labels_kmeans = kmeans.fit_predict(X_clustering)
    print(f"Cluster labels (first 10): {labels_kmeans[:10]}")
    print(f"Cluster centers:\n{kmeans.centroids}")
    
    # Data for PCA
    X_pca = np.random.randn(100, 5)
    X_pca[:, 0] = X_pca[:, 0] * 3  # Make first dimension have higher variance
    
    # PCA
    print("\n2. PRINCIPAL COMPONENT ANALYSIS (PCA)")
    print("-" * 60)
    pca = PCAScratch(n_components=2)
    X_transformed = pca.fit_transform(X_pca)
    print(f"Original shape: {X_pca.shape}")
    print(f"Transformed shape: {X_transformed.shape}")
    print(f"Explained variance ratio: {pca.explained_variance_ratio}")
    print(f"Total variance explained: {pca.explained_variance_ratio.sum():.4f}")
    
    # DBSCAN
    print("\n3. DBSCAN CLUSTERING")
    print("-" * 60)
    dbscan = DBSCANScratch(eps=1.5, min_samples=5)
    labels_dbscan = dbscan.fit_predict(X_clustering)
    n_clusters = len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)
    n_noise = list(labels_dbscan).count(-1)
    print(f"Number of clusters: {n_clusters}")
    print(f"Number of noise points: {n_noise}")
    print(f"Cluster labels (first 10): {labels_dbscan[:10]}")
    
    # Hierarchical Clustering
    print("\n4. HIERARCHICAL CLUSTERING")
    print("-" * 60)
    # Use smaller dataset for efficiency
    X_small = X_clustering[:30]
    for linkage in ['single', 'complete', 'average']:
        hierarchical = HierarchicalClusteringScratch(n_clusters=3, linkage=linkage)
        labels_hier = hierarchical.fit_predict(X_small)
        print(f"Linkage: {linkage}")
        print(f"  Cluster labels (first 10): {labels_hier[:10]}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    demonstrate_algorithms()
    
    print("\n" + "=" * 60)
    print("From-scratch implementations demonstration complete!")
    print("=" * 60)
