"""
Unsupervised Learning with Scikit-Learn
Clustering and Dimensionality Reduction Examples
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons, load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, MeanShift
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (silhouette_score, davies_bouldin_score, 
                             calinski_harabasz_score, adjusted_rand_score)

# ============================================================================
# K-MEANS CLUSTERING
# ============================================================================

def kmeans_clustering():
    """Demonstrate K-Means clustering"""
    print("=" * 60)
    print("K-MEANS CLUSTERING")
    print("=" * 60)
    
    # Generate sample data
    X, y_true = make_blobs(n_samples=300, centers=4, n_features=2, 
                           cluster_std=0.6, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # K-Means clustering
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    y_pred = kmeans.fit_predict(X_scaled)
    
    print(f"\nCluster Centers:\n{kmeans.cluster_centers_}")
    print(f"Inertia: {kmeans.inertia_:.2f}")
    
    # Evaluation metrics
    silhouette = silhouette_score(X_scaled, y_pred)
    davies_bouldin = davies_bouldin_score(X_scaled, y_pred)
    calinski_harabasz = calinski_harabasz_score(X_scaled, y_pred)
    
    print(f"\nEvaluation Metrics:")
    print(f"Silhouette Score: {silhouette:.4f}")
    print(f"Davies-Bouldin Index: {davies_bouldin:.4f}")
    print(f"Calinski-Harabasz Index: {calinski_harabasz:.4f}")
    
    # If true labels are available
    ari = adjusted_rand_score(y_true, y_pred)
    print(f"Adjusted Rand Index: {ari:.4f}")


# ============================================================================
# ELBOW METHOD
# ============================================================================

def elbow_method():
    """Demonstrate elbow method for optimal k"""
    print("\n" + "=" * 60)
    print("ELBOW METHOD FOR OPTIMAL K")
    print("=" * 60)
    
    # Generate sample data
    X, _ = make_blobs(n_samples=300, centers=4, n_features=2, 
                      cluster_std=0.6, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Test different values of k
    inertias = []
    silhouette_scores = []
    k_range = range(2, 11)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        y_pred = kmeans.fit_predict(X_scaled)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, y_pred))
    
    print("\nK   | Inertia | Silhouette Score")
    print("-" * 40)
    for k, inertia, silhouette in zip(k_range, inertias, silhouette_scores):
        print(f"{k:<3} | {inertia:7.2f} | {silhouette:.4f}")


# ============================================================================
# HIERARCHICAL CLUSTERING
# ============================================================================

def hierarchical_clustering():
    """Demonstrate hierarchical clustering"""
    print("\n" + "=" * 60)
    print("HIERARCHICAL CLUSTERING")
    print("=" * 60)
    
    # Generate sample data
    X, y_true = make_blobs(n_samples=100, centers=3, n_features=2, 
                           cluster_std=0.6, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Agglomerative clustering with different linkages
    linkages = ['ward', 'complete', 'average', 'single']
    
    print("\nLinkage   | Silhouette | Davies-Bouldin | Calinski-Harabasz")
    print("-" * 70)
    
    for linkage in linkages:
        agg_clustering = AgglomerativeClustering(n_clusters=3, linkage=linkage)
        y_pred = agg_clustering.fit_predict(X_scaled)
        
        silhouette = silhouette_score(X_scaled, y_pred)
        davies_bouldin = davies_bouldin_score(X_scaled, y_pred)
        calinski_harabasz = calinski_harabasz_score(X_scaled, y_pred)
        
        print(f"{linkage:<9} | {silhouette:10.4f} | {davies_bouldin:14.4f} | {calinski_harabasz:17.2f}")


# ============================================================================
# DBSCAN CLUSTERING
# ============================================================================

def dbscan_clustering():
    """Demonstrate DBSCAN clustering"""
    print("\n" + "=" * 60)
    print("DBSCAN CLUSTERING")
    print("=" * 60)
    
    # Generate sample data with noise
    X, _ = make_moons(n_samples=300, noise=0.05, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # DBSCAN clustering
    dbscan = DBSCAN(eps=0.3, min_samples=5)
    y_pred = dbscan.fit_predict(X_scaled)
    
    # Count clusters and noise points
    n_clusters = len(set(y_pred)) - (1 if -1 in y_pred else 0)
    n_noise = list(y_pred).count(-1)
    
    print(f"\nNumber of clusters: {n_clusters}")
    print(f"Number of noise points: {n_noise}")
    
    # Evaluation (excluding noise points)
    if n_clusters > 1:
        mask = y_pred != -1
        if mask.sum() > 0:
            silhouette = silhouette_score(X_scaled[mask], y_pred[mask])
            print(f"Silhouette Score (excl. noise): {silhouette:.4f}")


# ============================================================================
# GAUSSIAN MIXTURE MODEL
# ============================================================================

def gaussian_mixture_model():
    """Demonstrate Gaussian Mixture Model"""
    print("\n" + "=" * 60)
    print("GAUSSIAN MIXTURE MODEL")
    print("=" * 60)
    
    # Generate sample data
    X, y_true = make_blobs(n_samples=300, centers=3, n_features=2, 
                           cluster_std=0.8, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # GMM clustering
    gmm = GaussianMixture(n_components=3, random_state=42)
    y_pred = gmm.fit_predict(X_scaled)
    probabilities = gmm.predict_proba(X_scaled)
    
    print(f"\nConverged: {gmm.converged_}")
    print(f"Number of iterations: {gmm.n_iter_}")
    print(f"BIC: {gmm.bic(X_scaled):.2f}")
    print(f"AIC: {gmm.aic(X_scaled):.2f}")
    
    # Evaluation metrics
    silhouette = silhouette_score(X_scaled, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    
    print(f"\nSilhouette Score: {silhouette:.4f}")
    print(f"Adjusted Rand Index: {ari:.4f}")
    
    print(f"\nSample probabilities (first 5 samples):")
    for i in range(5):
        print(f"Sample {i}: {probabilities[i]}")


# ============================================================================
# PRINCIPAL COMPONENT ANALYSIS (PCA)
# ============================================================================

def pca_analysis():
    """Demonstrate Principal Component Analysis"""
    print("\n" + "=" * 60)
    print("PRINCIPAL COMPONENT ANALYSIS (PCA)")
    print("=" * 60)
    
    # Load Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    
    print(f"\nOriginal data shape: {X.shape}")
    print(f"Features: {feature_names}")
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    
    # Explained variance
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    print(f"\nExplained variance by component:")
    for i, (var, cum_var) in enumerate(zip(explained_variance, cumulative_variance)):
        print(f"PC{i+1}: {var:.4f} (Cumulative: {cum_var:.4f})")
    
    # Reduce to 2 components
    pca_2d = PCA(n_components=2)
    X_pca_2d = pca_2d.fit_transform(X_scaled)
    
    print(f"\nReduced data shape: {X_pca_2d.shape}")
    print(f"Total variance explained by 2 components: {pca_2d.explained_variance_ratio_.sum():.4f}")


# ============================================================================
# t-SNE
# ============================================================================

def tsne_analysis():
    """Demonstrate t-SNE"""
    print("\n" + "=" * 60)
    print("t-SNE (t-Distributed Stochastic Neighbor Embedding)")
    print("=" * 60)
    
    # Load Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    print(f"\nOriginal data shape: {X.shape}")
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X_scaled)
    
    print(f"Reduced data shape: {X_tsne.shape}")
    print(f"KL divergence: {tsne.kl_divergence_:.4f}")
    print(f"Number of iterations: {tsne.n_iter_}")


# ============================================================================
# COMPARING CLUSTERING ALGORITHMS
# ============================================================================

def compare_clustering_algorithms():
    """Compare different clustering algorithms"""
    print("\n" + "=" * 60)
    print("COMPARING CLUSTERING ALGORITHMS")
    print("=" * 60)
    
    # Generate sample data
    X, y_true = make_blobs(n_samples=300, centers=4, n_features=2, 
                           cluster_std=0.6, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Define algorithms
    algorithms = {
        'K-Means': KMeans(n_clusters=4, random_state=42, n_init=10),
        'Hierarchical (Ward)': AgglomerativeClustering(n_clusters=4, linkage='ward'),
        'GMM': GaussianMixture(n_components=4, random_state=42),
        'Mean Shift': MeanShift()
    }
    
    print("\nAlgorithm           | Silhouette | Davies-Bouldin | ARI")
    print("-" * 65)
    
    for name, algorithm in algorithms.items():
        y_pred = algorithm.fit_predict(X_scaled)
        
        silhouette = silhouette_score(X_scaled, y_pred)
        davies_bouldin = davies_bouldin_score(X_scaled, y_pred)
        ari = adjusted_rand_score(y_true, y_pred)
        
        print(f"{name:<19} | {silhouette:10.4f} | {davies_bouldin:14.4f} | {ari:7.4f}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    kmeans_clustering()
    elbow_method()
    hierarchical_clustering()
    dbscan_clustering()
    gaussian_mixture_model()
    pca_analysis()
    tsne_analysis()
    compare_clustering_algorithms()
    
    print("\n" + "=" * 60)
    print("Unsupervised learning examples complete!")
    print("=" * 60)
