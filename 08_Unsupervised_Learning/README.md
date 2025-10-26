# Unsupervised Learning Revision Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Clustering Algorithms](#clustering-algorithms)
3. [Dimensionality Reduction](#dimensionality-reduction)
4. [Association Rules](#association-rules)
5. [Anomaly Detection](#anomaly-detection)
6. [Evaluation Metrics](#evaluation-metrics)

## Introduction
Unsupervised learning finds patterns in unlabeled data without predefined outcomes.

## Clustering Algorithms
Grouping similar data points together

### 1. K-Means Clustering
- **Concept**: Partitions data into k clusters
- **Algorithm**:
  1. Initialize k centroids randomly
  2. Assign each point to nearest centroid
  3. Update centroids as mean of assigned points
  4. Repeat until convergence
- **Pros**: Fast, simple, scalable
- **Cons**: Requires k to be specified, sensitive to initialization
- **Use cases**: Customer segmentation, image compression

### 2. Hierarchical Clustering
- **Types**:
  - Agglomerative (bottom-up): Start with individual points, merge clusters
  - Divisive (top-down): Start with all points, split clusters
- **Linkage methods**: Single, complete, average, Ward
- **Visualization**: Dendrogram
- **Pros**: No need to specify k, hierarchical structure
- **Cons**: Computationally expensive

### 3. DBSCAN (Density-Based Spatial Clustering)
- **Concept**: Groups points based on density
- **Parameters**:
  - eps: Maximum distance between points
  - min_samples: Minimum points to form dense region
- **Point types**: Core, border, noise/outlier
- **Pros**: Finds arbitrary shaped clusters, identifies outliers
- **Cons**: Sensitive to parameters

### 4. Gaussian Mixture Models (GMM)
- **Concept**: Assumes data is generated from mixture of Gaussian distributions
- **Algorithm**: Expectation-Maximization (EM)
- **Pros**: Soft clustering (probabilistic), flexible cluster shapes
- **Cons**: Requires number of components, sensitive to initialization

### 5. Mean Shift
- **Concept**: Finds modes in feature density
- **Pros**: No need to specify k, finds arbitrary shapes
- **Cons**: Computationally expensive

## Dimensionality Reduction
Reducing number of features while preserving information

### 1. Principal Component Analysis (PCA)
- **Concept**: Linear transformation to orthogonal components
- **Method**: Eigenvalue decomposition of covariance matrix
- **Output**: Principal components (ordered by variance explained)
- **Use cases**: Visualization, noise reduction, feature extraction
- **Assumptions**: Linear relationships, high variance = high importance

### 2. t-SNE (t-Distributed Stochastic Neighbor Embedding)
- **Concept**: Non-linear dimensionality reduction for visualization
- **Method**: Preserves local structure
- **Pros**: Great for visualization (2D/3D)
- **Cons**: Computationally expensive, non-deterministic
- **Use cases**: Visualization of high-dimensional data

### 3. Linear Discriminant Analysis (LDA)
- **Concept**: Supervised dimensionality reduction
- **Method**: Maximizes class separability
- **Use cases**: Feature extraction for classification

### 4. Autoencoders
- **Concept**: Neural networks for non-linear dimensionality reduction
- **Architecture**: Encoder → Latent representation → Decoder

### 5. Independent Component Analysis (ICA)
- **Concept**: Separates multivariate signal into independent components
- **Use cases**: Signal processing, blind source separation

## Association Rules
Finding relationships between variables

### Apriori Algorithm
- **Metrics**:
  - Support: Frequency of itemset
  - Confidence: Likelihood of B given A
  - Lift: Strength of association
- **Use cases**: Market basket analysis, recommendation systems

## Anomaly Detection
Identifying unusual patterns

### Methods:
1. **Statistical**: Z-score, IQR
2. **Isolation Forest**: Isolates anomalies in tree structure
3. **One-Class SVM**: Learns boundary around normal data
4. **Local Outlier Factor (LOF)**: Based on local density
5. **Autoencoders**: High reconstruction error for anomalies

## Evaluation Metrics

### Clustering Metrics (When True Labels Available)
- **Adjusted Rand Index (ARI)**: Similarity between clustering and ground truth
- **Normalized Mutual Information (NMI)**: Shared information
- **Homogeneity**: Each cluster contains only one class
- **Completeness**: All members of a class in same cluster
- **V-measure**: Harmonic mean of homogeneity and completeness

### Clustering Metrics (Without True Labels)
- **Silhouette Score**: How similar point is to own cluster vs other clusters
  - Range: [-1, 1], higher is better
- **Davies-Bouldin Index**: Average similarity between clusters
  - Lower is better
- **Calinski-Harabasz Index**: Ratio of between-cluster to within-cluster variance
  - Higher is better
- **Inertia/Within-cluster sum of squares**: Lower is better

### Dimensionality Reduction Metrics
- **Explained Variance Ratio**: Proportion of variance retained
- **Reconstruction Error**: Difference between original and reconstructed data

## Best Practices
- **Feature scaling**: Always scale features for distance-based algorithms
- **Choose k wisely**: Use elbow method, silhouette analysis
- **Multiple runs**: Use different random initializations
- **Domain knowledge**: Consider interpretability and business context
