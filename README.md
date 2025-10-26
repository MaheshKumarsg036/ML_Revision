# ML_Revision - Machine Learning Comprehensive Guide

A comprehensive repository for revising core concepts in Machine Learning, Data Science, and related topics. This repository covers everything from fundamental libraries to advanced machine learning algorithms with both scikit-learn implementations and from-scratch code.

## 📚 Topics Covered

### 1. [Numpy](01_Numpy/)
Fundamental numerical computing library for Python
- Array creation and operations
- Indexing, slicing, and manipulation
- Mathematical functions and broadcasting
- Linear algebra operations
- Random number generation

**Files:**
- `README.md` - Complete Numpy concepts guide
- `numpy_basics.py` - Comprehensive examples and demonstrations

### 2. [Pandas](02_Pandas/)
Data manipulation and analysis library
- DataFrames and Series
- Data loading from various sources
- Data inspection and selection
- Data cleaning and transformation
- Grouping, aggregation, and merging
- Time series operations

**Files:**
- `README.md` - Complete Pandas concepts guide
- `pandas_basics.py` - Comprehensive examples and demonstrations

### 3. [Matplotlib](03_Matplotlib/)
Data visualization library for creating static, animated, and interactive plots
- Line plots, scatter plots, bar plots
- Histograms, pie charts, box plots
- Subplots and customization
- Saving figures

**Files:**
- `README.md` - Complete Matplotlib concepts guide
- `matplotlib_basics.py` - Comprehensive visualization examples

### 4. [Seaborn](04_Seaborn/)
Statistical data visualization built on matplotlib
- Distribution plots (histogram, KDE, violin)
- Categorical plots (bar, box, swarm)
- Relational plots (scatter, line)
- Heatmaps and correlation matrices
- Regression plots
- Multi-plot grids (pair plots, facet grids)

**Files:**
- `README.md` - Complete Seaborn concepts guide
- `seaborn_basics.py` - Comprehensive statistical visualization examples

### 5. [Statistics](05_Statistics/)
Statistical concepts and hypothesis testing
- Descriptive statistics (mean, median, variance, etc.)
- Probability distributions (normal, binomial, Poisson)
- Hypothesis testing (t-tests, ANOVA, chi-square)
- Correlation analysis
- Confidence intervals
- Non-parametric tests

**Files:**
- `README.md` - Complete Statistics concepts guide
- `statistics_basics.py` - Statistical tests and analysis examples

### 6. [Feature Engineering](06_Feature_Engineering/)
Data preprocessing and feature creation techniques
- Handling missing data (imputation methods)
- Encoding categorical variables (one-hot, label, ordinal)
- Feature scaling (standardization, normalization)
- Feature creation (polynomial, interaction, binning)
- Feature selection (filter, wrapper, embedded methods)
- Handling outliers
- Text and time series feature engineering

**Files:**
- `README.md` - Complete Feature Engineering guide
- `feature_engineering.py` - Comprehensive preprocessing examples

### 7. [Supervised Learning](07_Supervised_Learning/)
Machine learning algorithms for labeled data

#### Regression Algorithms:
- Linear Regression
- Ridge, Lasso, ElasticNet
- Polynomial Regression
- Decision Tree Regression
- Random Forest Regression
- Gradient Boosting Regression
- Support Vector Regression

#### Classification Algorithms:
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Trees
- Random Forest
- Support Vector Machines (SVM)
- Naive Bayes
- Gradient Boosting (XGBoost, LightGBM)

#### Additional Topics:
- Model evaluation metrics
- Cross-validation
- Hyperparameter tuning (Grid Search, Random Search)
- Ensemble methods (Bagging, Boosting, Stacking)

**Files:**
- `README.md` - Complete Supervised Learning guide
- `sklearn_examples.py` - Scikit-learn implementations
- `from_scratch.py` - From-scratch algorithm implementations

### 8. [Unsupervised Learning](08_Unsupervised_Learning/)
Machine learning algorithms for unlabeled data

#### Clustering Algorithms:
- K-Means
- Hierarchical Clustering (Agglomerative)
- DBSCAN
- Gaussian Mixture Models (GMM)
- Mean Shift

#### Dimensionality Reduction:
- Principal Component Analysis (PCA)
- t-SNE
- Linear Discriminant Analysis (LDA)

#### Evaluation Metrics:
- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Index
- Adjusted Rand Index

**Files:**
- `README.md` - Complete Unsupervised Learning guide
- `sklearn_examples.py` - Scikit-learn implementations
- `from_scratch.py` - From-scratch algorithm implementations

## 🚀 Getting Started

### Prerequisites
Install required dependencies:
```bash
pip install -r requirements.txt
```

### Running Examples
Each topic folder contains executable Python scripts. Navigate to any folder and run:

```bash
# Example: Run Numpy demonstrations
cd 01_Numpy
python numpy_basics.py

# Example: Run supervised learning examples
cd 07_Supervised_Learning
python sklearn_examples.py
python from_scratch.py
```

## 📋 Requirements
- Python 3.8+
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- SciPy
- Jupyter (optional, for notebooks)

See `requirements.txt` for specific versions.

## 📂 Repository Structure
```
ML_Revision/
├── 01_Numpy/
│   ├── README.md
│   └── numpy_basics.py
├── 02_Pandas/
│   ├── README.md
│   └── pandas_basics.py
├── 03_Matplotlib/
│   ├── README.md
│   └── matplotlib_basics.py
├── 04_Seaborn/
│   ├── README.md
│   └── seaborn_basics.py
├── 05_Statistics/
│   ├── README.md
│   └── statistics_basics.py
├── 06_Feature_Engineering/
│   ├── README.md
│   └── feature_engineering.py
├── 07_Supervised_Learning/
│   ├── README.md
│   ├── sklearn_examples.py
│   └── from_scratch.py
├── 08_Unsupervised_Learning/
│   ├── README.md
│   ├── sklearn_examples.py
│   └── from_scratch.py
├── datasets/
├── .gitignore
├── requirements.txt
└── README.md
```

## 🎯 Learning Path

### For Beginners:
1. Start with **Numpy** - Learn array operations and numerical computing
2. Move to **Pandas** - Master data manipulation
3. Learn **Matplotlib** and **Seaborn** - Visualize your data
4. Study **Statistics** - Understand the mathematical foundation
5. Practice **Feature Engineering** - Prepare data for ML
6. Begin with **Supervised Learning** - Start with simpler algorithms
7. Explore **Unsupervised Learning** - Understand pattern discovery

### For Intermediate/Advanced:
- Focus on **from_scratch.py** implementations to understand algorithms deeply
- Experiment with different parameters and datasets
- Study evaluation metrics and model comparison
- Practice on real-world datasets

## 💡 Key Features

### Comprehensive Coverage
- Each topic includes theoretical concepts and practical examples
- Both sklearn implementations and from-scratch code for deep understanding
- Real-world applicable examples and use cases

### Educational Value
- Clear, commented code for easy understanding
- Step-by-step demonstrations
- Best practices and common pitfalls highlighted

### Practical Focus
- Runnable examples that produce output
- Visualization where applicable
- Performance metrics and model evaluation

## 🤝 Contributing
This is a personal revision repository, but suggestions and improvements are welcome!

## 📝 License
This project is open source and available for educational purposes.

## 📧 Contact
For questions or suggestions, please open an issue in the repository.

---

**Happy Learning! 🎓**