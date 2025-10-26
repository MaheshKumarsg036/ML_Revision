# Seaborn - Statistical Data Visualization

## Overview
Seaborn is a Python data visualization library based on Matplotlib that provides a high-level interface for drawing attractive and informative statistical graphics.

## Key Concepts

### 1. Distribution Plots
- `histplot()`: Histograms
- `kdeplot()`: Kernel density estimation
- `distplot()`: Distribution plot

### 2. Categorical Plots
- `barplot()`: Bar plots
- `boxplot()`: Box plots
- `violinplot()`: Violin plots

### 3. Relationship Plots
- `scatterplot()`: Scatter plots
- `lineplot()`: Line plots
- `pairplot()`: Pairwise relationships

### 4. Matrix Plots
- `heatmap()`: Heatmaps
- `clustermap()`: Clustered heatmaps

## Quick Reference

```python
import seaborn as sns

# Set style
sns.set_style('whitegrid')

# Distribution plot
sns.histplot(data, kde=True)

# Box plot
sns.boxplot(x='category', y='value', data=df)

# Heatmap
sns.heatmap(correlation_matrix, annot=True)

# Pair plot
sns.pairplot(df, hue='category')
```

## Files in This Directory

- **seaborn_examples.py**: Seaborn visualization examples

---
← [Matplotlib](../03_Matplotlib/) | [Statistics →](../05_Statistics/)
