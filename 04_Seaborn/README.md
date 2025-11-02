# Seaborn - Statistical Visualization 🎨

Seaborn is a Python visualization library based on Matplotlib that provides a high-level interface for drawing attractive statistical graphics.

## 📚 Topics Covered

### 1. Distribution Plots
- Histograms (histplot)
- KDE plots (kdeplot)
- Distribution plots (displot)
- Rug plots

### 2. Categorical Plots
- Bar plots (barplot)
- Count plots (countplot)
- Box plots (boxplot)
- Violin plots (violinplot)
- Strip plots (stripplot)
- Swarm plots (swarmplot)

### 3. Regression Plots
- Linear models (lmplot)
- Regression plots (regplot)
- Residual plots (residplot)

### 4. Matrix Plots
- Heatmaps (heatmap)
- Cluster maps (clustermap)
- Correlation matrices

### 5. Multi-plot Grids
- FacetGrid
- PairGrid
- JointGrid
- Pair plots (pairplot)

### 6. Styling
- Color palettes
- Themes and contexts
- Axis styles

## 🎯 Learning Objectives

After completing this section, you will be able to:
- Create statistical visualizations
- Visualize distributions and relationships
- Make publication-ready plots
- Use color effectively
- Create multi-panel statistical graphics

## 📖 Resources

- **Official Documentation:** [seaborn.pydata.org](https://seaborn.pydata.org/)
- **Notebooks in this folder:**
  - `seaborn_basics.ipynb` - Comprehensive Seaborn tutorial

## 💡 Quick Examples

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Load example dataset
tips = sns.load_dataset('tips')

# Create visualization
sns.scatterplot(data=tips, x='total_bill', y='tip', hue='time')
plt.title('Tips vs Total Bill')
plt.show()
```

## 🔗 Next Steps

After mastering Seaborn, move on to **Statistics** to understand the theory!
