# Matplotlib - Data Visualization

## Overview
Matplotlib is a comprehensive library for creating static, animated, and interactive visualizations in Python.

## Key Concepts

### 1. Basic Plots
- Line plots
- Scatter plots
- Bar charts
- Histograms
- Pie charts

### 2. Customization
- Colors and styles
- Labels and titles
- Legends
- Grid
- Axis limits and scales

### 3. Subplots
- Creating multiple plots
- Figure and axes
- Layout management

## Quick Reference

```python
import matplotlib.pyplot as plt

# Line plot
plt.plot(x, y)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Title')
plt.show()

# Scatter plot
plt.scatter(x, y)

# Bar chart
plt.bar(categories, values)

# Histogram
plt.hist(data, bins=20)

# Subplots
fig, axes = plt.subplots(2, 2)
axes[0, 0].plot(x, y)
```

## Files in This Directory

- **matplotlib_basics.py**: Basic plotting examples
- **matplotlib_advanced.py**: Advanced customization

---
← [Pandas](../02_Pandas/) | [Seaborn →](../04_Seaborn/)
