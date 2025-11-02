# Statistics - Statistical Concepts and Methods

## Overview
Understanding statistics is fundamental to machine learning and data analysis. This section covers essential statistical concepts.

## Key Concepts

### 1. Descriptive Statistics
- Measures of central tendency: mean, median, mode
- Measures of dispersion: variance, standard deviation, range
- Percentiles and quartiles

### 2. Probability Distributions
- Normal distribution
- Binomial distribution
- Poisson distribution
- Uniform distribution

### 3. Hypothesis Testing
- Null and alternative hypotheses
- p-values and significance levels
- t-tests
- Chi-square tests
- ANOVA

### 4. Correlation and Regression
- Pearson correlation
- Spearman correlation
- Linear regression
- R-squared

### 5. Statistical Inference
- Confidence intervals
- Margin of error
- Sample size determination

## Quick Reference

```python
import numpy as np
from scipy import stats

# Descriptive statistics
mean = np.mean(data)
std = np.std(data)
median = np.median(data)

# Correlation
corr = np.corrcoef(x, y)

# T-test
t_stat, p_value = stats.ttest_ind(group1, group2)

# Normal distribution
normal = stats.norm(loc=0, scale=1)
```

## Files in This Directory

- **statistics_basics.py**: Descriptive statistics
- **hypothesis_testing.py**: Statistical tests
- **distributions.py**: Probability distributions

---
← [Seaborn](../04_Seaborn/) | [Feature Engineering →](../06_Feature_Engineering/)
