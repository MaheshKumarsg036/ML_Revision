# Statistics - Foundation of Data Science 📈

Statistics provides the mathematical foundation for machine learning and data science. Understanding statistics is crucial for making data-driven decisions.

## 📚 Topics Covered

### 1. Descriptive Statistics
- Measures of central tendency (mean, median, mode)
- Measures of dispersion (variance, standard deviation, range)
- Percentiles and quartiles
- Skewness and kurtosis

### 2. Probability
- Basic probability concepts
- Conditional probability
- Bayes' theorem
- Random variables

### 3. Probability Distributions
- Normal distribution
- Binomial distribution
- Poisson distribution
- Exponential distribution
- Uniform distribution

### 4. Sampling
- Sampling methods
- Central limit theorem
- Standard error
- Confidence intervals

### 5. Hypothesis Testing
- Null and alternative hypotheses
- Type I and Type II errors
- p-values and significance levels
- t-tests (one-sample, two-sample, paired)
- Chi-square tests
- ANOVA

### 6. Correlation and Regression
- Pearson correlation
- Spearman correlation
- Covariance
- Simple linear regression
- Multiple linear regression

### 7. A/B Testing
- Experimental design
- Power analysis
- Effect size
- Statistical significance vs practical significance

## 🎯 Learning Objectives

After completing this section, you will be able to:
- Perform exploratory data analysis
- Understand probability distributions
- Conduct hypothesis tests
- Interpret statistical results
- Design and analyze experiments

## 📖 Resources

- **Notebooks in this folder:**
  - `statistics_basics.ipynb` - Comprehensive Statistics tutorial

## 💡 Quick Examples

```python
import numpy as np
from scipy import stats

# Generate sample data
data = np.random.normal(100, 15, 1000)

# Descriptive statistics
print(f"Mean: {np.mean(data)}")
print(f"Std: {np.std(data)}")

# Hypothesis test
t_stat, p_value = stats.ttest_1samp(data, 100)
print(f"P-value: {p_value}")
```

## 🔗 Next Steps

After mastering Statistics, move on to **Feature Engineering** to prepare data for ML!
