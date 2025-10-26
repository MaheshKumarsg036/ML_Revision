"""
Statistics Basics - Descriptive Statistics and Distributions
============================================================

This module demonstrates statistical concepts and operations.
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


def descriptive_statistics():
    """Demonstrate descriptive statistics."""
    print("=" * 60)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 60)
    
    np.random.seed(42)
    data = np.random.randn(1000) * 10 + 50
    
    print(f"\nMean: {np.mean(data):.2f}")
    print(f"Median: {np.median(data):.2f}")
    print(f"Mode: {stats.mode(data.round()).mode:.2f}")
    print(f"Standard Deviation: {np.std(data):.2f}")
    print(f"Variance: {np.var(data):.2f}")
    print(f"Min: {np.min(data):.2f}")
    print(f"Max: {np.max(data):.2f}")
    print(f"Range: {np.max(data) - np.min(data):.2f}")
    
    print(f"\nQuartiles:")
    print(f"Q1 (25th percentile): {np.percentile(data, 25):.2f}")
    print(f"Q2 (50th percentile): {np.percentile(data, 50):.2f}")
    print(f"Q3 (75th percentile): {np.percentile(data, 75):.2f}")
    print(f"IQR: {np.percentile(data, 75) - np.percentile(data, 25):.2f}")


def probability_distributions():
    """Demonstrate probability distributions."""
    print("\n" + "=" * 60)
    print("PROBABILITY DISTRIBUTIONS")
    print("=" * 60)
    
    # Normal distribution
    x = np.linspace(-4, 4, 100)
    normal = stats.norm.pdf(x, 0, 1)
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(x, normal)
    plt.title('Normal Distribution')
    plt.xlabel('x')
    plt.ylabel('PDF')
    
    # Binomial distribution
    n, p = 10, 0.5
    x_binom = np.arange(0, n+1)
    binomial = stats.binom.pmf(x_binom, n, p)
    
    plt.subplot(2, 2, 2)
    plt.bar(x_binom, binomial)
    plt.title('Binomial Distribution (n=10, p=0.5)')
    plt.xlabel('x')
    plt.ylabel('PMF')
    
    # Poisson distribution
    mu = 3
    x_poisson = np.arange(0, 15)
    poisson = stats.poisson.pmf(x_poisson, mu)
    
    plt.subplot(2, 2, 3)
    plt.bar(x_poisson, poisson)
    plt.title('Poisson Distribution (μ=3)')
    plt.xlabel('x')
    plt.ylabel('PMF')
    
    # Uniform distribution
    x_uniform = np.linspace(0, 10, 100)
    uniform = stats.uniform.pdf(x_uniform, 0, 10)
    
    plt.subplot(2, 2, 4)
    plt.plot(x_uniform, uniform)
    plt.title('Uniform Distribution')
    plt.xlabel('x')
    plt.ylabel('PDF')
    
    plt.tight_layout()
    plt.savefig('/tmp/distributions.png')
    plt.close()
    print("\nDistributions plot saved to /tmp/distributions.png")


def correlation_analysis():
    """Demonstrate correlation analysis."""
    print("\n" + "=" * 60)
    print("CORRELATION ANALYSIS")
    print("=" * 60)
    
    np.random.seed(42)
    x = np.random.randn(100)
    y = 2 * x + np.random.randn(100) * 0.5
    
    # Pearson correlation
    pearson_corr, pearson_p = stats.pearsonr(x, y)
    print(f"\nPearson Correlation: {pearson_corr:.4f}")
    print(f"P-value: {pearson_p:.4f}")
    
    # Spearman correlation
    spearman_corr, spearman_p = stats.spearmanr(x, y)
    print(f"\nSpearman Correlation: {spearman_corr:.4f}")
    print(f"P-value: {spearman_p:.4f}")


def hypothesis_testing():
    """Demonstrate hypothesis testing."""
    print("\n" + "=" * 60)
    print("HYPOTHESIS TESTING")
    print("=" * 60)
    
    np.random.seed(42)
    group1 = np.random.randn(100) * 10 + 50
    group2 = np.random.randn(100) * 10 + 52
    
    # T-test
    t_stat, p_value = stats.ttest_ind(group1, group2)
    print(f"\nIndependent T-test:")
    print(f"T-statistic: {t_stat:.4f}")
    print(f"P-value: {p_value:.4f}")
    
    if p_value < 0.05:
        print("Result: Reject null hypothesis (significant difference)")
    else:
        print("Result: Fail to reject null hypothesis (no significant difference)")


def main():
    """Run all examples."""
    descriptive_statistics()
    probability_distributions()
    correlation_analysis()
    hypothesis_testing()


if __name__ == "__main__":
    main()
