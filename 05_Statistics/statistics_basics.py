"""
Statistics Basics - Descriptive and Inferential Statistics
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# DESCRIPTIVE STATISTICS
# ============================================================================

def descriptive_statistics():
    """Demonstrate descriptive statistics"""
    print("=" * 60)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 60)
    
    # Generate sample data
    np.random.seed(42)
    data = np.random.randn(1000) * 10 + 50
    
    print(f"\nSample size: {len(data)}")
    
    # Measures of central tendency
    print("\nMeasures of Central Tendency:")
    print(f"Mean: {np.mean(data):.2f}")
    print(f"Median: {np.median(data):.2f}")
    print(f"Mode: {stats.mode(data.round(), keepdims=True)[0][0]:.2f}")
    
    # Measures of dispersion
    print("\nMeasures of Dispersion:")
    print(f"Range: {np.ptp(data):.2f}")
    print(f"Variance: {np.var(data):.2f}")
    print(f"Standard Deviation: {np.std(data):.2f}")
    print(f"IQR: {stats.iqr(data):.2f}")
    
    # Measures of shape
    print("\nMeasures of Shape:")
    print(f"Skewness: {stats.skew(data):.2f}")
    print(f"Kurtosis: {stats.kurtosis(data):.2f}")
    
    # Percentiles
    print("\nPercentiles:")
    print(f"25th percentile: {np.percentile(data, 25):.2f}")
    print(f"50th percentile (Median): {np.percentile(data, 50):.2f}")
    print(f"75th percentile: {np.percentile(data, 75):.2f}")


# ============================================================================
# PROBABILITY DISTRIBUTIONS
# ============================================================================

def probability_distributions():
    """Demonstrate probability distributions"""
    print("\n" + "=" * 60)
    print("PROBABILITY DISTRIBUTIONS")
    print("=" * 60)
    
    # Normal Distribution
    print("\n1. Normal Distribution (μ=0, σ=1)")
    x = np.linspace(-4, 4, 1000)
    normal_pdf = stats.norm.pdf(x, 0, 1)
    print(f"   PDF at x=0: {stats.norm.pdf(0, 0, 1):.4f}")
    print(f"   CDF at x=0: {stats.norm.cdf(0, 0, 1):.4f}")
    
    # Binomial Distribution
    print("\n2. Binomial Distribution (n=10, p=0.5)")
    n, p = 10, 0.5
    print(f"   Mean: {n * p}")
    print(f"   Variance: {n * p * (1-p)}")
    print(f"   P(X=5): {stats.binom.pmf(5, n, p):.4f}")
    
    # Poisson Distribution
    print("\n3. Poisson Distribution (λ=3)")
    lambda_param = 3
    print(f"   Mean: {lambda_param}")
    print(f"   Variance: {lambda_param}")
    print(f"   P(X=3): {stats.poisson.pmf(3, lambda_param):.4f}")


# ============================================================================
# HYPOTHESIS TESTING - T-TEST
# ============================================================================

def hypothesis_testing_ttest():
    """Demonstrate t-tests"""
    print("\n" + "=" * 60)
    print("HYPOTHESIS TESTING - T-TESTS")
    print("=" * 60)
    
    np.random.seed(42)
    
    # One-sample t-test
    print("\n1. One-Sample T-Test")
    sample = np.random.normal(100, 15, 30)
    t_stat, p_value = stats.ttest_1samp(sample, 100)
    print(f"   Testing if sample mean = 100")
    print(f"   Sample mean: {np.mean(sample):.2f}")
    print(f"   T-statistic: {t_stat:.4f}")
    print(f"   P-value: {p_value:.4f}")
    print(f"   Result: {'Reject H0' if p_value < 0.05 else 'Fail to reject H0'} at α=0.05")
    
    # Independent samples t-test
    print("\n2. Independent Samples T-Test")
    group1 = np.random.normal(100, 15, 30)
    group2 = np.random.normal(105, 15, 30)
    t_stat, p_value = stats.ttest_ind(group1, group2)
    print(f"   Group 1 mean: {np.mean(group1):.2f}")
    print(f"   Group 2 mean: {np.mean(group2):.2f}")
    print(f"   T-statistic: {t_stat:.4f}")
    print(f"   P-value: {p_value:.4f}")
    print(f"   Result: {'Reject H0' if p_value < 0.05 else 'Fail to reject H0'} at α=0.05")
    
    # Paired samples t-test
    print("\n3. Paired Samples T-Test")
    before = np.random.normal(100, 15, 30)
    after = before + np.random.normal(5, 10, 30)
    t_stat, p_value = stats.ttest_rel(before, after)
    print(f"   Before mean: {np.mean(before):.2f}")
    print(f"   After mean: {np.mean(after):.2f}")
    print(f"   T-statistic: {t_stat:.4f}")
    print(f"   P-value: {p_value:.4f}")
    print(f"   Result: {'Reject H0' if p_value < 0.05 else 'Fail to reject H0'} at α=0.05")


# ============================================================================
# ANOVA
# ============================================================================

def anova_test():
    """Demonstrate ANOVA"""
    print("\n" + "=" * 60)
    print("ANALYSIS OF VARIANCE (ANOVA)")
    print("=" * 60)
    
    np.random.seed(42)
    
    # One-way ANOVA
    print("\nOne-Way ANOVA")
    group1 = np.random.normal(100, 15, 30)
    group2 = np.random.normal(105, 15, 30)
    group3 = np.random.normal(110, 15, 30)
    
    f_stat, p_value = stats.f_oneway(group1, group2, group3)
    
    print(f"Group 1 mean: {np.mean(group1):.2f}")
    print(f"Group 2 mean: {np.mean(group2):.2f}")
    print(f"Group 3 mean: {np.mean(group3):.2f}")
    print(f"F-statistic: {f_stat:.4f}")
    print(f"P-value: {p_value:.4f}")
    print(f"Result: {'Reject H0' if p_value < 0.05 else 'Fail to reject H0'} at α=0.05")


# ============================================================================
# CHI-SQUARE TEST
# ============================================================================

def chi_square_test():
    """Demonstrate Chi-Square test"""
    print("\n" + "=" * 60)
    print("CHI-SQUARE TEST")
    print("=" * 60)
    
    # Test of independence
    print("\nChi-Square Test of Independence")
    observed = np.array([[30, 10], [20, 40]])
    chi2, p_value, dof, expected = stats.chi2_contingency(observed)
    
    print("Observed frequencies:")
    print(observed)
    print("\nExpected frequencies:")
    print(expected)
    print(f"\nChi-square statistic: {chi2:.4f}")
    print(f"P-value: {p_value:.4f}")
    print(f"Degrees of freedom: {dof}")
    print(f"Result: {'Reject H0' if p_value < 0.05 else 'Fail to reject H0'} at α=0.05")


# ============================================================================
# CORRELATION
# ============================================================================

def correlation_analysis():
    """Demonstrate correlation analysis"""
    print("\n" + "=" * 60)
    print("CORRELATION ANALYSIS")
    print("=" * 60)
    
    np.random.seed(42)
    x = np.random.randn(100)
    y = 2 * x + np.random.randn(100) * 0.5
    
    # Pearson correlation
    pearson_r, pearson_p = stats.pearsonr(x, y)
    print("\nPearson Correlation:")
    print(f"Correlation coefficient (r): {pearson_r:.4f}")
    print(f"P-value: {pearson_p:.4f}")
    
    # Spearman correlation
    spearman_r, spearman_p = stats.spearmanr(x, y)
    print("\nSpearman Correlation:")
    print(f"Correlation coefficient (ρ): {spearman_r:.4f}")
    print(f"P-value: {spearman_p:.4f}")
    
    # Kendall correlation
    kendall_tau, kendall_p = stats.kendalltau(x, y)
    print("\nKendall Correlation:")
    print(f"Correlation coefficient (τ): {kendall_tau:.4f}")
    print(f"P-value: {kendall_p:.4f}")


# ============================================================================
# CONFIDENCE INTERVALS
# ============================================================================

def confidence_intervals():
    """Demonstrate confidence intervals"""
    print("\n" + "=" * 60)
    print("CONFIDENCE INTERVALS")
    print("=" * 60)
    
    np.random.seed(42)
    data = np.random.normal(100, 15, 30)
    
    # Calculate confidence interval
    mean = np.mean(data)
    std_err = stats.sem(data)
    confidence_level = 0.95
    
    # Using t-distribution
    ci = stats.t.interval(confidence_level, len(data)-1, loc=mean, scale=std_err)
    
    print(f"\nSample mean: {mean:.2f}")
    print(f"Standard error: {std_err:.2f}")
    print(f"95% Confidence Interval: ({ci[0]:.2f}, {ci[1]:.2f})")
    print(f"Interpretation: We are 95% confident that the true population mean")
    print(f"                lies between {ci[0]:.2f} and {ci[1]:.2f}")


# ============================================================================
# NON-PARAMETRIC TESTS
# ============================================================================

def non_parametric_tests():
    """Demonstrate non-parametric tests"""
    print("\n" + "=" * 60)
    print("NON-PARAMETRIC TESTS")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Mann-Whitney U Test
    print("\n1. Mann-Whitney U Test (alternative to independent t-test)")
    group1 = np.random.exponential(2, 30)
    group2 = np.random.exponential(2.5, 30)
    statistic, p_value = stats.mannwhitneyu(group1, group2)
    print(f"   Group 1 median: {np.median(group1):.2f}")
    print(f"   Group 2 median: {np.median(group2):.2f}")
    print(f"   U-statistic: {statistic:.4f}")
    print(f"   P-value: {p_value:.4f}")
    print(f"   Result: {'Reject H0' if p_value < 0.05 else 'Fail to reject H0'} at α=0.05")
    
    # Wilcoxon Signed-Rank Test
    print("\n2. Wilcoxon Signed-Rank Test (alternative to paired t-test)")
    before = np.random.exponential(2, 30)
    after = before + np.random.normal(0.5, 1, 30)
    statistic, p_value = stats.wilcoxon(before, after)
    print(f"   Before median: {np.median(before):.2f}")
    print(f"   After median: {np.median(after):.2f}")
    print(f"   W-statistic: {statistic:.4f}")
    print(f"   P-value: {p_value:.4f}")
    print(f"   Result: {'Reject H0' if p_value < 0.05 else 'Fail to reject H0'} at α=0.05")
    
    # Kruskal-Wallis Test
    print("\n3. Kruskal-Wallis Test (alternative to one-way ANOVA)")
    group1 = np.random.exponential(2, 30)
    group2 = np.random.exponential(2.5, 30)
    group3 = np.random.exponential(3, 30)
    statistic, p_value = stats.kruskal(group1, group2, group3)
    print(f"   Group 1 median: {np.median(group1):.2f}")
    print(f"   Group 2 median: {np.median(group2):.2f}")
    print(f"   Group 3 median: {np.median(group3):.2f}")
    print(f"   H-statistic: {statistic:.4f}")
    print(f"   P-value: {p_value:.4f}")
    print(f"   Result: {'Reject H0' if p_value < 0.05 else 'Fail to reject H0'} at α=0.05")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    descriptive_statistics()
    probability_distributions()
    hypothesis_testing_ttest()
    anova_test()
    chi_square_test()
    correlation_analysis()
    confidence_intervals()
    non_parametric_tests()
    
    print("\n" + "=" * 60)
    print("Statistics basics demonstration complete!")
    print("=" * 60)
