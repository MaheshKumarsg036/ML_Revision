"""
Seaborn Examples - Statistical Visualizations
=============================================

This module demonstrates Seaborn visualizations.
"""

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def distribution_plots():
    """Create distribution plots."""
    np.random.seed(42)
    data = np.random.randn(1000)
    
    # Histogram with KDE
    plt.figure(figsize=(10, 6))
    sns.histplot(data, kde=True, color='skyblue')
    plt.title('Distribution Plot with KDE')
    plt.savefig('/tmp/seaborn_dist.png')
    plt.close()
    print("Distribution plot saved to /tmp/seaborn_dist.png")


def categorical_plots():
    """Create categorical plots."""
    df = pd.DataFrame({
        'Category': ['A', 'B', 'C', 'A', 'B', 'C'] * 10,
        'Value': np.random.randn(60) * 10 + 50
    })
    
    # Box plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Category', y='Value', data=df)
    plt.title('Box Plot Example')
    plt.savefig('/tmp/seaborn_box.png')
    plt.close()
    print("Box plot saved to /tmp/seaborn_box.png")


def heatmap_example():
    """Create a heatmap."""
    data = np.random.rand(10, 10)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(data, annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Heatmap Example')
    plt.savefig('/tmp/seaborn_heatmap.png')
    plt.close()
    print("Heatmap saved to /tmp/seaborn_heatmap.png")


def pairplot_example():
    """Create a pairplot."""
    iris = sns.load_dataset('iris')
    
    sns.pairplot(iris, hue='species')
    plt.savefig('/tmp/seaborn_pairplot.png')
    plt.close()
    print("Pairplot saved to /tmp/seaborn_pairplot.png")


def main():
    """Run all examples."""
    print("Creating Seaborn examples...")
    sns.set_style('whitegrid')
    distribution_plots()
    categorical_plots()
    heatmap_example()
    pairplot_example()
    print("\nAll Seaborn plots saved successfully!")


if __name__ == "__main__":
    main()
