"""
Matplotlib Basics - Data Visualization
======================================

This module demonstrates basic Matplotlib visualizations.
"""

import matplotlib.pyplot as plt
import numpy as np


def line_plot_example():
    """Create a basic line plot."""
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label='sin(x)', color='blue', linewidth=2)
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.title('Line Plot Example')
    plt.legend()
    plt.grid(True)
    plt.savefig('/tmp/line_plot.png')
    plt.close()
    print("Line plot saved to /tmp/line_plot.png")


def scatter_plot_example():
    """Create a scatter plot."""
    np.random.seed(42)
    x = np.random.randn(50)
    y = np.random.randn(50)
    colors = np.random.rand(50)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, c=colors, cmap='viridis', s=100, alpha=0.6)
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.title('Scatter Plot Example')
    plt.colorbar(label='Color scale')
    plt.savefig('/tmp/scatter_plot.png')
    plt.close()
    print("Scatter plot saved to /tmp/scatter_plot.png")


def bar_chart_example():
    """Create a bar chart."""
    categories = ['A', 'B', 'C', 'D', 'E']
    values = [23, 45, 56, 78, 32]
    
    plt.figure(figsize=(10, 6))
    plt.bar(categories, values, color='steelblue')
    plt.xlabel('Categories')
    plt.ylabel('Values')
    plt.title('Bar Chart Example')
    plt.savefig('/tmp/bar_chart.png')
    plt.close()
    print("Bar chart saved to /tmp/bar_chart.png")


def histogram_example():
    """Create a histogram."""
    np.random.seed(42)
    data = np.random.randn(1000)
    
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=30, color='green', alpha=0.7, edgecolor='black')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram Example')
    plt.savefig('/tmp/histogram.png')
    plt.close()
    print("Histogram saved to /tmp/histogram.png")


def subplot_example():
    """Create multiple subplots."""
    x = np.linspace(0, 10, 100)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].plot(x, np.sin(x))
    axes[0, 0].set_title('sin(x)')
    
    axes[0, 1].plot(x, np.cos(x), 'r')
    axes[0, 1].set_title('cos(x)')
    
    axes[1, 0].plot(x, np.tan(x))
    axes[1, 0].set_title('tan(x)')
    axes[1, 0].set_ylim([-5, 5])
    
    axes[1, 1].plot(x, x**2)
    axes[1, 1].set_title('x²')
    
    plt.tight_layout()
    plt.savefig('/tmp/subplots.png')
    plt.close()
    print("Subplots saved to /tmp/subplots.png")


def main():
    """Run all examples."""
    print("Creating Matplotlib examples...")
    line_plot_example()
    scatter_plot_example()
    bar_chart_example()
    histogram_example()
    subplot_example()
    print("\nAll plots saved successfully!")


if __name__ == "__main__":
    main()
