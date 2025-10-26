"""
Seaborn Basics - Statistical Data Visualization
"""

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set seaborn style
sns.set_theme(style="whitegrid")

# ============================================================================
# DISTRIBUTION PLOTS
# ============================================================================

def distribution_plots():
    """Demonstrate distribution plots"""
    print("Creating distribution plots...")
    
    # Generate sample data
    np.random.seed(42)
    data = np.random.randn(1000)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Histogram
    sns.histplot(data, kde=True, ax=axes[0, 0])
    axes[0, 0].set_title('Histogram with KDE')
    
    # KDE Plot
    sns.kdeplot(data, ax=axes[0, 1])
    axes[0, 1].set_title('KDE Plot')
    
    # Box Plot
    sns.boxplot(y=data, ax=axes[1, 0])
    axes[1, 0].set_title('Box Plot')
    
    # Violin Plot
    sns.violinplot(y=data, ax=axes[1, 1])
    axes[1, 1].set_title('Violin Plot')
    
    plt.tight_layout()
    plt.savefig('/tmp/seaborn_distribution.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("Saved: /tmp/seaborn_distribution.png")


# ============================================================================
# CATEGORICAL PLOTS
# ============================================================================

def categorical_plots():
    """Demonstrate categorical plots"""
    print("\nCreating categorical plots...")
    
    # Generate sample data
    tips = sns.load_dataset('tips')
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Bar Plot
    sns.barplot(data=tips, x='day', y='total_bill', ax=axes[0, 0])
    axes[0, 0].set_title('Bar Plot: Average Bill by Day')
    
    # Count Plot
    sns.countplot(data=tips, x='day', ax=axes[0, 1])
    axes[0, 1].set_title('Count Plot: Count by Day')
    
    # Box Plot by Category
    sns.boxplot(data=tips, x='day', y='total_bill', ax=axes[1, 0])
    axes[1, 0].set_title('Box Plot by Day')
    
    # Violin Plot by Category
    sns.violinplot(data=tips, x='day', y='total_bill', ax=axes[1, 1])
    axes[1, 1].set_title('Violin Plot by Day')
    
    plt.tight_layout()
    plt.savefig('/tmp/seaborn_categorical.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("Saved: /tmp/seaborn_categorical.png")


# ============================================================================
# RELATIONAL PLOTS
# ============================================================================

def relational_plots():
    """Demonstrate relational plots"""
    print("\nCreating relational plots...")
    
    # Load sample data
    tips = sns.load_dataset('tips')
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scatter Plot
    sns.scatterplot(data=tips, x='total_bill', y='tip', 
                    hue='time', size='size', ax=axes[0])
    axes[0].set_title('Scatter Plot: Bill vs Tip')
    
    # Line Plot (using different dataset)
    flights = sns.load_dataset('flights')
    flights_subset = flights[flights['year'].isin([1949, 1950, 1951])]
    sns.lineplot(data=flights_subset, x='month', y='passengers', 
                 hue='year', marker='o', ax=axes[1])
    axes[1].set_title('Line Plot: Passengers over Months')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('/tmp/seaborn_relational.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("Saved: /tmp/seaborn_relational.png")


# ============================================================================
# HEATMAP
# ============================================================================

def heatmap_example():
    """Demonstrate heatmap"""
    print("\nCreating heatmap...")
    
    # Generate correlation matrix
    np.random.seed(42)
    data = pd.DataFrame(
        np.random.randn(100, 5),
        columns=['A', 'B', 'C', 'D', 'E']
    )
    correlation = data.corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, cmap='coolwarm', 
                center=0, square=True, linewidths=1)
    plt.title('Correlation Heatmap')
    plt.savefig('/tmp/seaborn_heatmap.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("Saved: /tmp/seaborn_heatmap.png")


# ============================================================================
# REGRESSION PLOTS
# ============================================================================

def regression_plots():
    """Demonstrate regression plots"""
    print("\nCreating regression plots...")
    
    tips = sns.load_dataset('tips')
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Regression Plot
    sns.regplot(data=tips, x='total_bill', y='tip', ax=axes[0])
    axes[0].set_title('Regression Plot: Bill vs Tip')
    
    # Residual Plot
    sns.residplot(data=tips, x='total_bill', y='tip', ax=axes[1])
    axes[1].set_title('Residual Plot')
    
    plt.tight_layout()
    plt.savefig('/tmp/seaborn_regression.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("Saved: /tmp/seaborn_regression.png")


# ============================================================================
# PAIR PLOT
# ============================================================================

def pair_plot_example():
    """Demonstrate pair plot"""
    print("\nCreating pair plot...")
    
    # Load iris dataset
    iris = sns.load_dataset('iris')
    
    # Create pair plot
    pair_grid = sns.pairplot(iris, hue='species', diag_kind='kde')
    pair_grid.fig.suptitle('Iris Dataset Pair Plot', y=1.02)
    plt.savefig('/tmp/seaborn_pairplot.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("Saved: /tmp/seaborn_pairplot.png")


# ============================================================================
# JOINT PLOT
# ============================================================================

def joint_plot_example():
    """Demonstrate joint plot"""
    print("\nCreating joint plot...")
    
    tips = sns.load_dataset('tips')
    
    # Create joint plot
    joint_grid = sns.jointplot(data=tips, x='total_bill', y='tip', 
                               kind='reg', height=8)
    joint_grid.fig.suptitle('Joint Plot: Bill vs Tip', y=1.02)
    plt.savefig('/tmp/seaborn_jointplot.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("Saved: /tmp/seaborn_jointplot.png")


# ============================================================================
# FACET GRID
# ============================================================================

def facet_grid_example():
    """Demonstrate facet grid"""
    print("\nCreating facet grid...")
    
    tips = sns.load_dataset('tips')
    
    # Create facet grid
    g = sns.FacetGrid(tips, col='time', row='sex', height=4, aspect=1.2)
    g.map(sns.scatterplot, 'total_bill', 'tip')
    g.add_legend()
    g.fig.suptitle('Facet Grid: Bill vs Tip by Time and Sex', y=1.02)
    plt.savefig('/tmp/seaborn_facetgrid.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("Saved: /tmp/seaborn_facetgrid.png")


# ============================================================================
# STYLING
# ============================================================================

def styling_example():
    """Demonstrate different styles"""
    print("\nCreating styling examples...")
    
    styles = ['darkgrid', 'whitegrid', 'dark', 'white']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    np.random.seed(42)
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    
    for idx, style in enumerate(styles):
        sns.set_style(style)
        ax = axes[idx]
        sns.lineplot(x=x, y=y, ax=ax)
        ax.set_title(f'Style: {style}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
    
    plt.tight_layout()
    plt.savefig('/tmp/seaborn_styles.png', dpi=100, bbox_inches='tight')
    plt.close()
    
    # Reset to default style
    sns.set_theme(style="whitegrid")
    print("Saved: /tmp/seaborn_styles.png")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SEABORN VISUALIZATION EXAMPLES")
    print("=" * 60)
    
    distribution_plots()
    categorical_plots()
    relational_plots()
    heatmap_example()
    regression_plots()
    pair_plot_example()
    joint_plot_example()
    facet_grid_example()
    styling_example()
    
    print("\n" + "=" * 60)
    print("All plots saved to /tmp/ directory")
    print("Seaborn basics demonstration complete!")
    print("=" * 60)
