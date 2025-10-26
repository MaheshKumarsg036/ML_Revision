"""
Matplotlib Basics - Data Visualization
"""

import matplotlib.pyplot as plt
import numpy as np

# ============================================================================
# BASIC LINE PLOT
# ============================================================================

def basic_line_plot():
    """Demonstrate basic line plotting"""
    print("Creating basic line plot...")
    
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y)
    plt.title('Basic Line Plot: Sine Wave')
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.grid(True)
    plt.savefig('/tmp/basic_line_plot.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("Saved: /tmp/basic_line_plot.png")


# ============================================================================
# MULTIPLE LINES
# ============================================================================

def multiple_lines():
    """Demonstrate plotting multiple lines"""
    print("\nCreating multiple lines plot...")
    
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    y3 = np.sin(x) * np.cos(x)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y1, label='sin(x)', color='blue', linestyle='-', linewidth=2)
    plt.plot(x, y2, label='cos(x)', color='red', linestyle='--', linewidth=2)
    plt.plot(x, y3, label='sin(x)*cos(x)', color='green', linestyle='-.', linewidth=2)
    
    plt.title('Multiple Lines Plot')
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('/tmp/multiple_lines.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("Saved: /tmp/multiple_lines.png")


# ============================================================================
# SCATTER PLOT
# ============================================================================

def scatter_plot():
    """Demonstrate scatter plotting"""
    print("\nCreating scatter plot...")
    
    np.random.seed(42)
    x = np.random.randn(100)
    y = 2 * x + np.random.randn(100) * 0.5
    colors = np.random.rand(100)
    sizes = np.random.randint(10, 200, 100)
    
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(x, y, c=colors, s=sizes, alpha=0.6, cmap='viridis')
    plt.colorbar(scatter, label='Color scale')
    plt.title('Scatter Plot with Variable Colors and Sizes')
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.grid(True, alpha=0.3)
    plt.savefig('/tmp/scatter_plot.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("Saved: /tmp/scatter_plot.png")


# ============================================================================
# BAR PLOT
# ============================================================================

def bar_plot():
    """Demonstrate bar plotting"""
    print("\nCreating bar plot...")
    
    categories = ['A', 'B', 'C', 'D', 'E']
    values = [25, 40, 30, 55, 45]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(categories, values, color=['red', 'blue', 'green', 'orange', 'purple'])
    plt.title('Bar Plot Example')
    plt.xlabel('Categories')
    plt.ylabel('Values')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height}',
                ha='center', va='bottom')
    
    plt.savefig('/tmp/bar_plot.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("Saved: /tmp/bar_plot.png")


# ============================================================================
# HISTOGRAM
# ============================================================================

def histogram():
    """Demonstrate histogram"""
    print("\nCreating histogram...")
    
    np.random.seed(42)
    data = np.random.randn(1000)
    
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title('Histogram: Normal Distribution')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3, axis='y')
    plt.savefig('/tmp/histogram.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("Saved: /tmp/histogram.png")


# ============================================================================
# PIE CHART
# ============================================================================

def pie_chart():
    """Demonstrate pie chart"""
    print("\nCreating pie chart...")
    
    labels = ['Python', 'Java', 'JavaScript', 'C++', 'Others']
    sizes = [35, 25, 20, 10, 10]
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc']
    explode = (0.1, 0, 0, 0, 0)  # explode 1st slice
    
    plt.figure(figsize=(10, 8))
    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=90)
    plt.title('Programming Languages Distribution')
    plt.axis('equal')
    plt.savefig('/tmp/pie_chart.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("Saved: /tmp/pie_chart.png")


# ============================================================================
# BOX PLOT
# ============================================================================

def box_plot():
    """Demonstrate box plot"""
    print("\nCreating box plot...")
    
    np.random.seed(42)
    data = [np.random.normal(0, std, 100) for std in range(1, 5)]
    
    plt.figure(figsize=(10, 6))
    plt.boxplot(data, labels=['Group 1', 'Group 2', 'Group 3', 'Group 4'])
    plt.title('Box Plot Example')
    plt.ylabel('Values')
    plt.grid(True, alpha=0.3, axis='y')
    plt.savefig('/tmp/box_plot.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("Saved: /tmp/box_plot.png")


# ============================================================================
# SUBPLOTS
# ============================================================================

def subplots_example():
    """Demonstrate subplots"""
    print("\nCreating subplots...")
    
    x = np.linspace(0, 10, 100)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Sine
    axes[0, 0].plot(x, np.sin(x), 'b-')
    axes[0, 0].set_title('Sine Wave')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Cosine
    axes[0, 1].plot(x, np.cos(x), 'r-')
    axes[0, 1].set_title('Cosine Wave')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Tangent (limited range)
    axes[1, 0].plot(x, np.tan(x), 'g-')
    axes[1, 0].set_title('Tangent Wave')
    axes[1, 0].set_ylim(-5, 5)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Exponential
    axes[1, 1].plot(x, np.exp(x/5), 'm-')
    axes[1, 1].set_title('Exponential')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/tmp/subplots.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("Saved: /tmp/subplots.png")


# ============================================================================
# HEATMAP
# ============================================================================

def heatmap():
    """Demonstrate heatmap"""
    print("\nCreating heatmap...")
    
    np.random.seed(42)
    data = np.random.rand(10, 10)
    
    plt.figure(figsize=(10, 8))
    im = plt.imshow(data, cmap='hot', interpolation='nearest')
    plt.colorbar(im, label='Intensity')
    plt.title('Heatmap Example')
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.savefig('/tmp/heatmap.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("Saved: /tmp/heatmap.png")


# ============================================================================
# CUSTOMIZATION
# ============================================================================

def customization_example():
    """Demonstrate advanced customization"""
    print("\nCreating customized plot...")
    
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    
    plt.figure(figsize=(12, 6))
    plt.plot(x, y, color='#2E86AB', linewidth=3, linestyle='-', 
             marker='o', markevery=10, markersize=8, markerfacecolor='red',
             label='sin(x)')
    
    plt.title('Highly Customized Plot', fontsize=20, fontweight='bold')
    plt.xlabel('X axis', fontsize=14, fontweight='bold')
    plt.ylabel('Y axis', fontsize=14, fontweight='bold')
    plt.legend(fontsize=12, loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Add annotation
    plt.annotate('Maximum', xy=(np.pi/2, 1), xytext=(np.pi/2 + 1, 1.2),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=12, color='red')
    
    plt.xlim(0, 10)
    plt.ylim(-1.5, 1.5)
    
    plt.savefig('/tmp/customized_plot.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("Saved: /tmp/customized_plot.png")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("MATPLOTLIB VISUALIZATION EXAMPLES")
    print("=" * 60)
    
    basic_line_plot()
    multiple_lines()
    scatter_plot()
    bar_plot()
    histogram()
    pie_chart()
    box_plot()
    subplots_example()
    heatmap()
    customization_example()
    
    print("\n" + "=" * 60)
    print("All plots saved to /tmp/ directory")
    print("Matplotlib basics demonstration complete!")
    print("=" * 60)
