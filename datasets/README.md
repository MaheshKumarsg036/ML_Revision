# Datasets 📊

This folder contains sample datasets used throughout the tutorials and examples in this repository.

## Available Datasets

Most examples use built-in datasets from libraries like:
- **Scikit-learn datasets:** iris, digits, boston, wine, breast_cancer
- **Seaborn datasets:** tips, titanic, diamonds, penguins
- **Pandas datasets:** Various CSV files for practice

## Adding Custom Datasets

You can add your own datasets to this folder:

1. Place CSV files, Excel files, or other data files here
2. Use descriptive names for your datasets
3. Consider adding a data dictionary or description

## Usage Example

```python
import pandas as pd

# Load a dataset
df = pd.read_csv('datasets/your_dataset.csv')
```

## Popular Dataset Sources

- **Kaggle:** [kaggle.com/datasets](https://www.kaggle.com/datasets)
- **UCI ML Repository:** [archive.ics.uci.edu/ml](https://archive.ics.uci.edu/ml)
- **Google Dataset Search:** [datasetsearch.research.google.com](https://datasetsearch.research.google.com)
- **Data.gov:** [data.gov](https://data.gov)
- **AWS Open Data:** [registry.opendata.aws](https://registry.opendata.aws)

## Note

Large datasets (>10MB) should not be committed to the repository. Instead, provide download links or scripts to fetch them.
