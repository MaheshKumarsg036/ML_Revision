# Pandas Mastery Guide

A thorough, interview-ready walkthrough of essential pandas concepts. Each section explains the idea first, then demonstrates it with focused, reproducible code snippets.

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(seed=42)
```

**Why `rng`?** The `default_rng` generator keeps random number usage reproducible during practice or interviews without polluting global state.

---

## 1. Reading and Writing Tabular Data

Pandas supports many formats, but interviews most often focus on CSV, Excel, and Parquet. Remember that readers infer schema, handle missing values, and can load data incrementally via chunks.

- **CSV**: Plain-text tables separated by commas (or alternative delimiters).
- **Excel**: Spreadsheets with multiple sheets; requires `openpyxl` or similar engine.
- **Parquet**: Columnar binary format, efficient for analytics and large datasets.

```python
csv_path = "datasets/customers.csv"
excel_path = "datasets/orders.xlsx"
parquet_path = "datasets/transactions.parquet"

customers = pd.read_csv(csv_path, dtype={"customer_id": "int64"})
orders = pd.read_excel(excel_path, sheet_name="2025", engine="openpyxl")
transactions = pd.read_parquet(parquet_path)

# Write artifacts back out (common interview follow-up)
customers.to_csv("output/customers_clean.csv", index=False)
orders.to_excel("output/orders_2025.xlsx", sheet_name="report", index=False)
transactions.to_parquet("output/transactions_subset.parquet", compression="snappy")
```

---

## 2. Core Data Structures: Series and DataFrame

- **Series**: One-dimensional labeled array; under the hood this wraps a NumPy array and an index.
- **DataFrame**: Two-dimensional labeled table with columns of potentially different types.

```python
series_example = pd.Series(["Widget", "Gadget", "Device"], name="product")
series_example.index  # RangeIndex(0, 3)

frame_example = pd.DataFrame(
    {
        "product": ["Widget", "Gadget", "Device"],
        "price": [15.5, 23.0, 9.75],
        "in_stock": [True, False, True],
    }
)
frame_example.dtypes
```

Know how indexes align during operations: adding two Series automatically aligns on the index labels.

---

## 3. Information Extraction and Quick Diagnostics

Use these methods to understand the structure and quality of your dataset quickly.

```python
frame_example.head(2)     # first rows
frame_example.tail(1)     # last row
frame_example.shape       # (rows, columns)
frame_example.describe()  # numeric summary (use include="all" for mixed data)
frame_example.info()      # column dtypes + non-null counts
frame_example.memory_usage(deep=True)
```

Mention `df.sample(n=5, random_state=0)` when asked about quick random inspections.

---

## 4. Accessing Rows, Columns, Slicing, Masking, and Filtering

- **Column access**: `df["col"]`, `df.col`, or `df[["col1", "col2"]]` for multiple columns.
- **Row access**: `.loc` for label-based selection, `.iloc` for positional selection.
- **Masking/filtering**: Boolean expressions generate masks you can use with `.loc`.

```python
cities = pd.DataFrame(
    {
        "city": ["Paris", "Berlin", "Madrid", "Rome"],
        "population_millions": [2.1, 3.6, 3.2, 2.9],
        "country": ["France", "Germany", "Spain", "Italy"],
    }
)

# Columns
cities["city"]
cities[["city", "country"]]

# Rows
cities_named = cities.set_index("city")
cities_named.loc["Berlin":"Rome"]
cities.iloc[1:3]

# Slicing
cities.iloc[::2, :2]

# Masking / filtering
population_mask = cities["population_millions"] > 3.0
large_cities = cities.loc[population_mask, ["city", "population_millions"]]

# Query-based filtering (handy in SQL-style questions)
large_german = cities.query("country == 'Germany' and population_millions > 3")
```

Explain chain assignment issues and recommend `.loc[...] = ...` for updates.

---

## 5. Data Manipulation: Adding, Deleting, Renaming, and Deduplicating

- **Add columns** via assignment or `assign` (latter keeps chaining clean).
- **Delete columns** with `del`, `drop`, or in-place modifications.
- **Rename columns** using `rename(columns={...})` or `set_axis`.
- **Duplicates**: identify with `duplicated`, `drop_duplicates`, or custom logic for first duplicates.

```python
sales = pd.DataFrame(
    {
        "order_id": [101, 102, 103, 103],
        "product": ["Widget", "Widget", "Gadget", "Gadget"],
        "quantity": [5, 7, 3, 3],
    }
)

# Adding
sales["total_price"] = sales["quantity"] * 10
sales = sales.assign(order_type=np.where(sales["quantity"] > 5, "bulk", "regular"))

# Deleting
sales = sales.drop(columns=["order_type"])
# or: del sales["total_price"]

# Renaming
sales = sales.rename(columns={"product": "product_name"})

# Duplicate detection
sales["is_duplicate"] = sales.duplicated(subset=["order_id", "product_name"], keep=False)
first_duplicates = sales[sales.duplicated(subset="order_id", keep="first")]
unique_sales = sales.drop_duplicates(subset=["order_id", "product_name"], keep="first")
```

Highlight that `keep=False` flags all duplicates, while `keep="first"` or `"last"` isolates rows beyond the first occurrence.

---

## 6. Descriptive Operations and Custom Functions

Pandas mirrors many SQL aggregate and analytic operations.

```python
metrics = pd.DataFrame(
    {
        "department": ["Sales", "Sales", "Marketing", "HR"],
        "revenue": [120_000, 150_000, 80_000, 50_000],
        "employees": [10, 14, 6, 4],
    }
)

metrics.sort_values(by="revenue", ascending=False)
metrics["revenue"].min(), metrics["revenue"].max()
metrics.count()  # non-null counts per column
metrics.agg({"revenue": ["sum", "mean"], "employees": "median"})

# Apply for row-level logic (emphasize vectorization preference when possible)
def revenue_per_employee(row: pd.Series) -> float:
    return row["revenue"] / row["employees"]

metrics["rev_per_emp"] = metrics.apply(revenue_per_employee, axis=1)
```

Mention `.pipe` and `.assign` for clean method chains if asked about readable pipelines.

---

## 7. Concatenation (Stacking Data Vertically or Horizontally)

`pd.concat` stacks DataFrames along rows or columns, optionally preserving a hierarchical index.

```python
north = pd.DataFrame({"region": ["North"], "revenue": [1200]})
south = pd.DataFrame({"region": ["South"], "revenue": [900]})

vertical_stack = pd.concat([north, south], ignore_index=True)
horizontal_stack = pd.concat([north.set_index("region"), south.set_index("region")], axis=1)
hierarchical_stack = pd.concat([north, south], keys=["Q1", "Q2"])
```

Remember to reset indexes after concatenation if you need a clean RangeIndex.

---

## 8. Relational Joins with `merge`

`merge` aligns data on key columns, mirroring SQL `JOIN` semantics. Set `validate` to catch data duplication mistakes.

```python
customers = pd.DataFrame(
    {
        "customer_id": [1, 2, 3],
        "segment": ["Enterprise", "SMB", "Consumer"],
    }
)
orders = pd.DataFrame(
    {
        "order_id": [101, 102, 103, 104],
        "customer_id": [1, 1, 2, 4],
        "amount": [500, 250, 90, 120],
    }
)

inner_join = customers.merge(orders, on="customer_id", how="inner", validate="1:m")
left_join = customers.merge(orders, on="customer_id", how="left", indicator=True)
```

Discuss `how` options (`inner`, `left`, `right`, `outer`) and the use of `indicator=True` to audit matched/unmatched rows.

---

## 9. GroupBy: Aggregation, Transformation, Filtering, Apply

GroupBy splits data into groups, applies a function, then combines the results.

```python
dept = pd.DataFrame(
    {
        "department": ["Sales", "Sales", "Marketing", "Marketing", "HR"],
        "salary": [80_000, 75_000, 90_000, 85_000, 60_000],
        "tenure": [2, 5, 3, 6, 4],
    }
)

# Single aggregation
avg_salary = dept.groupby("department")["salary"].mean()

# Multiple aggregations per column
summary = (
    dept.groupby("department")
    .agg(salary_avg=("salary", "mean"), salary_max=("salary", "max"), tenure_sum=("tenure", "sum"))
)

# Transform keeps original shape
salary_z = dept["salary"] - dept.groupby("department")["salary"].transform("mean")

# Filter drops entire groups based on condition
senior_teams = dept.groupby("department").filter(lambda g: g["tenure"].mean() > 3)

# Apply allows custom logic per group
medians = dept.groupby("department").apply(lambda g: g["salary"].median())
```

Clarify when to use `agg` versus `transform` versus `filter` versus `apply`, a common interview follow-up.

---

## 10. Data Cleaning: Handling Missing Values and Nulls

Detect nulls with `isna`/`isnull`, summarize with `sum`, and decide between dropping or imputing.

```python
raw = pd.DataFrame(
    {
        "temperature": [21.0, np.nan, 18.5, 19.0],
        "status": ["ok", "ok", None, "fail"],
    }
)

null_counts = raw.isna().sum()
rows_with_nulls = raw[raw.isna().any(axis=1)]

filled = raw.fillna({"temperature": raw["temperature"].mean(), "status": "unknown"})
dropped = raw.dropna(subset=["status"], how="any")
```

Mention `interpolate`, `ffill`, `bfill` for time-series interpolation if prompted.

---

## 11. Data Restructuring: Melt, Pivot, Cut, and Shift

Reshape data to fit modeling or reporting requirements.

```python
wide = pd.DataFrame(
    {
        "store": ["A", "B"],
        "2024": [100, 120],
        "2025": [110, 130],
    }
)

# Melt to long format
long = wide.melt(id_vars="store", var_name="year", value_name="sales")

# Pivot back to wide
pivoted = long.pivot(index="store", columns="year", values="sales")

# Cut: bin continuous data into categories
sales_amounts = pd.Series([50, 120, 200, 350, 500])
bins = pd.cut(sales_amounts, bins=[0, 100, 250, 500], labels=["low", "medium", "high"])

# Shift: align data with previous periods (time-series)
profits = pd.Series([10, 12, 9, 14, 16], index=pd.date_range("2024-01-01", periods=5, freq="M"))
profits_prev_month = profits.shift(1)
profit_growth = profits - profits_prev_month
```

Explain that `pd.cut` creates categorical bins and `shift` aligns with prior rows to compute deltas.

---

## 12. Datetime Mastery

Convert text to datetime, extract components, resample, and adjust time zones.

```python
orders = pd.DataFrame(
    {
        "order_id": range(4),
        "ordered_at": ["2024-01-05 10:15", "2024-01-06 12:30", "2024-01-06 14:05", "2024-01-08 09:20"],
        "amount": [120, 90, 150, 200],
    }
)

orders["ordered_at"] = pd.to_datetime(orders["ordered_at"])
orders["day_name"] = orders["ordered_at"].dt.day_name()
orders.set_index("ordered_at", inplace=True)

daily_totals = orders.resample("D")["amount"].sum().fillna(0)
weekly_totals = orders.resample("W")["amount"].sum()
tz_converted = orders.tz_localize("UTC").tz_convert("US/Eastern")
```

Remember `.dt` accessor works on Series of datetimes; `resample` requires a DatetimeIndex.

---

## 13. Vectorized String Operations

Pandas string methods (`.str`) operate element-wise, handle missing values, and simplify text munging.

```python
emails = pd.Series([
    "alice@example.com",
    "bob@company.org",
    "carol@university.edu",
    None,
])

domains = emails.str.split("@").str[1]
edu_mask = emails.str.contains(".edu", regex=False, na=False)
cleaned = emails.str.strip().str.lower()
```

Mention regex support (`str.extract`, `str.replace`) and translation to complex parsing tasks.

---

## 14. Putting It Together: Mini Workflow

```python
people = pd.DataFrame(
    {
        "name": ["Alice", "Bob", "Carol", "Dan", "Eve"],
        "dept": ["Sales", "Sales", "HR", "Marketing", "HR"],
        "salary": [80_000, 75_000, 60_000, 90_000, np.nan],
        "hired": pd.to_datetime(["2020-01-15", "2019-05-20", "2021-08-01", "2018-03-12", "2020-10-05"]),
    }
)

# Clean
people["salary"] = people["salary"].fillna(people["salary"].median())
people["tenure_years"] = (pd.Timestamp("2025-01-01") - people["hired"]).dt.days / 365

# Feature engineering
people["salary_band"] = pd.cut(people["salary"], bins=[0, 70000, 90000, 120000], labels=["low", "mid", "high"])
people["previous_salary"] = people["salary"].shift()

# Aggregate report
report = (
    people.groupby("dept")
    .agg(avg_salary=("salary", "mean"), headcount=("name", "count"), median_tenure=("tenure_years", "median"))
    .sort_values("avg_salary", ascending=False)
)
```

Practice narrating each transformation—you will be evaluated on clarity as much as correctness.

---

## 15. Quick Reference Prompts

- Detect and remove the first duplicate order per customer.
- Read partitioned Parquet data, combine with `pd.concat`, and compute monthly KPIs.
- Bin customer lifetime values with `pd.cut`, then pivot to analyze retentions.
- Merge customer and transaction tables, highlight unmatched keys with `indicator`.
- Build a daily active user metric using `groupby`, `shift`, and boolean masks.

Use these prompts to reinforce muscle memory before interviews.
