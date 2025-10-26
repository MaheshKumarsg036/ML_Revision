# NumPy Mastery Guide

A comprehensive, interview-ready walkthrough of NumPy fundamentals, idioms, and advanced techniques. Use this guide as a structured revision plan: read a section, run the snippets, then tackle the practice prompts.

---

## 1. Why NumPy Still Matters

- Backbone of the Python scientific stack (pandas, scikit-learn, SciPy, PyTorch all depend on it).
- Provides the high-performance `ndarray` along with vectorized operations that eliminate Python loops.
- Critical for interviews that test analytical thinking, optimization, and foundational machine learning math.

**Setup Tip**: Always start your scripts or notebooks with a deterministic random generator so results are reproducible during interviews.

```python
import numpy as np
rng = np.random.default_rng(seed=42)
```

**Why `rng`?** The `default_rng` constructor returns a Generator object that owns its own random state. Using an explicit generator instead of the legacy `np.random` module keeps randomness reproducible (handy for interviews) and prevents accidental cross-talk between separate scripts or tests.

---

## 2. ndarray Essentials

### 2.1 Creating Arrays Quickly

| Goal | Function | Snippet |
| ---- | -------- | ------- |
| From Python data | `np.array` | `np.array([[1, 2], [3, 4]], dtype=np.float32)` |
| Filled with zeros/ones | `np.zeros`, `np.ones`, `np.full` | `np.zeros((3, 4), dtype=int)` |
| Structured sequences | `np.arange`, `np.linspace`, `np.logspace` | `np.linspace(0, 1, num=5)` |
| Identity-like matrices | `np.eye`, `np.diag` | `np.diag([1, 2, 3])` |
| Random samples | `rng.normal`, `rng.uniform`, `rng.integers` | `rng.normal(size=(2, 3))` |

```python
vector = np.arange(0, 10, 2)  # [0 2 4 6 8]
angles = np.linspace(0, 2 * np.pi, num=100)
identity = np.eye(4)
```

**Interview Check**: Given a Python list of lists, create a float64 array and compute the row-wise mean without loops.

```python
values = [[1, 2, 3], [4, 5, 6]]
arr = np.array(values, dtype=np.float64)
row_means = arr.mean(axis=1)
```

### 2.2 Inspecting Arrays

```python
arr = rng.integers(low=0, high=10, size=(3, 4))
print(arr.shape)   # (3, 4)
print(arr.ndim)    # 2
print(arr.dtype)   # int64 (platform dependent)
print(arr.size)    # 12
print(arr.strides) # bytes to step per axis
```

`strides` matter when reasoning about views vs copies—an interview favorite.

### 2.3 Views vs Copies

- Slicing returns a **view** (shares memory).
- Boolean/fancy indexing returns a **copy**.
- Use `.copy()` explicitly when you must isolate mutations.

```python
base = np.arange(6).reshape(2, 3)
view = base[:, 1:]      # view
copy = base[:, [0, 2]].copy()  # copy
view[0, 0] = 999
```

After modifying `view`, the original `base` is affected, but `copy` is not.

---

## 3. Indexing & Slicing Patterns

### 3.1 Basic Slicing

```python
matrix = np.arange(1, 17).reshape(4, 4)
sub = matrix[1:3, 0:2]
row = matrix[2, :]
col = matrix[:, 3]
step_slice = matrix[::2, ::2]
```

### 3.2 Boolean Indexing

```python
scores = rng.normal(loc=70, scale=10, size=10)
above_avg = scores[scores > scores.mean()]
```

### 3.3 Fancy Indexing

```python
people = np.array(["Alice", "Bob", "Carol", "Dan"])
idxs = [3, 0, 1]
selected = people[idxs]
```

Combine boolean and fancy indexing to answer complex questions quickly.

### 3.4 Broadcasting Masks

```python
positions = rng.integers(0, 5, size=(4, 4))
mask = positions == positions.max(axis=1, keepdims=True)
```

The `mask` highlights the max column per row—a classic interview puzzle.

---

## 4. Universal Functions (ufuncs) & Broadcasting

- **ufuncs** are vectorized element-wise operations (`np.add`, `np.exp`, `np.sqrt`).
- **Broadcasting** aligns arrays of different shapes by adding new axes of length 1 when dimensions match or one equals 1.

```python
x = np.arange(5)
weights = np.array([0.1, 0.2, 0.3, 0.2, 0.2])
weighted = np.multiply(x, weights)
```

### Axis Semantics at a Glance

- `axis=0` means “collapse rows” (operate down columns).
- `axis=1` means “collapse columns” (operate across rows).
- Higher dimensions extend the pattern: axis numbers increase from left to right in `arr.shape`.

```python
grid = rng.integers(0, 9, size=(2, 3, 4))  # shape (batch=2, rows=3, cols=4)
batch_sums = grid.sum(axis=0)   # shape (3, 4): collapses the batch dimension
row_means = grid.mean(axis=2)   # shape (2, 3): collapses columns inside each matrix
keepdims = grid.sum(axis=1, keepdims=True)  # shape (2, 1, 4) useful for broadcasting back
```

### Broadcasting Walkthrough

```python
# Example 1: Feature scaling (shapes: (6, 3) minus (1, 3))
features = rng.normal(size=(6, 3))
col_min = features.min(axis=0, keepdims=True)
col_range = features.ptp(axis=0, keepdims=True)  # max - min per column
scaled = (features - col_min) / col_range

# Example 2: Adding a bias vector to every row (shapes: (4, 3) + (3,))
activations = rng.normal(size=(4, 3))
bias = np.array([0.5, -0.5, 0.25])
shifted = activations + bias

# Example 3: Pairwise Manhattan distance (shapes: (5, 1, 3) & (1, 5, 3))
points = rng.normal(size=(5, 3))
manhattan = np.abs(points[:, None, :] - points[None, :, :]).sum(axis=2)

# Example 4: Broadcasting with newaxis to compute z-scores per sample
logs = rng.normal(size=(3, 4))
sample_mean = logs.mean(axis=1, keepdims=True)  # shape (3, 1)
sample_std = logs.std(axis=1, keepdims=True)
z_scores = (logs - sample_mean) / sample_std
```

### Broadcasting Rules in Action

```python
features = rng.normal(size=(100, 3))
mean = features.mean(axis=0)  # shape (3,)
std = features.std(axis=0)    # shape (3,)
normalized = (features - mean) / std
```

If you forget `keepdims=True`, you can reshape manually:

```python
normalized = (features - mean.reshape(1, -1)) / std.reshape(1, -1)
```

### Custom ufunc Example

```python
@np.vectorize
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

logits = rng.normal(size=5)
probs = sigmoid(logits)
```

Use `np.vectorize` sparingly; writing in pure NumPy is faster when possible.

### Elementwise Arithmetic Patterns

```python
# Example 1: Vector-scalar arithmetic
temperatures_c = np.array([0.0, 12.5, 37.0])
temperatures_f = temperatures_c * 9 / 5 + 32

# Example 2: Matrix addition and Hadamard product
image_a = rng.integers(0, 256, size=(3, 3))
image_b = rng.integers(0, 256, size=(3, 3))
blend = 0.6 * image_a + 0.4 * image_b
elementwise_product = image_a * image_b

# Example 3: Broadcasting scalar to matrix
weights = rng.normal(size=(4, 2))
learning_rate = 0.01
updated = weights - learning_rate * weights

# Example 4: Safe division with `np.divide`
numerator = np.array([3, 4, 5])
denominator = np.array([1, 2, 0])
ratio = np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=float), where=denominator != 0)
```

---

## 5. Aggregations & Reductions

Common reducers: `sum`, `mean`, `std`, `var`, `min`, `max`, `argmin`, `argmax`, `any`, `all`.

```python
matrix = rng.normal(size=(4, 3))
col_means = matrix.mean(axis=0)
row_max_idx = matrix.argmax(axis=1)
```

### Cumulative Operations

```python
values = np.array([1, 3, 2, 4])
cumsum = values.cumsum()  # [1 4 6 10]
cumprod = values.cumprod()  # [ 1  3  6 24]
```

### Reduce vs Aggregate in Interviews

- Show proficiency with `axis` arguments.
- Explain the difference between aggregations returning scalars vs arrays.
- Demonstrate `keepdims=True` when shapes must align for broadcasting.

---

## 6. Reshaping & Structured Transformations

```python
arr = np.arange(12)
reshaped = arr.reshape(3, 4)
flatten = reshaped.ravel()  # view when possible
```

### Transpose & Axis Reordering

```python
# Example 1: Simple transpose of a 2D matrix
matrix = rng.normal(size=(2, 3))
matrix_T = matrix.T  # shape becomes (3, 2)

# Example 2: 3D tensor transpose to switch channel order
images = rng.normal(size=(10, 28, 28))        # (batch, height, width)
chw = np.transpose(images, axes=(0, 2, 1))    # swap height and width

# Example 3: Swap two axes without touching others
volumes = rng.normal(size=(4, 3, 2, 5))       # (batch, channels, depth, width)
swapped = np.swapaxes(volumes, 1, 2)          # (4, 2, 3, 5)

# Example 4: Bring channel axis to the end
time_series = rng.normal(size=(8, 16, 32))    # (batch, channels, time)
reordered = np.moveaxis(time_series, source=1, destination=-1)  # (8, 32, 16)
```

### Concatenation & Stacking

```python
a = np.ones((2, 3))
b = np.zeros((2, 3))
cat = np.concatenate([a, b], axis=0)
stack = np.stack([a, b], axis=1)
```

```python
# Example 1: Vertical stacking with `vstack`
top = np.full((2, 2), fill_value=1)
bottom = np.full((1, 2), fill_value=9)
vertical = np.vstack([top, bottom])

# Example 2: Horizontal stacking with `hstack`
left = np.arange(6).reshape(3, 2)
right = np.arange(100, 106).reshape(3, 2)
horizontal = np.hstack([left, right])

# Example 3: Depth stacking with `dstack`
foreground = np.ones((2, 2))
background = np.zeros((2, 2))
alpha = np.full((2, 2), 0.5)
image_rgba = np.dstack([foreground, background, alpha])

# Example 4: Stack multiple mini-batches along a new axis
batch1 = rng.normal(size=(32, 10))
batch2 = rng.normal(size=(32, 10))
all_batches = np.stack([batch1, batch2], axis=0)  # shape (2, 32, 10)
```

### Splitting Utilities

```python
chunk1, chunk2 = np.split(np.arange(10), [6])
```

### Tiling & Repeating

```python
pattern = np.array([[1, 0], [0, 1]])
tiled = np.tile(pattern, reps=(2, 3))
```

### Sorting & Ordering Patterns

```python
# Example 1: Global sort (flattened view returned)
unsorted = rng.integers(0, 100, size=6)
sorted_values = np.sort(unsorted)

# Example 2: Row-wise sort using axis
matrix = rng.integers(0, 50, size=(3, 4))
row_sorted = np.sort(matrix, axis=1)

# Example 3: Indices of sorted order with `argsort`
scores = rng.normal(size=5)
ranking = np.argsort(scores)[::-1]  # descending order indices

# Example 4: Partial sorting with `partition` (useful for top-k)
large = rng.integers(0, 1_000, size=20)
top3_unsorted = np.partition(large, -3)[-3:]  # fastest way to grab top 3 values
```

---

## 7. Linear Algebra Toolbox

```python
A = rng.normal(size=(3, 3))
b = rng.normal(size=3)
solution = np.linalg.solve(A, b)
det = np.linalg.det(A)
U, S, Vt = np.linalg.svd(A)
```

### Dot vs Matmul vs @

```python
x = rng.normal(size=(3,))
B = rng.normal(size=(3, 4))
print(np.dot(x, B))   # (4,)
print(x @ B)          # equivalent to dot here
```

### Norms & Projections

```python
vec = rng.normal(size=5)
vec_norm = np.linalg.norm(vec)
```

Emphasize `einsum` for advanced interviews (attention mechanisms, tensor contractions).

```python
C = np.einsum('ik,kj->ij', A, B[:3, :])
```

---

## 8. Randomness & Reproducibility

Prefer the `Generator` API over legacy `np.random`.

```python
rng = np.random.default_rng(seed=123)
normal_samples = rng.normal(loc=0.0, scale=1.0, size=(1000,))
permuted = rng.permutation(np.arange(10))
shuffled_matrix = rng.permuted(np.arange(12).reshape(3, 4), axis=1)
```

Key interview fact: `default_rng` decouples global state; reproducibility is per-generator.

---

## 9. Performance & Memory Tuning

1. **Vectorize** loops whenever possible.
2. **Use views** instead of copies by leveraging slicing.
3. **Choose dtypes** carefully (`float32` vs `float64`, `int8` for binary features).
4. **Profile** with `%timeit` or `np.testing.assert_allclose` for correctness/performance.

### Broadcasting vs Loop Performance

```python
vector = rng.normal(size=1_000_000)
scalar = 1.7
result = vector * scalar
```

### Memory Footprint Insight

```python
arr = np.zeros((10_000, 10_000), dtype=np.float32)  # ~381 MB
print(arr.nbytes / 1024**2)
```

Mention out-of-core strategies (memory mapping via `np.memmap`, chunk processing) if asked about large datasets.

---

## 10. Numerical Stability Patterns

- Replace `log(softmax)` with `logsumexp` trick.
- Use `np.exp(arr - arr.max())` to avoid overflow.
- When computing variance, prefer `np.var(..., ddof=1)` for sample variance.

```python
logits = rng.normal(size=5)
exps = np.exp(logits - logits.max())
softmax = exps / exps.sum()
```

---

## 11. Common Interview Exercises

1. **Standardize columns without loops**.
2. **Implement cosine similarity matrix**.
3. **Flatten a batch of images (N, H, W, C) into (N, -1)**.
4. **Compute pairwise Euclidean distances using broadcasting**.
5. **One-hot encode integer labels**.

### Sample Solution: Pairwise Distances

```python
points = rng.normal(size=(5, 3))
# (5, 1, 3) - (1, 5, 3) -> (5, 5, 3)
diffs = points[:, None, :] - points[None, :, :]
dists = np.sqrt((diffs**2).sum(axis=2))
```

Explain the shape transformations aloud—it shows deep understanding.

---

## 12. Practice Prompts

- Rebuild `np.corrcoef` using basic operations.
- Simulate a biased coin and estimate probabilities with bootstrapping.
- Implement softmax regression gradient calculations manually.
- Create a moving window view using `np.lib.stride_tricks.sliding_window_view`.
- Profile two implementations (loop vs vectorized) and document the speedup.

---

## 13. Quick Reference Cheat Sheet

- `arr.reshape(-1, n)` flattens the first dimension automatically.
- `np.newaxis` or `None` inserts axes for broadcasting.
- `np.where(cond, x, y)` selects elements conditionally.
- `np.unique(arr, return_counts=True)` aids frequency analysis.
- `np.stack` adds a new dimension; `np.concatenate` joins along an existing one.
- `np.save`, `np.load`, `np.savez` manage serialized arrays for interviews.

---

## 14. Additional Resources

- [NumPy User Guide](https://numpy.org/doc/stable/user/index.html)
- [Array Broadcasting Explained](https://numpy.org/doc/stable/user/basics.broadcasting.html)
- [NumPy Best Practices](https://numpy.org/doc/stable/user/basics.python.html)
- [Scientific Python Lectures](https://scipy-lectures.org/intro/numpy/index.html)

Revisit this guide periodically, augment it with your own notes and code snippets, and drill the practice prompts to cement mastery.
