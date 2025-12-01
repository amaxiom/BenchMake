---
# BenchMake
## Turn any Scientific Data Set into a Reproducible Benchmark

Version: **1.1.2**  
Date: **01/07/2025**  
Author: **Prof. Amanda S. Barnard, PhD DSc**  

BenchMake is a Python package that partitions a data set into train/test splits using **archetypal analysis**. It relies on an **NMF-based** approach, performing a multiplicative-update factorization and then computing distances to the discovered “archetypes.” The nearest unique data points become the test set (or optionally just the test **indices**). BenchMake supports **GPU acceleration** via **CuPy** (if available), or automatically falls back to CPU-based NumPy. 

## Table of Contents

1. [Features](#features)  
2. [Installation](#installation)  
3. [Quick Start](#quick-start)  
4. [Usage](#usage)  
   - [Partitioning Tabular Data](#partitioning-tabular-data)  
   - [Partitioning Images](#partitioning-images)  
   - [Partitioning Sequential Data](#partitioning-sequential-data)  
   - [Partitioning Signal Data](#partitioning-signal-data)  
   - [Partitioning Graph Data](#partitioning-graph-data)  
5. [Implementation](#implementation)
   - [Parallelism & GPU Acceleration](#parallelism--gpu-acceleration)  
   - [Algorithmic Details](#algorithmic-details)  
   - [Known Limitations](#known-limitations)  
6. [Acknowledgements](#acknowledgments)
   - [License](#license)  
   - [Citation](#citation) 

---

## Features

- **Archetypal Analysis Partitioning**  
  Automatically finds “extreme” points (archetypes) that best approximate the entire data set in a low-dimensional sense, and uses them to form a test set.

- **Multi-Domain Support**  
  BenchMake handles:
  - **Tabular** structured data (NumPy arrays, Pandas DataFrames)  
  - **Image** data (multi-dimensional arrays)  
  - **Sequential** data (strings, text)  
  - **Signal** data (time-series, audio, sensor arrays)  
  - **Graph** data (node-feature matrices)

- **Deterministic**  
  Fixed random seeds and consistent initialization ensure you get the same splits every time for the same data, whatever split size you select, regardless of data order.

- **Automatic Batch Size**  
  Dynamically chooses a batch size for distance computations based on the data size and number of CPU jobs available.

- **Optional Return of Indices**  
  Users can obtain either `(X_train, X_test, y_train, y_test)` **or** `(Train_indices, Test_indices)` for maximum flexibility.

---

## Installation

BenchMake requires Python 3.7 or higher. To install via pip:

```bash
pip install benchmake
```

**Optional**: For GPU support, install [CuPy](https://docs.cupy.dev/en/stable/install.html) appropriate to your CUDA version. Install CuPy separately if you require this capability, making sure it is compatible with your CUDA installation too. If CuPy is not found or no compatible GPU exists, BenchMake reverts to CPU silently (with a warning).

---

## Quick Start

```python
from benchmake import BenchMake
import numpy as np

# Sample data: 1000 samples, 20 features
X = np.random.rand(1000, 20)
y = np.random.randint(0, 5, size=1000)

# Instantiate BenchMake with 4 parallel CPU jobs
bm = BenchMake(n_jobs=4)

# Partition the data into train/test using 20% test split
X_train, X_test, y_train, y_test = bm.partition(
    X, 
    y, 
    test_size=0.2, 
    data_type='tabular', 
    return_indices=False
)

print("Train size:", len(X_train), "Test size:", len(X_test))
```

---

## Usage

### Partitioning Tabular Data

When tabular data is provided (as a NumPy array, Pandas DataFrame, or list), BenchMake first converts it to a consistent NumPy array (if it isn’t already) so that all numerical operations are performed in float32. Next, it reorders the data rows deterministically by computing a stable hash (using the MD5 algorithm) for each row. This guarantees that the same data, regardless of the original row order, produces the same sorted order. BenchMake then applies a min–max scaling to the data before partitioning. BenchMake returns either four splits (`X_train, X_test, y_train, y_test`) in the same data type as the user provided (for example, if you used Pandas DataFrames, you get DataFrames back) or, if requested, just the lists of indices for the training and testing sets.

Assume you have `(X, y)` in either a NumPy array or a Pandas DataFrame/Series. Just specify `data_type='tabular'`:

```python
X_train, X_test, y_train, y_test = bm.partition(
    X, 
    y, 
    test_size=0.2, 
    data_type='tabular',
    return_indices=False
)
 
Train_indices, Test_indices = bm.partition(
    X, 
    y, 
    test_size=0.2, 
    data_type='tabular',
    return_indices=True
)
```

### Partitioning Images

For image data, BenchMake expects input in the form of a multi-dimensional array or DataFrame, where each image is typically structured as (`n_samples`, `height`, `width`, `channels`). It first converts the data to a float32 NumPy array (if it isn’t already) and then flattens each image (`n_samples, height*width*channels`) into a one-dimensional vector so that every image is represented as a row vector. The rows are then reordered deterministically using the stable hashing strategy. The images (now as 1D vectors) are min–max scaled, and the data is partitioned. BenchMake returns either the training and testing subsets in the same format as the original input (e.g., as NumPy arrays or DataFrames) or the corresponding indices if the user has requested that mode.

```python
# Suppose X is shape (n_samples, height, width, channels)
X_train, X_test, y_train, y_test = bm.partition(
    X, 
    y, 
    test_size=0.2, 
    data_type='image',
    return_indices=False
)
 
Train_indices, Test_indices = bm.partition(
    X, 
    y, 
    test_size=0.2, 
    data_type='image',
    return_indices=True
)
```

### Partitioning Sequential Data

BenchMake handles sequential data such as text strings, SMILES strings, or DNA sequences by first taking the provided list or Pandas Series and converting each sequence into a numerical (vector) representation using a character-level `CountVectorizer`. This transformation results in a two-dimensional NumPy array (float32) where each row corresponds to the numeric representation of a sequence. The rows of this numeric representation are then deterministically reordered via the stable hash. BenchMake then applies min–max scaling and partitions the data. Finally, the original sequences are re-ordered using the same hash order, and BenchMake returns either the full training and testing splits (in the same type as the original input, e.g., list or Series) or the indices of the splits if that is requested.

For sequences or text:

```python
sequences = ["ACGTG", "GGTTA", "TTACG", ...]  # e.g., list of strings
# sequences can also be SMILEs, e.g., ["CCO", "c1ccccc1", "CC(=O)O",  ...]
y = [label1, label2, ...]  # labels

X_train, X_test, y_train, y_test = bm.partition(
    sequences, 
    y, 
    test_size=0.2,
    data_type='sequential',
    return_indices=False
)
 
Train_indices, Test_indices = bm.partition(
    sequences, 
    y, 
    test_size=0.2, 
    data_type='sequential',
    return_indices=True
)
```

### Partitioning Signal Data

For signal data such as time series, audio signals, or sensor outputs BenchMake first ensures that the data is represented as a consistent float32 NumPy array. If the signals are provided in a multi-dimensional format (for example, if each signal has multiple channels or timepoints arranged in a 3D array), they are flattened so that each signal becomes a single row vector. Once in this unified 2D format, the rows are deterministically sorted using a stable hashing method. After min–max scaling BenchMake partitions the data.  BenchMake returns either the resulting training and testing data in the same structure as the input (e.g., NumPy arrays or DataFrames) or simply the lists of indices for each split.

Signal data (time-series, audio, sensors) can be 2D `(n_signals, n_features)` or 3D `(n_signals, length, channels)`:

```python
X_train, X_test, y_train, y_test = bm.partition(
    X, 
    y, 
    test_size=0.2,
    data_type='signal',
    return_indices=False
)
 
Train_indices, Test_indices = bm.partition(
    X, 
    y, 
    test_size=0.2, 
    data_type='signal',
    return_indices=True
)
```
### Partitioning Graph Data

When dealing with graph data, BenchMake assumes that the user provides a node-feature matrix where each row represents a node and each column represents a feature (this can be in a Pandas DataFrame, NumPy array, or list format). If necessary, the multi-dimensional input is first converted into a two-dimensional float32 array (by flattening any extra dimensions). Stable hashing is applied to the rows to reorder the data, and following min–max scaling, BenchMake partitions the data based on the nodes. The final output will be either the training and testing splits in the same format as the input data or, if specified by the user, the lists of indices corresponding to these splits.

If you have a node-feature matrix `(n_nodes, n_features)`, treat it as `data_type='graph'`:

```python
X_train, X_test, y_train, y_test = bm.partition(
    X_node_features, 
    node_labels, 
    test_size=0.2,
    data_type='graph',
    return_indices=False
)
 
Train_indices, Test_indices = bm.partition(
    X_node_features, 
    node_labels, 
    test_size=0.2, 
    data_type='graph',
    return_indices=True
)
```

## Implementation 

### Parallelism & GPU Acceleration

CPU Parallelism:  
BenchMake uses Python’s [joblib](https://joblib.readthedocs.io/) for parallelizing the **distance** computations only.  
The main NMF loop is effectively single-threaded from Python’s perspective, though an optimized BLAS library (MKL/OpenBLAS) can provide multi-threaded matrix multiplication.

GPU Acceleration:  
If [CuPy](https://docs.cupy.dev/) is installed and you have a CUDA-capable GPU, BenchMake calls GPU code for the NMF factorization and distance calculations.  
If insufficient GPU memory is detected, or if any GPU error occurs, BenchMake warns and reverts to the CPU.

Batch Size:  
Automatically chosen to balance memory usage and overhead. You can control the number of CPU jobs via `n_jobs` when creating `BenchMake(n_jobs=4)`. Use `BenchMake(n_jobs=-1)` to access all available processors.

Important: Because most of the work is in the NMF loop, you may not see dramatic multi-CPU speedups unless you rely on a multi-threaded NumPy/BLAS or CuPy on GPU.



### Algorithmic Details

1. NMF (Multiplicative Update):  
BenchMake performs a basic multiplicative‐update NMF with a fixed random seed for determinism. The number of components is equal to the desired **test set size** (i.e., `ceil(n_samples * test_size)`).

2. Archetype Selection:  
After NMF, the code computes distances from each sample to each of the `k` archetypes, picks the closest sample to each archetype, and forms the test set from those selected indices.

3. Stable Hash Sorting:  
BenchMake reorders all data by a hash of each row’s bytes to ensure that identical data yields identical partitions no matter the input order. This ensures strict determinism.



### Known Limitations

Scaling:  
Because `k ~ O(n)` (for a constant fraction test size) and factorization and distances (d) each scale approximately as `O(n² d)`, BenchMake can become slow for very large data sets.  BenchMake is not a fast alternative to random splits, it is a better alternative; delivering reproducible, and more challenging testing sets.

Limited Parallelism:  
The NMF step is effectively single-threaded except for what is inherent in BLAS. Only the distance computations are joblib-parallelized. GPU usage (if available) provides a bigger speedup for NMF and distance steps.

Memory Consumption:  
For large `n`, or if `test_size` is large, memory usage can be significant. BenchMake attempts to estimate GPU memory usage and revert to CPU if insufficient.

Simplicity Over Customization:  
BenchMake does not expose advanced NMF algorithms (such as HALS or block-coordinate). The code may be extended to accommodate more sophisticated or distributed approaches in the future.


## Acknowledgments

### License

The project is distributed under an MIT License.

This software is provided 'as-is', without any express or implied warranty. Use at your own risk.

### Citation

Amanda S. Barnard, "BenchMake: Turn any Scientific Data Set into a Reproducible Benchmark" arXiv preprint arXiv:2506.23419, 2025.

```python
@misc{barnard2025benchmake,
      title={BenchMake: Turn any scientific data set into a reproducible benchmark}, 
      author={Amanda S Barnard},
      year={2025},
      eprint={2506.23419},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2506.23419}, 
}
```


Happy BenchMaking! 