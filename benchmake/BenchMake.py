#BenchMake
#(c) 2025, Prof Dr Amanda S Barnard PhD DSc
# Version 1.0, 06/02/2025

import os
import math
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import CountVectorizer
import hashlib

# Attempt to import CuPy for GPU computations and check GPU compatibility
try:
    import cupy as cp
    # Check if at least one CUDA-capable device is available
    if cp.cuda.runtime.getDeviceCount() > 0:
        try:
            from cupy.cuda import Device
            device = Device(0)  # Use the first device
            _ = device.compute_capability  # Verify compatibility
            gpu_available = True
            cp.random.seed(42)
            print("GPU is available. Using CuPy for GPU computations.")
        except cp.cuda.runtime.CUDARuntimeError as e:
            cp = None
            gpu_available = False
            print(f"Warning: CuPy is installed but the GPU is not compatible: {e}. Reverting to CPU.")
    else:
        cp = None
        gpu_available = False
        print("Warning: No CUDA-capable devices found. Reverting to CPU.")
except ImportError:
    cp = None
    gpu_available = False
    print("CuPy not installed. Using NumPy for CPU computations.")
except Exception as e:
    cp = None
    gpu_available = False
    print(f"Warning: An unexpected error occurred while initializing CuPy: {e}\nReverting to CPU computations.")

# Fixed random seed for NumPy
np.random.seed(42)

class NMF:
    FIXED_RANDOM_STATE = 42  # Fixed seed for determinism

    def __init__(self, n_components, max_iter=1000, tol=1e-4, random_state=FIXED_RANDOM_STATE):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit_transform(self, X):
        """
        Fit the NMF model to X (in FP32) and return the factorized matrices (W, H).

        If GPU is available, attempts GPU-based factorization with CuPy.
        Otherwise, falls back to CPU-based factorization with NumPy.
        """
        if gpu_available:
            try:
                return self._fit_transform_gpu(X)
            except cp.cuda.memory.OutOfMemoryError:
                print("GPU out of memory during NMF computation. Switching to CPU.")
                return self._fit_transform_cpu(cp.asnumpy(X))
        else:
            return self._fit_transform_cpu(X)

    def _fit_transform_gpu(self, X):
        cp.random.seed(self.random_state)
        X = cp.asarray(X, dtype=cp.float32)
        n_samples, n_features = X.shape

        W = cp.maximum(cp.random.rand(n_samples, self.n_components, dtype=cp.float32), 1e-4)
        H = cp.maximum(cp.random.rand(self.n_components, n_features, dtype=cp.float32), 1e-4)

        for _ in range(self.max_iter):
            try:
                numerator_H = W.T @ X
                denominator_H = W.T @ W @ H + 1e-9
                H *= numerator_H / denominator_H
                H = cp.maximum(H, 1e-4)

                numerator_W = X @ H.T
                denominator_W = W @ H @ H.T + 1e-9
                W *= numerator_W / denominator_W
                W = cp.maximum(W, 1e-4)
            except cp.cuda.memory.OutOfMemoryError:
                print("GPU out of memory during NMF iteration. Switching to CPU.")
                del W, H, X
                cp._default_memory_pool.free_all_blocks()
                return self._fit_transform_cpu(cp.asnumpy(X))
        return W.get(), H.get()

    def _fit_transform_cpu(self, X):
        np.random.seed(self.random_state)
        X = X.astype(np.float32)
        n_samples, n_features = X.shape

        W = np.maximum(np.random.rand(n_samples, self.n_components).astype(np.float32), 1e-4)
        H = np.maximum(np.random.rand(self.n_components, n_features).astype(np.float32), 1e-4)

        for _ in range(self.max_iter):
            numerator_H = W.T @ X
            denominator_H = W.T @ W @ H + 1e-9
            H *= numerator_H / denominator_H
            H = np.maximum(H, 1e-4)

            numerator_W = X @ H.T
            denominator_W = W @ H @ H.T + 1e-9
            W *= numerator_W / denominator_W
            W = np.maximum(W, 1e-4)
        return W, H

class BenchMake:
    FIXED_RANDOM_STATE = 42  # For any random operations in NMF, etc.

    def __init__(self, n_jobs=-1):
        """
        Initialize BenchMake with the specified number of parallel jobs.
        """
        self.n_jobs = n_jobs

    def _determine_batch_size(self, n_samples):
        """
        Automatically determine a batch size based on the number of samples and the number of jobs.
        Aim for roughly 10 batches per worker; clamp the result between 100 and 1000.
        """
        p = self.n_jobs if self.n_jobs > 0 else (os.cpu_count() or 1)
        batch_size = math.ceil(n_samples / (10 * p))
        batch_size = max(100, min(batch_size, 1000))
        return batch_size

    def partition(self, X, y, test_size=0.2, data_type='tabular', return_indices=False):
        """
        Partition the dataset into training and testing sets via archetypal analysis
        (with stable hashing) using FP32 throughout.

        The only inputs required from the user are:
            X, y, test_size, data_type, and return_indices.
        Batch size is automatically determined.

        Before GPU computations, the code estimates the required memory and checks
        available GPU memory. If insufficient, it prints a warning and reverts to CPU.

        If return_indices is False (default):
            returns (X_train, X_test, y_train, y_test) in the same type as the inputs.
        If return_indices is True:
            returns (train_indices, test_indices) as lists of integers.
        """
        if data_type == 'tabular':
            return self._partition_tabular(X, y, test_size, return_indices)
        elif data_type == 'image':
            return self._partition_image(X, y, test_size, return_indices)
        elif data_type == 'sequential':
            return self._partition_sequential(X, y, test_size, return_indices)
        elif data_type == 'signal':
            return self._partition_signal(X, y, test_size, return_indices)
        elif data_type == 'graph':
            return self._partition_graph(X, y, test_size, return_indices)
        else:
            raise ValueError(f"Unsupported data type: {data_type}")

    # --- Partitioning methods for each data type ---

    def _partition_tabular(self, X, y, test_size, return_indices):
        if isinstance(X, pd.DataFrame):
            X = X.values.astype(np.float32)
        X_ordered, y_ordered = self._order_data_by_hash(X, y)
        if gpu_available:
            self._check_gpu_memory(X_ordered, test_size)
        batch_size = self._determine_batch_size(len(X_ordered))
        num_samples = len(X_ordered)
        num_archetypes = max(1, int(np.ceil(num_samples * test_size)))
        _, selected_indices = self._perform_archetypal_analysis(X_ordered, test_size, num_archetypes, batch_size)
        return self._split_data(X_ordered, y_ordered, selected_indices, return_indices)

    def _partition_image(self, X, y, test_size, return_indices):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if X.ndim >= 3:
            n_samples = X.shape[0]
            X = X.reshape(n_samples, -1).astype(np.float32)
        elif X.ndim == 2:
            X = X.astype(np.float32)
        else:
            raise ValueError("Unsupported image data shape.")
        X_ordered, y_ordered = self._order_data_by_hash(X, y)
        if gpu_available:
            self._check_gpu_memory(X_ordered, test_size)
        batch_size = self._determine_batch_size(len(X_ordered))
        num_samples = len(X_ordered)
        num_archetypes = max(1, int(np.ceil(num_samples * test_size)))
        _, selected_indices = self._perform_archetypal_analysis(X_ordered, test_size, num_archetypes, batch_size)
        return self._split_data(X_ordered, y_ordered, selected_indices, return_indices)

    def _partition_signal(self, X, y, test_size, return_indices):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(X, np.ndarray):
            if X.ndim == 1:
                X = X.reshape(-1, 1).astype(np.float32)
            elif X.ndim > 2:
                n_samples = X.shape[0]
                X = X.reshape(n_samples, -1).astype(np.float32)
            else:
                X = X.astype(np.float32)
        X_ordered, y_ordered = self._order_data_by_hash(X, y)
        if gpu_available:
            self._check_gpu_memory(X_ordered, test_size)
        batch_size = self._determine_batch_size(len(X_ordered))
        num_samples = len(X_ordered)
        num_archetypes = max(1, int(np.ceil(num_samples * test_size)))
        _, selected_indices = self._perform_archetypal_analysis(X_ordered, test_size, num_archetypes, batch_size)
        return self._split_data(X_ordered, y_ordered, selected_indices, return_indices)

    def _partition_sequential(self, X, y, test_size, return_indices):
        if isinstance(X, pd.Series):
            X_seq = X.tolist()
        elif isinstance(X, list):
            X_seq = X
        else:
            raise ValueError("Sequential data must be a list or Series.")
        vectorizer = CountVectorizer(analyzer='char', lowercase=False)
        X_numerical = vectorizer.fit_transform(X_seq).toarray().astype(np.float32)
        X_ordered_num, y_ordered = self._order_data_by_hash(X_numerical, y)
        if gpu_available:
            self._check_gpu_memory(X_ordered_num, test_size)
        batch_size = self._determine_batch_size(len(X_ordered_num))
        num_samples = len(X_ordered_num)
        num_archetypes = max(1, int(np.ceil(num_samples * test_size)))
        _, selected_indices = self._perform_archetypal_analysis(X_ordered_num, test_size, num_archetypes, batch_size)
        X_seq_ordered, _ = self._order_data_by_hash(X_seq, y)
        return self._split_data(X_seq_ordered, y_ordered, selected_indices, return_indices)

    def _partition_graph(self, X, y, test_size, return_indices):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if X.ndim > 2:
            n_samples = X.shape[0]
            X = X.reshape(n_samples, -1)
        X = X.astype(np.float32)
        X_ordered, y_ordered = self._order_data_by_hash(X, y)
        if gpu_available:
            self._check_gpu_memory(X_ordered, test_size)
        batch_size = self._determine_batch_size(len(X_ordered))
        num_samples = len(X_ordered)
        num_archetypes = max(1, int(np.ceil(num_samples * test_size)))
        _, selected_indices = self._perform_archetypal_analysis(X_ordered, test_size, num_archetypes, batch_size)
        return self._split_data(X_ordered, y_ordered, selected_indices, return_indices)

    # --- Archetypal Analysis and Distance Computations (FP32) ---

    def _perform_archetypal_analysis(self, X, test_size, num_archetypes, batch_size=1000):
        if isinstance(X, pd.DataFrame):
            X = X.values
        elif isinstance(X, list):
            X = np.array(X, dtype=np.float32)
        X = X.astype(np.float32)
        X_min = np.min(X, axis=0).astype(np.float32)
        X_max = np.max(X, axis=0).astype(np.float32)
        X_scaled = (X - X_min) / (X_max - X_min + 1e-9)
        X_scaled = X_scaled.astype(np.float32)
        n_samples, n_features = X_scaled.shape
        test_size_count = max(1, int(np.ceil(n_samples * test_size)))
        num_archetypes = test_size_count
        nmf = NMF(n_components=num_archetypes, max_iter=200, tol=1e-4, random_state=42)
        W, H = nmf.fit_transform(X_scaled)
        explained_variances = np.var(W, axis=0)
        sorted_indices = np.argsort(explained_variances)[::-1]
        H_sorted = H[sorted_indices]
        distances = self._compute_distances(X_scaled, H_sorted, batch_size)
        selected_indices = set()
        for i in range(num_archetypes):
            archetype_distances = distances[:, i].copy()
            if selected_indices:
                archetype_distances[list(selected_indices)] = np.inf
            idx = np.argmin(archetype_distances)
            selected_indices.add(idx)
        selected_indices = sorted(selected_indices)
        return X_scaled, selected_indices

    def _compute_distances(self, X_scaled, H_sorted, batch_size=1000):
        if gpu_available and cp is not None:
            return self._compute_distances_gpu_batched(X_scaled, H_sorted, batch_size)
        else:
            return self._compute_distances_cpu_batched(X_scaled, H_sorted, batch_size)

    def _compute_distances_gpu_batched(self, X_scaled, H_sorted, batch_size=1000):
        n_samples = X_scaled.shape[0]
        n_archetypes = H_sorted.shape[0]
        distances = np.empty((n_samples, n_archetypes), dtype=np.float32)
        try:
            H_sorted_cp = cp.asarray(H_sorted, dtype=cp.float32)
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                X_batch = cp.asarray(X_scaled[start_idx:end_idx], dtype=cp.float32)
                batch_distances = cp.linalg.norm(
                    X_batch[:, cp.newaxis, :] - H_sorted_cp[cp.newaxis, :, :],
                    axis=2
                ).get()
                distances[start_idx:end_idx] = batch_distances
                del X_batch, batch_distances
                cp._default_memory_pool.free_all_blocks()
            del H_sorted_cp
            cp._default_memory_pool.free_all_blocks()
        except cp.cuda.memory.OutOfMemoryError:
            print("GPU out of memory. Switching to CPU distance computation.")
            cp._default_memory_pool.free_all_blocks()
            return self._compute_distances_cpu_batched(X_scaled, H_sorted, batch_size)
        return distances

    def _compute_distances_cpu_batched(self, X_scaled, H_sorted, batch_size=500):
        n_samples = X_scaled.shape[0]
        n_archetypes = H_sorted.shape[0]
        distances = np.empty((n_samples, n_archetypes), dtype=np.float32)
        def compute_batch(start_idx, end_idx):
            X_batch = X_scaled[start_idx:end_idx]
            batch_distances = np.linalg.norm(
                X_batch[:, np.newaxis, :] - H_sorted[np.newaxis, :, :],
                axis=2
            ).astype(np.float32)
            return (start_idx, end_idx, batch_distances)
        batches = [(i, min(i + batch_size, n_samples)) for i in range(0, n_samples, batch_size)]
        results = Parallel(n_jobs=self.n_jobs)(delayed(compute_batch)(start, end) for start, end in batches)
        for start, end, batch_distances in results:
            distances[start:end] = batch_distances
        return distances

    # --- GPU Memory Estimation and Check ---

    def _estimate_memory(self, X, test_size):
        n_samples, n_features = X.shape
        n_components = max(1, int(np.ceil(n_samples * test_size)))
        size_X = n_samples * n_features * 4
        size_W = n_samples * n_components * 4
        size_H = n_components * n_features * 4
        size_X_scaled = size_X
        size_distance = n_samples * n_components * 4
        total = size_X + size_W + size_H + size_X_scaled + size_distance
        safety_factor = 1.5
        return total * safety_factor

    def _check_gpu_memory(self, X, test_size):
        global gpu_available
        estimated = self._estimate_memory(X, test_size)
        free_mem = cp.cuda.runtime.memGetInfo()[0]
        if free_mem < estimated:
            print(f"Warning: Estimated GPU memory required is {estimated} bytes, "
                  f"but only {free_mem} bytes are free. Switching to CPU.")
            gpu_available = False

    # --- Hash-based ordering ---

    def _order_data_by_hash(self, X, y):
        row_hashes = []
        num_rows = len(X)
        for i in range(num_rows):
            row_i = self._get_row(X, i)
            row_bytes = self._row_to_bytes(row_i)
            h = hashlib.md5(row_bytes).hexdigest()
            row_hashes.append(h)
        sorted_indices = np.argsort(row_hashes)
        X_ordered = self._reindex_data(X, sorted_indices)
        y_ordered = self._reindex_data(y, sorted_indices)
        return X_ordered, y_ordered

    def _get_row(self, X, i):
        if isinstance(X, pd.DataFrame):
            return X.iloc[i].values
        elif isinstance(X, pd.Series):
            return X.iloc[i]
        elif isinstance(X, np.ndarray):
            return X[i]
        elif isinstance(X, list):
            return X[i]
        else:
            raise TypeError("Unsupported type for X in _get_row().")

    def _row_to_bytes(self, row_i):
        if isinstance(row_i, str):
            return row_i.encode('utf-8')
        if isinstance(row_i, (np.ndarray, list, tuple)):
            try:
                arr = np.array(row_i, dtype=np.float32).reshape(-1)
                return arr.tobytes()
            except ValueError:
                return str(row_i).encode('utf-8')
        else:
            return str(row_i).encode('utf-8')

    def _reindex_data(self, data, indices):
        if isinstance(data, pd.DataFrame):
            return data.iloc[indices].reset_index(drop=True)
        elif isinstance(data, pd.Series):
            return data.iloc[indices].reset_index(drop=True)
        elif isinstance(data, np.ndarray):
            return data[indices]
        elif isinstance(data, list):
            return [data[i] for i in indices]
        else:
            raise TypeError("Unsupported data type in _reindex_data.")

    # --- Splitting train/test with optional return of indices ---

    def _split_data(self, X, y, selected_indices, return_indices):
        test_set = set(selected_indices)
        all_set = set(range(len(X)))
        train_set = sorted(all_set - test_set)
        test_set = sorted(test_set)
        if len(test_set) == 0:
            raise ValueError("No instances selected for the test set.")
        if len(train_set) == 0:
            raise ValueError("No instances left for the training set.")
        if return_indices:
            return train_set, test_set
        else:
            X_train = self._reindex_data(X, train_set)
            X_test = self._reindex_data(X, test_set)
            y_train = self._reindex_data(y, train_set)
            y_test = self._reindex_data(y, test_set)
            return X_train, X_test, y_train, y_test
