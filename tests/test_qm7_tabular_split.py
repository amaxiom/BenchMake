import pytest
import numpy as np
import pandas as pd
from pathlib import Path

scipy = pytest.importorskip("scipy")
from scipy.io import loadmat  # noqa: E402

from benchmake import BenchMake


DATA_PATH = Path(__file__).parent / "data" / "qm7.mat"

if not DATA_PATH.exists():
    pytest.skip(
        f"QM7 data file not found at {DATA_PATH}. "
        "Download qm7.mat from quantum-machine.org and place it there.",
        allow_module_level=True,
    )


@pytest.mark.integration
def test_benchmake_qm7_tabular_25pct_reproducible():
    """
    Integration test using the QM7 dataset (Coulomb matrices + atomization energies).

    Dataset:
        - QM7: 7165 small organic molecules
        - Inputs: 23x23 Coulomb matrices
        - Target: atomization energies

    Procedure:
        - Load qm7.mat
        - Flatten Coulomb matrices to tabular features
        - Run BenchMake with 25% test split
        - Check determinism and basic invariants
    """

    mat = loadmat(DATA_PATH)

    # According to the QM7 documentation, X is (7165, 23, 23) Coulomb matrices
    # and T is the corresponding atomization energies.  :contentReference[oaicite:3]{index=3}
    X = mat["X"]  # shape (n_samples, 23, 23)
    T = mat["T"]  # shape (n_samples, 1) or (1, n_samples) depending on version

    # Normalise shapes
    X = np.asarray(X)
    T = np.asarray(T).reshape(-1)

    n_samples = X.shape[0]
    assert X.shape[1] == 23 and X.shape[2] == 23

    # Flatten each Coulomb matrix into a feature vector of length 23*23
    X_flat = X.reshape(n_samples, -1)
    X_df = pd.DataFrame(X_flat)
    y = pd.Series(T)

    test_size = 0.25

    bm = BenchMake(n_jobs=2)

    # First run: return indices
    train_idx_1, test_idx_1 = bm.partition(
        X_df,
        y,
        test_size=test_size,
        data_type="tabular",
        return_indices=True,
    )

    # Second run: same call should be deterministic
    train_idx_2, test_idx_2 = bm.partition(
        X_df,
        y,
        test_size=test_size,
        data_type="tabular",
        return_indices=True,
    )

    # Deterministic indices
    assert set(train_idx_1) == set(train_idx_2)
    assert set(test_idx_1) == set(test_idx_2)

    # Train and test sets are disjoint
    assert set(train_idx_1).isdisjoint(test_idx_1)

    # Every sample appears in exactly one of train or test
    assert len(train_idx_1) + len(test_idx_1) == n_samples

    # Test size is exactly 25% (with standard rounding)
    expected_test_size = int(round(test_size * n_samples))
    assert len(test_idx_1) == expected_test_size
