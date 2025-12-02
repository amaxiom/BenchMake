import pytest
import pandas as pd

openml = pytest.importorskip("openml")

from benchmake import BenchMake


@pytest.mark.integration
def test_benchmake_german_credit_tabular_reproducible():
    """
    Integration test:
    - Load German Credit (credit-g, OpenML id=31)
    - One-hot encode categoricals
    - Run BenchMake tabular partition
    - Check determinism and basic invariants
    """

    # Load dataset from OpenML as pandas DataFrame
    dataset = openml.datasets.get_dataset("credit-g")  # id=31
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        target="class",
        dataset_format="dataframe",
    )

    # Ensure clean integer index
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    # One-hot encode categorical variables to obtain a numeric matrix
    X_encoded = pd.get_dummies(X).astype("float32")

    n_samples = len(X_encoded)
    test_size = 0.2

    bm = BenchMake(n_jobs=2)

    # First run: get train / test indices
    train_idx_1, test_idx_1 = bm.partition(
        X_encoded,
        y,
        test_size=test_size,
        data_type="tabular",
        return_indices=True,
    )

    # Second run: should produce exactly the same indices
    train_idx_2, test_idx_2 = bm.partition(
        X_encoded,
        y,
        test_size=test_size,
        data_type="tabular",
        return_indices=True,
    )

    # Deterministic behaviour
    assert set(train_idx_1) == set(train_idx_2)
    assert set(test_idx_1) == set(test_idx_2)

    # Train and test are disjoint
    assert set(train_idx_1).isdisjoint(test_idx_1)

    # Coverage: every sample is in exactly one of train or test
    assert len(train_idx_1) + len(test_idx_1) == n_samples

    # Test size matches requested fraction (with standard rounding)
    expected_test_size = int(round(test_size * n_samples))
    assert len(test_idx_1) == expected_test_size
