from __future__ import annotations

import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal

from vicinity.utils import normalize, normalize_or_copy


def test_normalize(vectors: np.ndarray) -> None:
    """Test the normalize function."""
    vector = np.array([3.0, 4.0])
    expected = np.array([0.6, 0.8])
    result = normalize(vector)
    assert_almost_equal(result, expected)

    # Test normalizing a zero vector
    zero_vector = np.array([0.0, 0.0])
    expected_zero = np.array([0.0, 0.0])
    result_zero = normalize(zero_vector)
    assert_array_equal(result_zero, expected_zero)

    # Test normalizing a matrix of vectors
    vectors = np.array([[3.0, 4.0], [1.0, 0.0], [0.0, 0.0]])
    expected_vectors = np.array([[0.6, 0.8], [1.0, 0.0], [0.0, 0.0]])
    result_vectors = normalize(vectors)
    assert_almost_equal(result_vectors, expected_vectors)


def test_normalize_or_copy() -> None:
    """Test the normalize_or_copy function."""
    # Test when vectors are already normalized
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]])
    result = normalize_or_copy(vectors)
    assert result is vectors, "Should return the original array"

    # Test when vectors are not normalized
    vectors = np.array([[3.0, 4.0], [5.0, 12.0]])
    result = normalize_or_copy(vectors)
    assert result is not vectors, "Should return a new array"

    # Test with zero vectors
    zero_vectors = np.array([[0.0, 0.0], [0.0, 0.0]])
    result_zero = normalize_or_copy(zero_vectors)
    assert_array_equal(result_zero, zero_vectors)
    assert result_zero is zero_vectors, "Should return the original array"


def test_normalize_or_copy_dtypes() -> None:
    """Test that normalized vectors are recognized for every float dtype."""
    rng = np.random.default_rng(42)
    for dtype in (np.float16, np.float32, np.float64):
        vectors = normalize(rng.standard_normal((1000, 384)).astype(dtype))
        assert normalize_or_copy(vectors) is vectors, f"Should return the original array for {dtype}"

        scaled = vectors * dtype(1.01)
        assert normalize_or_copy(scaled) is not scaled, f"Should return a new array for {dtype}"
