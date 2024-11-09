from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from nearest import Nearest
from nearest.datatypes import Backend


def test_nearest_init(backend_type: Backend, items: list[str], vectors: np.ndarray) -> None:
    """
    Test Nearest.init.

    :param backend_type: The backend type to use (BASIC or HNSW).
    :param items: A list of item names.
    :param vectors: An array of vectors.
    """
    nearest = Nearest.from_vectors_and_items(vectors, items, backend_type=backend_type)
    assert len(nearest) == len(items)
    assert nearest.items == items
    assert nearest.dim == vectors.shape[1]

    vectors = np.random.default_rng(42).random((len(items) - 1, 5))

    with pytest.raises(ValueError):
        nearest = Nearest.from_vectors_and_items(vectors, items, backend_type=backend_type)


def test_nearest_from_vectors_and_items(backend_type: Backend, items: list[str], vectors: np.ndarray) -> None:
    """
    Test Nearest.from_vectors_and_items.

    :param backend_type: The backend type to use (BASIC or HNSW).
    :param items: A list of item names.
    :param vectors: An array of vectors.
    """
    nearest = Nearest.from_vectors_and_items(vectors, items, backend_type=backend_type)

    assert len(nearest) == len(items)
    assert nearest.items == items
    assert nearest.dim == vectors.shape[1]


def test_nearest_query(nearest_instance: Nearest, query_vector: np.ndarray) -> None:
    """
    Test Nearest.query.

    :param nearest_instance: A Nearest instance.
    :param query_vector: A query vector.
    """
    results = nearest_instance.query(query_vector, k=2)

    assert len(results) == 1


def test_nearest_query_threshold(nearest_instance: Nearest, query_vector: np.ndarray) -> None:
    """
    Test Nearest.query_threshold method.

    :param nearest_instance: A Nearest instance.
    :param query_vector: A query vector.
    """
    results = nearest_instance.query_threshold(query_vector, threshold=0.7)

    assert len(results) >= 1


def test_nearest_insert(backend_type: Backend, nearest_instance: Nearest, query_vector: np.ndarray) -> None:
    """
    Test Nearest.insert method.

    :param backend_type: The backend type to use.
    :param nearest_instance: A Nearest instance.
    :param query_vector: A query vector.
    """
    if backend_type == Backend.HNSW:
        # Don't test insert for HNSW backend
        return
    new_item = ["item101"]
    new_vector = query_vector

    nearest_instance.insert(new_item, new_vector[None, :])

    results = nearest_instance.query(query_vector, k=1)
    returned_item = results[0][0][0]

    assert returned_item == "item101"


def test_nearest_delete(nearest_instance: Nearest, items: list[str], vectors: np.ndarray) -> None:
    """
    Test Nearest.delete method by verifying that the vector for a deleted item is not returned in subsequent queries.

    :param nearest_instance: A Nearest instance.
    :param items: List of item names.
    :param vectors: Array of vectors corresponding to items.
    """
    # Get the vector corresponding to "item2"
    item2_index = items.index("item2")
    item2_vector = vectors[item2_index]

    # Delete "item2" from the Nearest instance
    nearest_instance.delete(["item2"])

    # Ensure "item2" is no longer in the items list
    assert "item2" not in nearest_instance.items

    # Query using the vector of "item2"
    results = nearest_instance.query(item2_vector, k=5)  # Adjust k as needed
    returned_items = [item for item, _ in results[0]]

    # Check that "item2" is not in the results
    assert "item2" not in returned_items


def test_nearest_save_and_load(tmp_path: Path, nearest_instance: Nearest) -> None:
    """
    Test Nearest.save and Nearest.load methods.

    :param tmp_path: Temporary directory provided by pytest.
    :param nearest_instance: A Nearest instance.
    """
    save_path = tmp_path / "nearest_data"
    nearest_instance.save(save_path)

    Nearest.load(save_path)


def test_nearest_insert_duplicate(nearest_instance: Nearest, query_vector: np.ndarray) -> None:
    """
    Test that Nearest.insert raises ValueError when inserting duplicate items.

    :param nearest_instance: A Nearest instance.
    :raises ValueError: If inserting items that already exist.
    """
    new_items = ["item1"]
    new_vector = query_vector

    with pytest.raises(ValueError):
        nearest_instance.insert(new_items, new_vector[None, :])


def test_nearest_delete_nonexistent(nearest_instance: Nearest) -> None:
    """
    Test that Nearest.delete raises ValueError when deleting non-existent items.

    :param nearest_instance: A Nearest instance.
    :raises ValueError: If deleting items that do not exist.
    """
    with pytest.raises(ValueError):
        nearest_instance.delete(["item102"])


def test_nearest_insert_mismatched_lengths(nearest_instance: Nearest, query_vector: np.ndarray) -> None:
    """
    Test that Nearest.insert raises ValueError when tokens and vectors lengths do not match.

    :param nearest_instance: A Nearest instance.
    :raises ValueError: If tokens and vectors lengths differ.
    """
    new_items = ["item4", "item5"]
    new_vector = query_vector

    with pytest.raises(ValueError):
        nearest_instance.insert(new_items, new_vector[None, :])


def test_nearest_insert_wrong_dimension(nearest_instance: Nearest) -> None:
    """
    Test that Nearest.insert raises ValueError when inserting vectors of incorrect dimension.

    :param nearest_instance: A Nearest instance.
    :raises ValueError: If vectors have wrong dimension.
    """
    new_item = ["item102"]
    new_vector = np.array([[0.5, 0.5, 0.5]])

    with pytest.raises(ValueError):
        nearest_instance.insert(new_item, new_vector)
