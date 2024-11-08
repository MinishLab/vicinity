"""Unit tests for the Nearest class using pytest."""

from __future__ import annotations

import numpy as np
import pytest

from nearest import Nearest
from nearest.backends import get_backend_class
from nearest.datatypes import Backend


def test_nearest_init_mismatched_lengths(backend_type: Backend, items: list[str]) -> None:
    """
    Test that Nearest.__init__ raises ValueError when items and backend have different lengths.

    :param backend_type: The backend type to use (BASIC or HNSW).
    :param items: A list of item names.
    :raises ValueError: If items and backend lengths differ.
    """
    vectors = np.random.rand(len(items) - 1, 5)  # Mismatched length

    backend_cls = get_backend_class(backend_type)
    arguments = backend_cls.argument_class()
    backend = backend_cls.from_vectors(vectors, **arguments.dict())

    with pytest.raises(ValueError) as exc_info:
        Nearest(items, backend)

    assert "Your vector space and list of items are not the same length" in str(exc_info.value)


def test_nearest_init_success(backend_type: Backend, items: list[str], vectors: np.ndarray) -> None:
    """
    Test Nearest.__init__ with matching items and backend lengths.

    :param backend_type: The backend type to use (BASIC or HNSW).
    :param items: A list of item names.
    :param vectors: An array of vectors.
    """
    backend_cls = get_backend_class(backend_type)
    arguments = backend_cls.argument_class()
    backend = backend_cls.from_vectors(vectors, **arguments.dict())

    nearest = Nearest(items, backend)
    assert len(nearest) == len(items)
    assert nearest.items == items
    assert nearest.dim == vectors.shape[1]


def test_nearest_from_vectors_and_items(backend_type: Backend, items: list[str], vectors: np.ndarray) -> None:
    """
    Test Nearest.from_vectors_and_items class method.

    :param backend_type: The backend type to use (BASIC or HNSW).
    :param items: A list of item names.
    :param vectors: An array of vectors.
    """
    nearest = Nearest.from_vectors_and_items(vectors, items, backend_type=backend_type)

    assert len(nearest) == len(items)
    assert nearest.items == items
    assert nearest.dim == vectors.shape[1]


def test_nearest_query(nearest_instance: Nearest) -> None:
    """
    Test Nearest.query method.

    :param nearest_instance: A Nearest instance.
    """
    query_vector = np.array([1.0, 0.0])

    results = nearest_instance.query(query_vector, k=2)

    expected_items = ["item1", "item3"]
    returned_items = [item for item, _ in results[0]]

    assert returned_items == expected_items


def test_nearest_query_threshold(nearest_instance: Nearest) -> None:
    """
    Test Nearest.query_threshold method.

    :param nearest_instance: A Nearest instance.
    """
    query_vector = np.array([1.0, 0.0])

    results = nearest_instance.query_threshold(query_vector, threshold=0.7)

    expected_items = ["item1", "item3"]
    returned_items = results[0]

    assert returned_items == expected_items


def test_nearest_insert(nearest_instance: Nearest) -> None:
    """
    Test Nearest.insert method.

    :param nearest_instance: A Nearest instance.
    """
    new_items = ["item4"]
    new_vectors = np.array([[0.5, 0.5]])

    nearest_instance.insert(new_items, new_vectors)

    assert len(nearest_instance) == 4
    assert nearest_instance.items == ["item1", "item2", "item3", "item4"]

    query_vector = np.array([0.5, 0.5])

    results = nearest_instance.query(query_vector, k=1)
    returned_item = results[0][0][0]

    assert returned_item == "item4"


def test_nearest_delete(nearest_instance: Nearest) -> None:
    """
    Test Nearest.delete method.

    :param nearest_instance: A Nearest instance.
    """
    nearest_instance.delete(["item2"])

    assert len(nearest_instance) == 2
    assert nearest_instance.items == ["item1", "item3"]

    query_vector = np.array([0.0, 1.0])

    results = nearest_instance.query(query_vector, k=2)
    returned_items = [item for item, _ in results[0]]

    assert "item2" not in returned_items


def test_nearest_save_and_load(tmp_path, nearest_instance: Nearest) -> None:
    """
    Test Nearest.save and Nearest.load methods.

    :param tmp_path: Temporary directory provided by pytest.
    :param nearest_instance: A Nearest instance.
    """
    save_path = tmp_path / "nearest_data"
    nearest_instance.save(save_path)

    loaded_nearest = Nearest.load(save_path)

    assert len(loaded_nearest) == len(nearest_instance)
    assert loaded_nearest.items == nearest_instance.items
    assert loaded_nearest.dim == nearest_instance.dim

    query_vector = np.array([1.0, 0.0])
    results = loaded_nearest.query(query_vector, k=1)
    returned_item = results[0][0][0]

    assert returned_item == "item1"


def test_nearest_insert_duplicate(nearest_instance: Nearest) -> None:
    """
    Test that Nearest.insert raises ValueError when inserting duplicate items.

    :param nearest_instance: A Nearest instance.
    :raises ValueError: If inserting items that already exist.
    """
    new_items = ["item1"]
    new_vectors = np.array([[0.5, 0.5]])

    with pytest.raises(ValueError) as exc_info:
        nearest_instance.insert(new_items, new_vectors)

    assert "Token item1 is already in the vector space." in str(exc_info.value)


def test_nearest_delete_nonexistent(nearest_instance: Nearest) -> None:
    """
    Test that Nearest.delete raises ValueError when deleting non-existent items.

    :param nearest_instance: A Nearest instance.
    :raises ValueError: If deleting items that do not exist.
    """
    with pytest.raises(ValueError) as exc_info:
        nearest_instance.delete(["item4"])

    assert "Token item4 was not in the vector space." in str(exc_info.value)


def test_nearest_insert_mismatched_lengths(nearest_instance: Nearest) -> None:
    """
    Test that Nearest.insert raises ValueError when tokens and vectors lengths do not match.

    :param nearest_instance: A Nearest instance.
    :raises ValueError: If tokens and vectors lengths differ.
    """
    new_items = ["item4", "item5"]
    new_vectors = np.array([[0.5, 0.5]])

    with pytest.raises(ValueError) as exc_info:
        nearest_instance.insert(new_items, new_vectors)

    assert "Your tokens and vectors are not the same length" in str(exc_info.value)


def test_nearest_insert_wrong_dimension(nearest_instance: Nearest) -> None:
    """
    Test that Nearest.insert raises ValueError when inserting vectors of incorrect dimension.

    :param nearest_instance: A Nearest instance.
    :raises ValueError: If vectors have wrong dimension.
    """
    new_items = ["item4"]
    new_vectors = np.array([[0.5, 0.5, 0.5]])  # Incorrect dimension

    with pytest.raises(ValueError) as exc_info:
        nearest_instance.insert(new_items, new_vectors)

    assert "The inserted vectors must have the same dimension as the backend." in str(exc_info.value)


def test_nearest_save_invalid_path(tmp_path, nearest_instance: Nearest) -> None:
    """
    Test that Nearest.save raises ValueError when given an invalid path.

    :param tmp_path: Temporary directory provided by pytest.
    :param nearest_instance: A Nearest instance.
    :raises ValueError: If path is not a directory.
    """
    invalid_path = tmp_path / "file.txt"
    invalid_path.touch()

    with pytest.raises(ValueError) as exc_info:
        nearest_instance.save(invalid_path)

    assert f"Path {invalid_path} should be a directory." in str(exc_info.value)


def test_nearest_load_invalid_path(tmp_path) -> None:
    """
    Test that Nearest.load raises FileNotFoundError when loading from an invalid path.

    :param tmp_path: Temporary directory provided by pytest.
    :raises FileNotFoundError: If path does not exist.
    """
    invalid_path = tmp_path / "nonexistent_directory"

    with pytest.raises(FileNotFoundError):
        Nearest.load(invalid_path)
