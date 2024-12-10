from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vicinity import Vicinity
from vicinity.datatypes import Backend

BackendType = tuple[Backend, str]


def test_vicinity_init(backend_type: BackendType, items: list[str], vectors: np.ndarray) -> None:
    """
    Test Vicinity.init.

    :param backend_type: The backend type to use.
    :param items: A list of item names.
    :param vectors: An array of vectors.
    """
    backend = backend_type[0]
    vicinity = Vicinity.from_vectors_and_items(vectors, items, backend_type=backend)
    assert len(vicinity) == len(items)
    assert vicinity.items == items
    assert vicinity.dim == vectors.shape[1]

    vectors = np.random.default_rng(42).random((len(items) - 1, 5))

    with pytest.raises(ValueError):
        vicinity = Vicinity.from_vectors_and_items(vectors, items, backend_type=backend)


def test_vicinity_from_vectors_and_items(backend_type: BackendType, items: list[str], vectors: np.ndarray) -> None:
    """
    Test Vicinity.from_vectors_and_items.

    :param backend_type: The backend type to use.
    :param items: A list of item names.
    :param vectors: An array of vectors.
    """
    backend = backend_type[0]
    vicinity = Vicinity.from_vectors_and_items(vectors, items, backend_type=backend)

    assert len(vicinity) == len(items)
    assert vicinity.items == items
    assert vicinity.dim == vectors.shape[1]


def test_vicinity_query(vicinity_instance: Vicinity, query_vector: np.ndarray) -> None:
    """
    Test Vicinity.query.

    :param vicinity_instance: A Vicinity instance.
    :param query_vector: A query vector.
    """
    results = vicinity_instance.query(query_vector, k=2)

    assert len(results) == 1


def test_vicinity_query_threshold(vicinity_instance: Vicinity, query_vector: np.ndarray) -> None:
    """
    Test Vicinity.query_threshold method.

    :param vicinity_instance: A Vicinity instance.
    :param query_vector: A query vector.
    """
    results = vicinity_instance.query_threshold(query_vector, threshold=0.7)

    assert len(results) >= 1


def test_vicinity_insert(vicinity_instance: Vicinity, query_vector: np.ndarray) -> None:
    """
    Test Vicinity.insert method.

    :param backend_type: The backend type to use.
    :param vicinity_instance: A Vicinity instance.
    :param query_vector: A query vector.
    """
    if vicinity_instance.backend.backend_type in {Backend.HNSW, Backend.ANNOY, Backend.PYNNDESCENT}:
        # Skip insert for HNSW or Annoy backends.
        return
    new_item = ["item10001"]
    new_vector = query_vector
    vicinity_instance.insert(new_item, new_vector[None, :])

    results = vicinity_instance.query(query_vector, k=50)

    returned_items = [item for item, _ in results[0]]
    assert "item10001" in returned_items


def test_vicinity_delete(vicinity_instance: Vicinity, items: list[str], vectors: np.ndarray) -> None:
    """
    Test Vicinity.delete method by verifying that the vector for a deleted item is not returned in subsequent queries.

    :param vicinity_instance: A Vicinity instance.
    :param items: List of item names.
    :param vectors: Array of vectors corresponding to items.
    """
    if vicinity_instance.backend.backend_type != Backend.BASIC:
        # Skip delete for non-basic backends
        return

    # Get the vector corresponding to "item2"
    item2_index = items.index("item2")
    item2_vector = vectors[item2_index]

    # Delete "item2" from the Vicinity instance
    vicinity_instance.delete(["item2"])

    # Ensure "item2" is no longer in the items list
    assert "item2" not in vicinity_instance.items

    # Query using the vector of "item2"
    results = vicinity_instance.query(item2_vector, k=5)
    returned_items = [item for item, _ in results[0]]

    # Check that "item2" is not in the results
    assert "item2" not in returned_items


def test_vicinity_save_and_load(tmp_path: Path, vicinity_instance: Vicinity) -> None:
    """
    Test Vicinity.save and Vicinity.load.

    :param tmp_path: Temporary directory provided by pytest.
    :param vicinity_instance: A Vicinity instance.
    """
    save_path = tmp_path / "vicinity_data"
    vicinity_instance.save(save_path)
    assert vicinity_instance.vector_store is None

    v = Vicinity.load(save_path)
    assert v.vector_store is None


def test_vicinity_save_and_load_vector_store(tmp_path: Path, vicinity_instance_with_stored_vectors: Vicinity) -> None:
    """
    Test Vicinity.save and Vicinity.load.

    :param tmp_path: Temporary directory provided by pytest.
    :param vicinity_instance: A Vicinity instance.
    """
    save_path = tmp_path / "vicinity_data"
    vicinity_instance_with_stored_vectors.save(save_path)

    assert (save_path / "store").exists()
    assert (save_path / "store" / "vectors.npy").exists()

    v = Vicinity.load(save_path)
    assert v.vector_store is not None


def test_index_vector_store(vicinity_with_basic_backend: Vicinity, vectors: np.ndarray) -> None:
    """
    Index vectors in the Vicinity instance.

    :param vicinity_instance: A Vicinity instance.
    :param vectors: Array of vectors to index.
    """
    v = vicinity_with_basic_backend.get_vector_by_index(0)
    assert np.allclose(v, vectors[0])

    idx = [0, 1, 2, 3, 4, 10]
    v = vicinity_with_basic_backend.get_vector_by_index(idx)
    assert np.allclose(v, vectors[idx])

    with pytest.raises(ValueError):
        vicinity_with_basic_backend.get_vector_by_index([10_000])

    with pytest.raises(ValueError):
        vicinity_with_basic_backend.get_vector_by_index([-1])


def test_vicinity_insert_duplicate(vicinity_instance: Vicinity, query_vector: np.ndarray) -> None:
    """
    Test that Vicinity.insert raises ValueError when inserting duplicate items.

    :param vicinity_instance: A Vicinity instance.
    :raises ValueError: If inserting items that already exist.
    """
    new_items = ["item1"]
    new_vector = query_vector

    with pytest.raises(ValueError):
        vicinity_instance.insert(new_items, new_vector[None, :])


def test_vicinity_delete_nonexistent(vicinity_instance: Vicinity) -> None:
    """
    Test that Vicinity.delete raises ValueError when deleting non-existent items.

    :param vicinity_instance: A Vicinity instance.
    :raises ValueError: If deleting items that do not exist.
    """
    if vicinity_instance.backend.backend_type != Backend.BASIC:
        # Skip delete for non-basic backends
        return
    with pytest.raises(ValueError):
        vicinity_instance.delete(["item10002"])


def test_vicinity_insert_mismatched_lengths(vicinity_instance: Vicinity, query_vector: np.ndarray) -> None:
    """
    Test that Vicinity.insert raises ValueError when tokens and vectors lengths do not match.

    :param vicinity_instance: A Vicinity instance.
    :raises ValueError: If tokens and vectors lengths differ.
    """
    new_items = ["item10002", "item10003"]
    new_vector = query_vector

    with pytest.raises(ValueError):
        vicinity_instance.insert(new_items, new_vector[None, :])


def test_vicinity_insert_wrong_dimension(vicinity_instance: Vicinity) -> None:
    """
    Test that Vicinity.insert raises ValueError when inserting vectors of incorrect dimension.

    :param vicinity_instance: A Vicinity instance.
    :raises ValueError: If vectors have wrong dimension.
    """
    new_item = ["item10002"]
    new_vector = np.array([[0.5, 0.5, 0.5]])

    with pytest.raises(ValueError):
        vicinity_instance.insert(new_item, new_vector)


def test_vicinity_delete_and_query(vicinity_instance: Vicinity, items: list[str], vectors: np.ndarray) -> None:
    """
    Test Vicinity's delete and query methods together to ensure that indices are correctly handled after deletions.

    :param vicinity_instance: A Vicinity instance.
    :param items: List of item names.
    :param vectors: Array of vectors corresponding to items.
    """
    if vicinity_instance.backend.backend_type != Backend.BASIC:
        # Skip delete for non-basic backends
        return

    # Delete some items from the Vicinity instance
    items_to_delete = ["item2", "item4", "item6"]
    vicinity_instance.delete(items_to_delete)

    # Ensure the items are no longer in the items list
    for item in items_to_delete:
        assert item not in vicinity_instance.items

    # Query using a vector of an item that wasn't deleted
    item3_index = items.index("item3")
    item3_vector = vectors[item3_index]

    results = vicinity_instance.query(item3_vector, k=10)
    returned_items = [item for item, _ in results[0]]

    # Check that the queried item is in the results
    assert "item3" in returned_items


def test_vicinity_evaluate(vicinity_instance: Vicinity, vectors: np.ndarray) -> None:
    """
    Test the evaluate method of the Vicinity instance.

    :param vicinity_instance: A Vicinity instance.
    :param vectors: The full dataset vectors used to build the index.
    """
    query_vectors = vectors[:10]
    qps, recall = vicinity_instance.evaluate(vectors, query_vectors)

    # Ensure the QPS and recall values are within valid ranges
    assert qps > 0
    assert 0 <= recall <= 1

    # Test with an unsupported metric
    vicinity_instance.backend.arguments.metric = "manhattan"
    with pytest.raises(ValueError):
        vicinity_instance.evaluate(vectors, query_vectors)
