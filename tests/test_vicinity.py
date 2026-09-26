from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from orjson import JSONEncodeError

from vicinity import Vicinity
from vicinity.backends.faiss import FaissBackend
from vicinity.datatypes import Backend
from vicinity.utils import Metric, normalize

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
    vicinity = Vicinity.from_vectors_and_items(vectors, items, backend_type=backend, metric="cosine")

    assert len(vicinity) == len(items)
    assert vicinity.items == items
    assert vicinity.dim == vectors.shape[1]
    assert vicinity.metric is Metric.COSINE


def test_vicinity_query(vicinity_instance: Vicinity, query_vector: np.ndarray) -> None:
    """
    Test Vicinity.query.

    :param vicinity_instance: A Vicinity instance.
    :param query_vector: A query vector.
    """
    results = vicinity_instance.query(query_vector, k=2)

    assert len(results) == 1

    results = vicinity_instance.query(np.stack([query_vector, query_vector]), k=2)

    assert results[0] == results[1]


def test_vicinity_query_threshold(vicinity_instance: Vicinity, query_vector: np.ndarray) -> None:
    """
    Test Vicinity.query_threshold method.

    :param vicinity_instance: A Vicinity instance.
    :param query_vector: A query vector.
    """
    results = vicinity_instance.query_threshold(query_vector, threshold=0.7)

    assert len(results) >= 1

    results = vicinity_instance.query_threshold(np.stack([query_vector, query_vector]), threshold=0.7)

    assert results[0] == results[1]


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
    assert v.query(np.ones(v.dim), k=5)[0]


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


def test_vicinity_save_and_load_non_serializable_items(
    tmp_path: Path, non_serializable_items: list[str], vectors: np.ndarray
) -> None:
    """
    Test Vicinity.save and Vicinity.load with non-serializable items.

    :param tmp_path: Temporary directory provided by pytest.
    :param non_serializable_items: A list of non-serializable items.
    """
    vicinity = Vicinity.from_vectors_and_items(vectors=vectors, items=non_serializable_items)
    save_path = tmp_path / "vicinity_data"
    with pytest.raises(JSONEncodeError):
        vicinity.save(save_path)


def test_index_vector_store(vicinity_with_basic_backend_and_store: Vicinity, vectors: np.ndarray) -> None:
    """
    Index vectors in the Vicinity instance.

    :param vicinity_instance: A Vicinity instance.
    :param vectors: Array of vectors to index.
    """
    v = vicinity_with_basic_backend_and_store.get_vector_by_index(0)
    assert np.allclose(v, vectors[0])

    idx = [0, 1, 2, 3, 4, 10]
    v = vicinity_with_basic_backend_and_store.get_vector_by_index(idx)
    assert np.allclose(v, vectors[idx])

    with pytest.raises(ValueError):
        vicinity_with_basic_backend_and_store.get_vector_by_index([10_000])

    with pytest.raises(ValueError):
        vicinity_with_basic_backend_and_store.get_vector_by_index([-1])


def test_vicinity_insert_duplicate(items: list[str], vicinity_instance: Vicinity, query_vector: np.ndarray) -> None:
    """
    Test that Vicinity.insert raises ValueError when inserting duplicate items.

    :param vicinity_instance: A Vicinity instance.
    :raises ValueError: If inserting items that already exist.
    """
    new_vector = query_vector

    with pytest.raises(ValueError):
        vicinity_instance.insert(items[0], new_vector[None, :])


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


def test_vicinity_insert_with_store(vicinity_with_basic_backend_and_store: Vicinity) -> None:
    """
    Test that Vicinity.insert raises ValueError when trying to insert vectors into a Vicinity instance with stored vectors.

    :param vicinity_with_basic_backend_and_store: A Vicinity instance with stored vectors.
    """
    new_item = ["item10002"]
    new_vector = np.full((1, vicinity_with_basic_backend_and_store.dim), 0.5)

    vicinity_with_basic_backend_and_store.insert(new_item, new_vector)
    assert vicinity_with_basic_backend_and_store.vector_store is not None
    assert len(vicinity_with_basic_backend_and_store) == len(vicinity_with_basic_backend_and_store.vector_store)


def test_vicinity_delete_with_store(vicinity_with_basic_backend_and_store: Vicinity) -> None:
    """
    Test Vicinity.delete method by verifying that the vector for a deleted item is not returned in subsequent queries.

    :param vicinity_with_basic_backend_and_store: A Vicinity instance.
    """
    assert vicinity_with_basic_backend_and_store.vector_store is not None
    # Delete "item2" from the Vicinity instance
    vicinity_with_basic_backend_and_store.delete(["item2"])

    # Ensure "item2" is no longer in the items list
    assert "item2" not in vicinity_with_basic_backend_and_store.items
    assert len(vicinity_with_basic_backend_and_store) == len(vicinity_with_basic_backend_and_store.vector_store)


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
    non_existing_items_indices = [0, 1, 2]
    items_to_delete = [items[i] for i in non_existing_items_indices]
    vicinity_instance.delete(items_to_delete)

    # Ensure the items are no longer in the items list
    for item in items_to_delete:
        assert item not in vicinity_instance.items

    # Query using a vector of an item that wasn't deleted
    existing_item_index = 3
    item3_vector = vectors[existing_item_index]

    results = vicinity_instance.query(item3_vector, k=10)
    returned_items = [item for item, _ in results[0]]

    # Check that the queried item is in the results
    assert items[existing_item_index] in returned_items


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


@pytest.mark.parametrize("metric", ["hamming", "tanimoto"])
def test_vicinity_usearch_binary_metrics(tmp_path: Path, metric: str) -> None:
    """
    Test that the usearch backend supports binary metrics on bit-packed vectors.

    :param tmp_path: Temporary directory for saving and loading.
    :param metric: The binary metric to use.
    """
    bits = np.random.default_rng(42).integers(0, 2, size=(50, 64)).astype(bool)
    packed = np.packbits(bits, axis=1)
    items = [str(i) for i in range(len(bits))]
    vicinity = Vicinity.from_vectors_and_items(packed, items, backend_type=Backend.USEARCH, metric=metric)
    assert vicinity.dim == packed.shape[1]

    if metric == "hamming":
        expected = (bits[0] != bits).sum(axis=1)
    else:
        expected = 1 - (bits[0] & bits).sum(axis=1) / (bits[0] | bits).sum(axis=1)
    ((_, distance),) = vicinity.query(packed[:1], k=1)[0]
    ((_, farthest),) = vicinity.query(packed[:1], k=len(items))[0][-1:]
    assert distance == pytest.approx(0)
    assert farthest == pytest.approx(expected.max(), abs=1e-5)

    vicinity.insert(["new"], packed[:1])
    vicinity.save(tmp_path / "binary")
    assert Vicinity.load(tmp_path / "binary").query(packed[:1], k=3)[0] == vicinity.query(packed[:1], k=3)[0]

    with pytest.raises(ValueError, match="bit-packed"):
        Vicinity.from_vectors_and_items(bits.astype(np.float32), items, backend_type=Backend.USEARCH, metric=metric)


@pytest.mark.parametrize("metric", ["cosine", "euclidean"])
@pytest.mark.parametrize(
    "backend_type,kwargs,atol",
    [
        (Backend.BASIC, {}, 1e-4),
        (Backend.HNSW, {}, 1e-4),
        (Backend.PYNNDESCENT, {}, 1e-4),
        (Backend.VOYAGER, {}, 1e-4),
        (Backend.FAISS, {"index_type": "flat"}, 1e-4),
        # Clusters of about 10 vectors, so a query for 100 neighbours gets padded results.
        (Backend.FAISS, {"index_type": "ivf", "nlist": 1000}, 1e-4),
        (Backend.FAISS, {"index_type": "hnsw"}, 1e-4),
        # Scalar quantization makes distances approximate.
        (Backend.FAISS, {"index_type": "scalar"}, 0.05),
        (Backend.FAISS, {"index_type": "ivf_scalar", "nlist": 50}, 0.05),
    ],
)
def test_backend_distances_match_metric(
    backend_type: Backend, kwargs: dict, atol: float, metric: str, vectors: np.ndarray, query_vector: np.ndarray
) -> None:
    """Backends return true cosine or Euclidean distances without padding; exact backends return every close item."""
    # Centred and scaled, so distances go beyond 0.5 (cosine) and 1 (Euclidean), where FAISS range radii differ.
    vectors, query = 2 * (vectors - 0.5), 2 * (query_vector - 0.5)
    vicinity = Vicinity.from_vectors_and_items(
        vectors, list(range(len(vectors))), backend_type=backend_type, metric=metric, **kwargs
    )

    def distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if metric == "cosine":
            return 1 - np.sum(normalize(a) * normalize(b), axis=-1)
        return np.linalg.norm(a - b, axis=-1)

    expected = distance(vectors, query)
    # Halfway between the 11th and 12th closest items, so no item sits on the boundary.
    threshold = float(np.sort(expected)[10:12].mean())
    for result in (vicinity.query(query, k=100)[0], vicinity.query_threshold(query, threshold)[0]):
        items, distances = zip(*result)
        assert np.allclose(distances, expected[list(items)], atol=atol)
    # Every stored vector must come back with its true distance, which catches items mapped to the wrong vector.
    items, distances = zip(*(result[0] for result in vicinity.query(vectors, k=1)))
    assert np.allclose(distances, distance(vectors, vectors[list(items)]), atol=atol)
    if backend_type == Backend.BASIC or kwargs.get("index_type") == "flat":
        for limit in (threshold, float(np.median(expected))):
            returned = {item for item, _ in vicinity.query_threshold(query, limit)[0]}
            assert returned == set(np.flatnonzero(expected < limit).tolist())


@pytest.mark.parametrize(
    "backend_type,kwargs",
    [
        (Backend.BASIC, {}),
        (Backend.FAISS, {"index_type": "flat"}),
        (Backend.FAISS, {"index_type": "hnsw"}),
        (Backend.FAISS, {"index_type": "scalar"}),
        (Backend.FAISS, {"index_type": "ivf_scalar", "nlist": 1}),
        (Backend.FAISS, {"index_type": "pq", "m": 1, "nbits": 3}),
        (Backend.FAISS, {"index_type": "ivfpq", "nlist": 1, "m": 1, "nbits": 3}),
        (Backend.FAISS, {"index_type": "ivfpqr", "nlist": 1, "m": 1, "nbits": 3, "refine_nbits": 3}),
    ],
)
def test_cosine_distance_to_zero_vector(tmp_path: Path, backend_type: Backend, kwargs: dict) -> None:
    """Zero vectors have cosine distance 1 to everything: built, inserted, reloaded and as queries."""
    # Apart from the zero vectors, every vector points away from the query, so nothing falls within the threshold.
    vectors = np.array([[0.0, 0.0]] + [[-1.0, 0.1 * i] for i in range(15)], dtype=np.float32)
    vicinity = Vicinity.from_vectors_and_items(vectors, list(range(len(vectors))), backend_type=backend_type, **kwargs)
    vicinity.insert([16], np.zeros((1, 2), dtype=np.float32))
    vicinity.save(tmp_path / "vicinity")
    vicinity = Vicinity.load(tmp_path / "vicinity")
    query = np.array([1.0, 0.0], dtype=np.float32)
    distances = dict(vicinity.query(query, k=len(vectors) + 1)[0])
    assert [distances[0], distances[16]] == pytest.approx([1.0, 1.0], abs=0.01)
    assert vicinity.query_threshold(query, threshold=0.75)[0] == []
    zero_query_distances = [distance for _, distance in vicinity.query(np.zeros(2, dtype=np.float32), k=5)[0]]
    assert np.allclose(zero_query_distances, 1.0, atol=0.01)


def test_faiss_lsh_returns_hamming_distances(vectors: np.ndarray, query_vector: np.ndarray) -> None:
    """LSH distances are FAISS's Hamming distances, which cannot be converted to cosine distances."""
    vicinity = Vicinity.from_vectors_and_items(
        vectors, list(range(len(vectors))), backend_type=Backend.FAISS, index_type="lsh", nbits=32
    )
    assert isinstance(vicinity.backend, FaissBackend)
    hamming, _ = vicinity.backend.index.search(normalize(query_vector)[None], 10)
    assert [distance for _, distance in vicinity.query(query_vector, k=10)[0]] == hamming[0].tolist()
