from __future__ import annotations

import numpy as np
import pytest

from vicinity import Vicinity
from vicinity.datatypes import Backend

random_gen = np.random.default_rng(42)

_faiss_index_types = [
    "flat",
    "ivf",
    "hnsw",
    "lsh",
    "scalar",
    "pq",
    "ivf_scalar",
    "ivfpq",
    "ivfpqr",
]


@pytest.fixture(scope="session")
def items() -> list[str]:
    """Fixture providing a list of item names."""
    return [f"item{i}" for i in range(1, 10001)]


@pytest.fixture(scope="session")
def vectors() -> np.ndarray:
    """Fixture providing an array of vectors corresponding to items."""
    return random_gen.random((10000, 8))


@pytest.fixture(scope="session")
def query_vector() -> np.ndarray:
    """Fixture providing a query vector."""
    return random_gen.random(8)


BACKEND_PARAMS = [(Backend.FAISS, index_type) for index_type in _faiss_index_types] + [
    (Backend.BASIC, None),
    (Backend.HNSW, None),
    (Backend.ANNOY, None),
    (Backend.PYNNDESCENT, None),
    (Backend.USEARCH, None),
    (Backend.VOYAGER, None),
]


# Create human-readable ids for each backend type
BACKEND_IDS = [f"{backend.name}-{index_type}" if index_type else backend.name for backend, index_type in BACKEND_PARAMS]


@pytest.fixture(params=BACKEND_PARAMS)
def backend_type(request: pytest.FixtureRequest) -> Backend:
    """Fixture parametrizing over all backend types defined in Backend."""
    return request.param


@pytest.fixture(params=BACKEND_PARAMS, ids=BACKEND_IDS)
def vicinity_instance(request: pytest.FixtureRequest, items: list[str], vectors: np.ndarray) -> Vicinity:
    """Fixture providing a Vicinity instance for each backend type."""
    backend_type, index_type = request.param
    # Handle FAISS backend with specific FAISS index types
    if backend_type == Backend.FAISS:
        if index_type in ("pq", "ivfpq", "ivfpqr"):
            # Use smaller values for pq indexes since the dataset is small
            return Vicinity.from_vectors_and_items(
                vectors,
                items,
                backend_type=backend_type,
                index_type=index_type,
                m=2,
                nbits=4,
            )
        else:
            return Vicinity.from_vectors_and_items(
                vectors,
                items,
                backend_type=backend_type,
                index_type=index_type,
                nlist=2,
                nbits=32,
            )

    return Vicinity.from_vectors_and_items(vectors, items, backend_type=backend_type)


@pytest.fixture(params=BACKEND_PARAMS, ids=BACKEND_IDS)
def vicinity_instance_with_stored_vectors(
    request: pytest.FixtureRequest, items: list[str], vectors: np.ndarray
) -> Vicinity:
    """Fixture providing a Vicinity instance for each backend type."""
    backend_type, index_type = request.param
    # Handle FAISS backend with specific FAISS index types
    if backend_type == Backend.FAISS:
        if index_type in ("pq", "ivfpq", "ivfpqr"):
            # Use smaller values for pq indexes since the dataset is small
            return Vicinity.from_vectors_and_items(
                vectors, items, backend_type=backend_type, index_type=index_type, m=2, nbits=4, store_vectors=True
            )
        else:
            return Vicinity.from_vectors_and_items(
                vectors, items, backend_type=backend_type, index_type=index_type, nlist=2, nbits=32, store_vectors=True
            )

    return Vicinity.from_vectors_and_items(vectors, items, backend_type=backend_type, store_vectors=True)


@pytest.fixture()
def vicinity_with_basic_backend(vectors: np.ndarray, items: list[str]) -> Vicinity:
    """Fixture providing a BasicBackend instance."""
    return Vicinity.from_vectors_and_items(vectors, items, backend_type=Backend.BASIC, store_vectors=True)
